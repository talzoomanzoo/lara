import json
import argparse
import os
import glob
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Older Qwen HF tokenizers lack `all_special_tokens_extended`, which vLLM reads.
# Apply before importing vLLM in this process (EngineCore subprocesses rely on
# PYTHONPATH compat_site/sitecustomize — see `_prepend_compat_site_to_pythonpath()`).
try:
    from tokenizer_extended_compat import (
        apply_transformers_special_tokens_extended_getattr_compat,
    )

    apply_transformers_special_tokens_extended_getattr_compat()
except ImportError:
    pass

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import copy
import json as _json
from collections.abc import Mapping


class VLLMCompatibility:
    """Compatibility setup for the model and worker processes."""

    @staticmethod
    def _maybe_patch_model_dir_for_vllm(model_path: str) -> str:
        """
        Apply small `config.json` fixes for vLLM / Transformers quirks:

        * Some Qwen2-style configs use `rope_scaling` without a `factor` key; vLLM
          asserts `"factor" in rope_scaling`. We add a no-op linear scaling
          (factor=1.0) when needed.
        * OLMo3 checkpoints may store `rope_parameters.beta_{fast,slow}` as JSON
          integers; Transformers expects floats and logs validation noise otherwise.
        """
        try:
            mp = Path(model_path)
            cfg_path = mp / "config.json"
            if not mp.exists() or not mp.is_dir() or not cfg_path.exists():
                return model_path

            with open(cfg_path, "r") as f:
                cfg_json = json.load(f)

            rope_scaling = cfg_json.get("rope_scaling")
            # `null` in config.json is expanded by Transformers to e.g.
            # {"rope_type": "default", "rope_theta": ...} without "factor", which vLLM rejects.
            fix_rope_scaling = rope_scaling is None or (
                isinstance(rope_scaling, dict) and "factor" not in rope_scaling
            )

            rp = cfg_json.get("rope_parameters")
            fix_rope_params = False
            if isinstance(rp, dict):
                for key in ("beta_fast", "beta_slow"):
                    if key in rp and isinstance(rp.get(key), int):
                        fix_rope_params = True
                        break

            if not fix_rope_scaling and not fix_rope_params:
                return model_path

            patched_dir = Path(tempfile.mkdtemp(prefix="vllm_model_patched_"))
            for p in mp.iterdir():
                if p.name == "config.json":
                    continue
                try:
                    os.symlink(p, patched_dir / p.name)
                except Exception:
                    return model_path

            if fix_rope_scaling:
                cfg_json["rope_scaling"] = {"type": "linear", "factor": 1.0}
            if fix_rope_params and isinstance(cfg_json.get("rope_parameters"), dict):
                rp2 = cfg_json["rope_parameters"]
                for key in ("beta_fast", "beta_slow"):
                    if key in rp2 and isinstance(rp2.get(key), int):
                        rp2[key] = float(rp2[key])

            with open(patched_dir / "config.json", "w") as f:
                json.dump(cfg_json, f)

            reasons = []
            if fix_rope_scaling:
                reasons.append("rope_scaling (add factor)")
            if fix_rope_params:
                reasons.append("rope_parameters (int→float)")
            print(
                f"[warn] Patched model config for vLLM ({', '.join(reasons)}).\n"
                f"       original: {model_path}\n"
                f"       patched:  {str(patched_dir)}"
            )
            return str(patched_dir)
        except Exception:
            return model_path


    @staticmethod
    def _patch_vllm_get_cached_tokenizer_safe() -> None:
        """
        vLLM wraps HF tokenizers with get_cached_tokenizer for speed. Some tokenizers
        lack `all_special_tokens_extended`, which older vLLM paths assumed.

        vLLM 0.19+ defines get_cached_tokenizer in `vllm.tokenizers.hf`; older
        releases exposed it on `vllm.transformers_utils.tokenizer`.
        """
        mod = None
        try:
            from vllm.tokenizers import hf as _hf_mod

            if getattr(_hf_mod, "get_cached_tokenizer", None) is not None:
                mod = _hf_mod
        except Exception:
            pass
        if mod is None:
            try:
                from vllm.transformers_utils import tokenizer as _legacy_mod

                if getattr(_legacy_mod, "get_cached_tokenizer", None) is not None:
                    mod = _legacy_mod
            except Exception:
                pass
        if mod is None:
            return

        _orig = mod.get_cached_tokenizer

        def _get_cached_tokenizer_safe(tokenizer):
            if not hasattr(tokenizer, "all_special_tokens_extended"):
                tokenizer.all_special_tokens_extended = list(tokenizer.all_special_tokens)
            return _orig(tokenizer)

        mod.get_cached_tokenizer = _get_cached_tokenizer_safe


    @staticmethod
    def _prepend_compat_site_to_pythonpath_for_vllm_workers() -> None:
        """vLLM v1 spawn's EngineCore needs sitecustomize; prepend compat_site early."""
        root = Path(__file__).resolve().parent
        compat_site = root / "compat_site"
        if not compat_site.is_dir():
            return
        marker = str(compat_site.resolve())
        cur = os.environ.get("PYTHONPATH", "")
        parts = [p for p in cur.split(os.pathsep) if p]
        if marker in parts:
            return
        os.environ["PYTHONPATH"] = marker if not cur else marker + os.pathsep + cur


class PromptAdapter:
    """Normalize and format prompts without changing their contents."""

    @staticmethod
    def _normalize_prompt_obj(obj):
        """Best-effort normalization across pandas/pyarrow/numpy prompt representations."""
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        # Some parquet readers may return pyarrow Scalars; convert if possible.
        try:
            import pyarrow as pa  # type: ignore
            if isinstance(obj, pa.Scalar):
                obj = obj.as_py()
        except Exception:
            pass
        # If prompt is a JSON-encoded string, try to parse it.
        if isinstance(obj, str):
            s = obj.strip()
            if s and s[0] in "[{":
                try:
                    obj = _json.loads(s)
                except Exception:
                    pass
        return obj


    @staticmethod
    def get_user_content(prompt_obj):
        prompt_obj = _normalize_prompt_obj(prompt_obj)
        if isinstance(prompt_obj, str):
            return prompt_obj
        if isinstance(prompt_obj, Mapping):
            # Sometimes prompt is a single message dict.
            return prompt_obj.get("content") or prompt_obj.get("text") or prompt_obj.get("prompt")
        if isinstance(prompt_obj, (list, tuple)) and len(prompt_obj) > 0:
            last = _normalize_prompt_obj(prompt_obj[-1])
            if isinstance(last, Mapping):
                return last.get("content") or last.get("text")
            if isinstance(last, str):
                return last
        return None


    @staticmethod
    def _coerce_messages(prompt_obj):
        """Coerce various prompt formats into a list of {role, content} dicts."""
        prompt_obj = _normalize_prompt_obj(prompt_obj)
        if prompt_obj is None:
            return []
        if isinstance(prompt_obj, (list, tuple)):
            msgs = []
            for m in prompt_obj:
                m = _normalize_prompt_obj(m)
                if isinstance(m, Mapping):
                    role = m.get("role") or "user"
                    content = m.get("content") or m.get("text") or ""
                    msgs.append({"role": role, "content": content})
                elif isinstance(m, str):
                    msgs.append({"role": "user", "content": m})
            return msgs
        if isinstance(prompt_obj, Mapping):
            role = prompt_obj.get("role") or "user"
            content = prompt_obj.get("content") or prompt_obj.get("text") or prompt_obj.get("prompt") or ""
            return [{"role": role, "content": content}]
        if isinstance(prompt_obj, str):
            return [{"role": "user", "content": prompt_obj}]
        # Unknown type: fallback to string representation.
        return [{"role": "user", "content": str(prompt_obj)}]


    @staticmethod
    def format_prompt(message, template, tokenizer):
        if not message: return ""
        question = message[-1]['content']
        if template == 'prime_sft':
            content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
            msg = [{"role": "user", "content": content}]
            if len(message) > 1 and message[0]['role'] == 'system':
                msg.insert(0, message[0])
            return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        elif template == 'own':
            return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        else:
            raise ValueError(f'Unknown template: {template}')


class EvaluationDataset:
    """Read explicitly labeled generation inputs."""

    @staticmethod
    def _read_parquet_with_fallback(path: str) -> pd.DataFrame:
        """
        Read parquet robustly across environments.

        Some parquet engines (e.g., fastparquet) can silently drop/NULL nested columns
        like our `prompt` (list-of-struct). Prefer pyarrow; fallback to datasets if available.
        """
        # 1) Prefer pyarrow engine if available.
        try:
            import pyarrow  # noqa: F401

            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            pass

        # 2) Try Hugging Face Datasets (also pyarrow-backed) if installed.
        try:
            from datasets import Dataset  # type: ignore

            return Dataset.from_parquet(path).to_pandas()
        except Exception:
            pass

        # 3) Last resort: pandas default engine.
        return pd.read_parquet(path)


    @staticmethod
    def load_evaluation_tasks(data_path: str) -> pd.DataFrame:
        """Load explicitly labeled evaluation inputs from a file or directory."""
        path = Path(data_path)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.suffix in {".jsonl", ".parquet"})
        else:
            raise FileNotFoundError(f"Evaluation input does not exist: {path}")
        if not files:
            raise ValueError(f"No .jsonl/.parquet inputs found under {path}")
        frames = []
        for file in files:
            if file.suffix == ".parquet":
                frame = _read_parquet_with_fallback(str(file))
            elif file.suffix == ".jsonl":
                frame = pd.read_json(file, lines=True)
            else:
                raise ValueError(f"Unsupported evaluation input: {file}")
            missing = {"prompt", "member"} - set(frame.columns)
            if missing:
                raise ValueError(f"{file}: missing required columns {sorted(missing)}; "
                                 "evaluation requires explicit membership labels")
            if frame["member"].isna().any() or not frame["member"].isin([0, 1, False, True]).all():
                raise ValueError(f"{file}: member must contain only explicit 0/1 labels")
            if frame["prompt"].isna().any():
                raise ValueError(f"{file}: prompt contains null values")
            if "data_source" not in frame:
                frame["data_source"] = "unknown"
            else:
                frame["data_source"] = frame["data_source"].fillna("unknown")
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        if result.empty:
            raise ValueError(f"Evaluation input contains no rows: {path}")
        return result


class TokenDistributionStatistics:
    """Calculate the existing per-token generation statistics."""

    @staticmethod
    def process_single_output(output, tokenizer, approx_mode: str = "renorm"):
        """Approximate token distributions while preserving raw likelihoods.

        renorm conditions the retained tokens on their total probability mass.
        rest represents all omitted mass as one additional category. Min-K++ uses
        log probabilities in the same distribution as its mean and variance;
        raw logprobs remain unchanged for perplexity and ordinary Min-K.
        """
        if approx_mode not in {"renorm", "rest"}:
            raise ValueError(f"Unknown approximation mode: {approx_mode}")
        raw_logprobs, mink_logprobs, entropies, mus, sigmas = [], [], [], [], []
        distributions = output.logprobs or []
        for i, token_id in enumerate(output.token_ids):
            step = distributions[i] if i < len(distributions) else None
            step = step or {}
            token = step.get(token_id)
            raw_lp = float(token.logprob) if token is not None else None
            raw_logprobs.append(raw_lp)
            finite_lps = np.array([float(p.logprob) for p in step.values()
                                   if np.isfinite(p.logprob)], dtype=float)
            if finite_lps.size == 0:
                mink_logprobs.append(None)
                entropies.append(float("nan"))
                mus.append(float("nan"))
                sigmas.append(float("nan"))
                continue
            # Stable log-sum-exp also handles tiny retained mass.
            peak = float(np.max(finite_lps))
            log_mass = peak + float(np.log(np.exp(finite_lps - peak).sum()))
            if approx_mode == "renorm":
                lps = finite_lps - log_mass
                probs = np.exp(lps)
                adjusted_lp = None if raw_lp is None else raw_lp - log_mass
            else:
                # Normalize only round-off excess above total mass 1.
                correction = max(0.0, log_mass)
                lps = finite_lps - correction
                probs = np.exp(lps)
                residual = max(0.0, 1.0 - float(probs.sum()))
                if residual > 0:
                    probs = np.append(probs, residual)
                    lps = np.append(lps, np.log(residual))
                adjusted_lp = None if raw_lp is None else raw_lp - correction
            mu = float(np.sum(probs * lps))
            variance = float(np.sum(probs * (lps - mu) ** 2))
            mink_logprobs.append(adjusted_lp)
            entropies.append(-mu)
            mus.append(mu)
            sigmas.append(float(np.sqrt(max(variance, 1e-12))))
        return {
            "generated_text": output.text,
            "logprobs": raw_logprobs,
            "mink_logprobs": mink_logprobs,
            "mus": mus,
            "sigmas": sigmas,
            "token_ids": output.token_ids,
            "entropies": entropies,
        }


class GenerationOutputStore:
    """Replace resumed generation files atomically."""

    @staticmethod
    def _write_generation_rows(path, rows):
        """Atomically replace a resumed generation file."""
        target = Path(path)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name + ".", delete=False) as out:
            tmp_path = out.name
            try:
                for row in rows:
                    out.write(json.dumps(row) + "\n")
                out.flush()
                os.fsync(out.fileno())
            except BaseException:
                os.unlink(tmp_path)
                raise
        os.replace(tmp_path, target)


class MicrobatchGenerator:
    """Run generation with the existing microbatch fallback behavior."""

    @staticmethod
    def _generate_with_microbatches(
        llm,
        prompts,
        sampling_params,
        *,
        initial_microbatch_size: int = 8,
        label: str = "",
    ):
        """
        Run vLLM generation in microbatches with fallback to smaller microbatches on failure.
        Returns a list aligned to `prompts`, with None for prompts that fail even at microbatch size 1.
        """
        if not prompts:
            return []
        results = [None] * len(prompts)
        mb = max(1, int(initial_microbatch_size))
        i = 0
        while i < len(prompts):
            chunk = prompts[i : i + mb]
            try:
                outs = llm.generate(chunk, sampling_params)
                for j, out in enumerate(outs):
                    results[i + j] = out
                i += mb
            except Exception as e:
                if mb == 1:
                    print(f"[warn] {label} generation failed for idx={i}: {e}")
                    i += 1
                    continue
                new_mb = max(1, mb // 2)
                print(
                    f"[warn] {label} generation failed for microbatch={mb} at idx={i}; "
                    f"retrying with microbatch={new_mb}. Error: {e}"
                )
                mb = new_mb
        return results


class GenerationApplication:
    """Own the command-line configuration and execute the existing workflow."""

    def __init__(self):
        self.parser = self.create_parser()

    @staticmethod
    def create_parser():
        parser = argparse.ArgumentParser(description="Generate three types of responses for each prompt: Original Greedy, Perturbed Greedy, Original Random.")
        # --- All required parameters ---
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--data_root_dir", type=str, required=True,
                            help="A labeled .parquet/.jsonl file or a directory of labeled files.")
        parser.add_argument("--output_file", type=str, required=True)
        parser.add_argument("--prompt_template", type=str, default="own")
        parser.add_argument("--perturbation_prefix", type=str, default="[SYSTEM NOTE: Please double check your reasoning.]")
        parser.add_argument("--perturbation_suffix", type=str, default="[END OF QUERY]")
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        parser.add_argument("--max_tokens", type=int, default=4096)
        parser.add_argument("--temperature_random", type=float, default=0.8)
        parser.add_argument("--num_random_samples", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument(
            "--gpu_memory_utilization",
            type=float,
            default=0.9,
            help="Fraction of GPU memory vLLM may use for the KV cache / weights.",
        )
        parser.add_argument(
            "--max_model_len",
            type=int,
            default=None,
            help="Cap vLLM context length (prompt + completion). Use when the HF "
            "config max (e.g. 131072) exceeds available KV cache on your GPU.",
        )
        parser.add_argument(
            "--swap_space",
            type=float,
            default=16.0,
            help="CPU memory (GiB) per GPU for vLLM KV swap when GPU cache is full. "
            "Raise if you see 'lack of CPU swap space'.",
        )
        parser.add_argument(
            "--max_num_seqs",
            type=int,
            default=None,
            help="Optional cap on concurrent sequences in vLLM (lower reduces KV preemption).",
        )
        parser.add_argument(
            "--max_num_batched_tokens",
            type=int,
            default=None,
            help="Optional vLLM max_num_batched_tokens (lower caps scheduling memory).",
        )
        parser.add_argument(
            "--vllm_attention_backend",
            type=str,
            default=None,
            help="vLLM v1 attention backend name (e.g. FLASHINFER, TRITON_ATTN). "
            "Avoids pip flash-attn when ABI-mismatched with torch. "
            "If unset, uses env VLLM_ATTENTION_BACKEND when non-empty.",
        )
        parser.add_argument(
            "--vllm_random_microbatch",
            type=int,
            default=None,
            help="Microbatch size for consistency/random n>1 generation (default: env VLLM_RANDOM_MICROBATCH or 8).",
        )
        parser.add_argument(
            "--vllm_critique_microbatch",
            type=int,
            default=None,
            help="Microbatch size for self_critique pass (default: env VLLM_CRITIQUE_MICROBATCH or 8).",
        )
        parser.add_argument("--subset_source", type=str, default=None)
        parser.add_argument("--num_samples_per_source", type=int, default=-1)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--K", type=int, default=50)
        parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
        parser.add_argument("--methods_to_run", type=str, nargs='*', default=[],
                            choices=['dime', 'consistency', 'self_critique', 'self_critique_ablation'])
        parser.add_argument("--approx_mode", type=str, default="renorm", choices=["renorm", "rest"])
        return parser

    def run(self):
        parser = self.parser
        args = parser.parse_args()
        # vLLM v1 EngineCore subprocesses inherit PYTHONPATH but not this script's monkey-patches;
        # prepend compat_site so startup imports compat_site/sitecustomize.py before Transformers+vLLM.
        _prepend_compat_site_to_pythonpath_for_vllm_workers()

        # --- Compatibility patch for older Transformers ---
        # Some tokenizer configs (e.g. Qwen) store `extra_special_tokens` as a *list*.
        # Transformers<=4.55 can crash expecting a dict (special_tokens.keys()).
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase

            _orig_set_model_specific_special_tokens = PreTrainedTokenizerBase._set_model_specific_special_tokens

            def _set_model_specific_special_tokens_compat(self, special_tokens=None):
                if isinstance(special_tokens, list):
                    # Treat list form as "no model-specific mapping overrides".
                    return
                return _orig_set_model_specific_special_tokens(self, special_tokens=special_tokens)

            PreTrainedTokenizerBase._set_model_specific_special_tokens = _set_model_specific_special_tokens_compat
        except Exception:
            pass

        # --- 1. Resume from checkpoint (with backfill for missing method outputs) ---
        # `generate_full_data.py` historically skipped prompts solely based on `original_user_content` being present
        # in the output file. This can silently produce incomplete rows when you change `--methods_to_run`
        # (e.g., adding `self_critique` later won't backfill `critique_greedy_results`).
        #
        # New behavior: treat a prompt as "processed" only if all required result fields for the requested
        # methods are present and non-empty. If some required fields are missing, we reprocess that prompt
        # and then rewrite the output JSONL without duplicating rows.
        def _has_nonempty_list_field(item: dict, key: str) -> bool:
            v = item.get(key)
            return isinstance(v, list) and len(v) > 0

        required_result_keys = {"original_greedy_results"}
        if "dime" in args.methods_to_run:
            required_result_keys.add("perturbed_greedy_results")
        if "consistency" in args.methods_to_run:
            required_result_keys.add("original_random_results")
        if "self_critique" in args.methods_to_run:
            required_result_keys.add("critique_greedy_results")
        if "self_critique_ablation" in args.methods_to_run:
            required_result_keys.add("unfamiliar_greedy_results")

        existing_by_content = {}
        existing_order = []
        processed_contents = set()
        if os.path.exists(args.output_file):
            with open(args.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    content = item.get("original_user_content")
                    if not content:
                        continue
                    if content not in existing_by_content:
                        existing_order.append(content)
                    # Keep the last occurrence if duplicates exist.
                    existing_by_content[content] = item

            for content, item in existing_by_content.items():
                if (item.get("generation_schema_version") == GENERATION_SCHEMA_VERSION
                        and item.get("approx_mode") == args.approx_mode
                        and all(_has_nonempty_list_field(item, k) for k in required_result_keys)):
                    processed_contents.add(content)

            incomplete = len(existing_by_content) - len(processed_contents)
            if incomplete > 0:
                print(
                    f"Found {len(existing_by_content)} prompts in existing output; "
                    f"{len(processed_contents)} complete, {incomplete} missing required fields for "
                    f"methods_to_run={args.methods_to_run}. Will backfill missing fields."
                )
            else:
                print(f"Found {len(processed_contents)} already processed prompts, will skip automatically.")

        # --- 2. Load, filter and sample ---
        df_all = load_evaluation_tasks(args.data_root_dir)

        if args.subset_source:
            df_filtered = df_all[df_all['data_source'] == args.subset_source].copy()
        else:
            df_filtered = df_all

        if args.num_samples_per_source > 0:
            df_sampled = df_filtered.groupby('data_source').apply(lambda x: x.sample(n=min(len(x), args.num_samples_per_source), random_state=42)).reset_index(drop=True)
        else:
            df_sampled = df_filtered

        # Drop rows from old input selections (including previously inferred labels).
        selected = {
            get_user_content(row["prompt"]): row for _, row in df_sampled.iterrows()
            if get_user_content(row["prompt"])
        }
        existing_by_content = {
            key: row for key, row in existing_by_content.items()
            if key in selected
            and row.get("generation_schema_version") == GENERATION_SCHEMA_VERSION
            and row.get("approx_mode") == args.approx_mode
            and row.get("ground_truth_label") == int(selected[key]["member"])
            and row.get("data_source") == selected[key]["data_source"]
        }
        existing_order = [key for key in existing_order if key in existing_by_content]
        processed_contents.intersection_update(existing_by_content)

        tasks_to_process = [
            row
            for _, row in df_sampled.iterrows()
            if get_user_content(row["prompt"]) and get_user_content(row["prompt"]) not in processed_contents
        ]

        if not tasks_to_process:
            if os.path.exists(args.output_file):
                _write_generation_rows(args.output_file, (existing_by_content[key] for key in existing_order))
            # Extra diagnostics to avoid silent empty runs.
            total = len(df_sampled)
            missing_content = 0
            sample_type = None
            try:
                if total > 0:
                    sample = df_sampled.iloc[0]['prompt']
                    sample_type = type(sample).__name__
                for _, r in df_sampled.iterrows():
                    if not get_user_content(r.get('prompt')):
                        missing_content += 1
            except Exception:
                pass
            print(
                "All target prompts have been processed or not found. Program exiting.\n"
                f"- total_rows_after_filter: {total}\n"
                f"- processed_contents: {len(processed_contents)}\n"
                f"- rows_with_missing_user_content: {missing_content}\n"
                f"- sample_prompt_type: {sample_type}"
            )
            return
        print(f"\nNumber of new prompts to process: {len(tasks_to_process)}")
        print(f"Data source distribution to process:\n{pd.Series([t['data_source'] for t in tasks_to_process]).value_counts()}")

        # --- 3. Load VLLM ---
        # Patch vLLM tokenizer caching to tolerate missing all_special_tokens_extended
        # in older/custom tokenizer implementations (works across vLLM API moves).
        _patch_vllm_get_cached_tokenizer_safe()

        logprobs_to_request = args.K
        patched_model_path = _maybe_patch_model_dir_for_vllm(args.model_path)
        print(f"Loading model: {patched_model_path} (TP={args.tensor_parallel_size})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(patched_model_path, trust_remote_code=True)
        except AttributeError as e:
            # Fallback for the same "extra_special_tokens list" issue.
            if "keys" in str(e):
                tokenizer = AutoTokenizer.from_pretrained(
                    patched_model_path,
                    trust_remote_code=True,
                    extra_special_tokens={},
                )
            else:
                raise
        llm_kwargs = dict(
            model=patched_model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_logprobs=logprobs_to_request,
            dtype=args.dtype,
            enforce_eager=True,
            swap_space=float(args.swap_space),
        )
        attn_backend = (args.vllm_attention_backend or "").strip()
        if not attn_backend:
            attn_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "").strip()
        if attn_backend:
            # vLLM 0.19+: maps to AttentionConfig.backend (see vllm.config.attention).
            llm_kwargs["attention_config"] = {"backend": attn_backend}
            print(f"[vLLM] attention_config.backend={attn_backend!r}")
        if args.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(args.max_num_seqs)
        if args.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = int(args.max_model_len)
            print(f"[vLLM] max_model_len={llm_kwargs['max_model_len']}")
        llm = LLM(**llm_kwargs)
        # Get model's maximum length limit
        max_model_len = llm.llm_engine.model_config.max_model_len
        print(f"Detected model maximum length: {max_model_len}")

        # --- 4. Define sampling strategies ---
        greedy_params = SamplingParams(temperature=0, n=1, max_tokens=args.max_tokens, logprobs=logprobs_to_request)
        # vLLM treats temperature=0 as greedy sampling, which requires n=1.
        # If consistency is requested with n>1, ensure temperature_random > 0.
        temp_random = float(args.temperature_random)
        if temp_random <= 0.0 and int(args.num_random_samples) > 1:
            # Keep the run going with a tiny epsilon instead of crashing.
            # This still behaves "almost greedy" but satisfies vLLM's constraint.
            print(
                f"[warn] temperature_random={temp_random} with num_random_samples={args.num_random_samples} "
                "is invalid for vLLM (greedy requires n=1). Using temperature_random=1e-5."
            )
            temp_random = 1e-5
        random_params = SamplingParams(
            temperature=temp_random,
            n=args.num_random_samples,
            max_tokens=args.max_tokens,
            logprobs=logprobs_to_request,
            top_p=0.95,
        )

        # --- 5. Incremental processing and saving ---
        # Compact any interrupted append (including a partial final line) before resuming.
        updated_by_content = existing_by_content
        _write_generation_rows(args.output_file,
                               (updated_by_content[key] for key in existing_order))
        f_out = open(args.output_file, "a", encoding="utf-8")
        try:
            for i in tqdm(range(0, len(tasks_to_process), args.batch_size), desc="Processing Batches"):
                batch_tasks = tasks_to_process[i:i+args.batch_size]

                # Prepare two sets of prompts: original and perturbed
                batch_original_prompts_formatted = []
                batch_perturbed_prompts_formatted = []
                for task in batch_tasks:
                    original_prompt = _coerce_messages(task.get('prompt'))
                    perturbed_prompt = copy.deepcopy(original_prompt)
                    user_content = get_user_content(original_prompt)
                    perturbed_content = f"{args.perturbation_prefix} {user_content} {args.perturbation_suffix}".strip()
                    if perturbed_prompt:
                        perturbed_prompt[-1]['content'] = perturbed_content

                    batch_original_prompts_formatted.append(format_prompt(original_prompt, args.prompt_template, tokenizer))
                    batch_perturbed_prompts_formatted.append(format_prompt(perturbed_prompt, args.prompt_template, tokenizer))

                # 1. Original Greedy sampling (required by all methods)
                try:
                    original_greedy_outputs = llm.generate(batch_original_prompts_formatted, greedy_params)
                except Exception as e:
                    print(f"Batch {i // args.batch_size} original Greedy sampling failed: {e}")
                    continue

                # 2. Perturbed Greedy sampling
                perturbed_greedy_outputs = None
                if 'dime' in args.methods_to_run:
                    try:
                        perturbed_greedy_outputs = llm.generate(batch_perturbed_prompts_formatted, greedy_params)
                    except Exception as e:
                        print(f"Batch {i // args.batch_size} perturbed Greedy sampling failed: {e}")
                        perturbed_greedy_outputs = None # Ensure None after failure

                # 3. Original Random sampling (only when consistency is needed)
                original_random_outputs = None
                if 'consistency' in args.methods_to_run:
                    try:
                        mb = (
                            int(args.vllm_random_microbatch)
                            if args.vllm_random_microbatch is not None
                            else int(os.getenv("VLLM_RANDOM_MICROBATCH", "8"))
                        )
                        original_random_outputs = _generate_with_microbatches(
                            llm,
                            batch_original_prompts_formatted,
                            random_params,
                            initial_microbatch_size=mb,
                            label="consistency/random",
                        )
                    except Exception as e:
                        print(f"Batch {i // args.batch_size} original Random sampling failed: {e}")
                        original_random_outputs = None

                # 4. Self-critique Greedy sampling (only when self_critique is needed)
                critique_greedy_outputs = None
                if 'self_critique' in args.methods_to_run:
                    batch_critique_prompts = []
                    SELF_CRITIQUE_INSTRUCTION = "\nA possible answer is provided below (it may or may not be correct). Please provide a response that follows a different reasoning path or provides an alternative solution:\n---\n{response}\n---\nPlease now provide your new, different response:"
                    for j in range(len(batch_tasks)):
                        first_pass_text = original_greedy_outputs[j].outputs[0].text
                        task = batch_tasks[j]
                        original_prompt = _coerce_messages(task.get('prompt'))
                        critique_prompt = copy.deepcopy(original_prompt)
                        template_prompt_formatted = format_prompt(critique_prompt, args.prompt_template, tokenizer)
                        template_token_ids = tokenizer.encode(template_prompt_formatted)
                        # Leave headroom for instruction + critique generation.
                        max_response_len = max(0, max_model_len - len(template_token_ids) - 512)

                        # As the model context window is limited, we may need to truncate the response
                        response_token_ids = tokenizer.encode(first_pass_text)
                        if len(response_token_ids) > max_response_len:
                            truncated_response_ids = response_token_ids[:max_response_len]
                            truncated_response_text = tokenizer.decode(truncated_response_ids)
                            print('Warning: truncated response due to context window limit')
                        else:
                            truncated_response_text = first_pass_text


                        user_content = get_user_content(original_prompt)
                        new_user_content = user_content + SELF_CRITIQUE_INSTRUCTION.format(response=truncated_response_text)
                        critique_prompt[-1]['content'] = new_user_content
                        batch_critique_prompts.append(format_prompt(critique_prompt, args.prompt_template, tokenizer))
                    try:
                        # Ensure we never exceed context window: prompt_len + max_tokens <= max_model_len.
                        prompt_lens = [len(tokenizer.encode(p)) for p in batch_critique_prompts]
                        max_prompt_len = max(prompt_lens) if prompt_lens else 0
                        available = max(1, max_model_len - max_prompt_len - 32)
                        max_tokens_critique = int(min(args.max_tokens, available))
                        greedy_critique_params = SamplingParams(
                            temperature=args.temperature,
                            n=1,
                            max_tokens=max_tokens_critique,
                            logprobs=logprobs_to_request,
                        )
                        mb = (
                            int(args.vllm_critique_microbatch)
                            if args.vllm_critique_microbatch is not None
                            else int(os.getenv("VLLM_CRITIQUE_MICROBATCH", "8"))
                        )
                        critique_greedy_outputs = _generate_with_microbatches(
                            llm,
                            batch_critique_prompts,
                            greedy_critique_params,
                            initial_microbatch_size=mb,
                            label="self_critique",
                        )
                    except Exception as e:
                        print(f"Batch {i // args.batch_size} self-critique sampling failed: {e}")
                        critique_greedy_outputs = None

                # 5. Ablation version "unfamiliar/unconventional method" Greedy sampling (without concatenating first-pass answer content)
                unfamiliar_greedy_outputs = None
                if 'self_critique_ablation' in args.methods_to_run:
                    batch_unfamiliar_prompts = []
                    UNFAMILIAR = "Answer using a technique you’d typically avoid or a deliberately unconventional line of reasoning."
                    for task in batch_tasks:
                        original_prompt = _coerce_messages(task.get('prompt'))
                        prompt2 = copy.deepcopy(original_prompt)
                        user_content = get_user_content(original_prompt)
                        # Only append instruction, without first-pass answer
                        new_user = f"{user_content}\n\n{UNFAMILIAR}"
                        prompt2[-1]['content'] = new_user
                        batch_unfamiliar_prompts.append(format_prompt(prompt2, args.prompt_template, tokenizer))
                    try:
                        unfamiliar_params = SamplingParams(
                            temperature=args.temperature, n=1,
                            max_tokens=args.max_tokens*2, logprobs=logprobs_to_request
                        )
                        unfamiliar_greedy_outputs = llm.generate(batch_unfamiliar_prompts, unfamiliar_params)
                    except Exception as e:
                        print(f"Batch {i // args.batch_size} unfamiliar method ablation sampling failed: {e}")
                        unfamiliar_greedy_outputs = None

                for j in range(len(batch_tasks)):
                    task = batch_tasks[j]
                    original_prompt = _coerce_messages(task.get('prompt'))
                    content = get_user_content(original_prompt)
                    final_item = {
                        "generation_schema_version": GENERATION_SCHEMA_VERSION,
                        "approx_mode": args.approx_mode,
                        "original_user_content": content,
                        "ground_truth_label": 1 if task["member"] else 0,
                        "data_source": task["data_source"],
                    }

                    # Add result fields as needed
                    final_item["original_greedy_results"] = [process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in original_greedy_outputs[j].outputs]

                    if perturbed_greedy_outputs:
                        final_item["perturbed_greedy_results"] = [process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in perturbed_greedy_outputs[j].outputs]

                    if original_random_outputs:
                        ro = original_random_outputs[j] if j < len(original_random_outputs) else None
                        if ro is not None:
                            final_item["original_random_results"] = [
                                process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in ro.outputs
                            ]

                    if critique_greedy_outputs:
                        co = critique_greedy_outputs[j] if j < len(critique_greedy_outputs) else None
                        if co is not None:
                            final_item["critique_greedy_results"] = [
                                process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in co.outputs
                            ]

                    if unfamiliar_greedy_outputs:
                        final_item["unfamiliar_greedy_results"] = [process_single_output(o, tokenizer, approx_mode=args.approx_mode) for o in unfamiliar_greedy_outputs[j].outputs]

                    # Loud warnings when requested methods are missing for this sample.
                    if "self_critique" in args.methods_to_run and "critique_greedy_results" not in final_item:
                        print(f"[warn] Missing critique_greedy_results for sample idx={i + j}.")
                    if "consistency" in args.methods_to_run and "original_random_results" not in final_item:
                        print(f"[warn] Missing original_random_results for sample idx={i + j}.")

                    prev = updated_by_content.get(content, {})
                    merged = dict(prev)
                    merged.update(final_item)
                    if content not in updated_by_content:
                        existing_order.append(content)
                    updated_by_content[content] = merged
                    # Append and sync every completed sample, including resumed/backfill runs.
                    f_out.write(json.dumps(merged) + "\n")
                    f_out.flush()
                    os.fsync(f_out.fileno())
        finally:
            f_out.close()

        order = dict.fromkeys(existing_order + list(updated_by_content))
        _write_generation_rows(args.output_file,
                               (updated_by_content[key] for key in order if key in updated_by_content))

        print(f"\nAll data successfully saved to: {args.output_file}")


# Compatibility functions preserve the existing importable API.
def _maybe_patch_model_dir_for_vllm(model_path: str) -> str:
    """
    Apply small `config.json` fixes for vLLM / Transformers quirks:

    * Some Qwen2-style configs use `rope_scaling` without a `factor` key; vLLM
      asserts `"factor" in rope_scaling`. We add a no-op linear scaling
      (factor=1.0) when needed.
    * OLMo3 checkpoints may store `rope_parameters.beta_{fast,slow}` as JSON
      integers; Transformers expects floats and logs validation noise otherwise.
    """
    return VLLMCompatibility._maybe_patch_model_dir_for_vllm(model_path)


def _patch_vllm_get_cached_tokenizer_safe() -> None:
    """
    vLLM wraps HF tokenizers with get_cached_tokenizer for speed. Some tokenizers
    lack `all_special_tokens_extended`, which older vLLM paths assumed.

    vLLM 0.19+ defines get_cached_tokenizer in `vllm.tokenizers.hf`; older
    releases exposed it on `vllm.transformers_utils.tokenizer`.
    """
    return VLLMCompatibility._patch_vllm_get_cached_tokenizer_safe()


def _prepend_compat_site_to_pythonpath_for_vllm_workers() -> None:
    """vLLM v1 spawn's EngineCore needs sitecustomize; prepend compat_site early."""
    return VLLMCompatibility._prepend_compat_site_to_pythonpath_for_vllm_workers()


def _normalize_prompt_obj(obj):
    """Best-effort normalization across pandas/pyarrow/numpy prompt representations."""
    return PromptAdapter._normalize_prompt_obj(obj)


def get_user_content(prompt_obj):
    return PromptAdapter.get_user_content(prompt_obj)


def _coerce_messages(prompt_obj):
    """Coerce various prompt formats into a list of {role, content} dicts."""
    return PromptAdapter._coerce_messages(prompt_obj)


def _read_parquet_with_fallback(path: str) -> pd.DataFrame:
    """
    Read parquet robustly across environments.

    Some parquet engines (e.g., fastparquet) can silently drop/NULL nested columns
    like our `prompt` (list-of-struct). Prefer pyarrow; fallback to datasets if available.
    """
    return EvaluationDataset._read_parquet_with_fallback(path)

def format_prompt(message, template, tokenizer):
    return PromptAdapter.format_prompt(message, template, tokenizer)

GENERATION_SCHEMA_VERSION = 2


def process_single_output(output, tokenizer, approx_mode: str='renorm'):
    """Approximate token distributions while preserving raw likelihoods.

    renorm conditions the retained tokens on their total probability mass.
    rest represents all omitted mass as one additional category. Min-K++ uses
    log probabilities in the same distribution as its mean and variance;
    raw logprobs remain unchanged for perplexity and ordinary Min-K.
    """
    return TokenDistributionStatistics.process_single_output(output, tokenizer, approx_mode)


def load_evaluation_tasks(data_path: str) -> pd.DataFrame:
    """Load explicitly labeled evaluation inputs from a file or directory."""
    return EvaluationDataset.load_evaluation_tasks(data_path)


def _write_generation_rows(path, rows):
    """Atomically replace a resumed generation file."""
    return GenerationOutputStore._write_generation_rows(path, rows)


def _generate_with_microbatches(llm, prompts, sampling_params, *, initial_microbatch_size: int=8, label: str=''):
    """
    Run vLLM generation in microbatches with fallback to smaller microbatches on failure.
    Returns a list aligned to `prompts`, with None for prompts that fail even at microbatch size 1.
    """
    return MicrobatchGenerator._generate_with_microbatches(llm, prompts, sampling_params, initial_microbatch_size=initial_microbatch_size, label=label)

def main():
    return GenerationApplication().run()

if __name__ == "__main__":
    main()
