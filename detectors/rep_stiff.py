from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import BaseDetector
from .utils.make_similar_questions import generate_similar_questions
from .utils.make_incomplete_questions import (
    identify_information_to_remove,
    generate_incomplete_question_with_guidance,
    generate_incomplete_questions_with_guidance_batch,
    generate_incomplete_question_by_importance,
    generate_incomplete_questions_by_importance_batch,
    generate_perturbed_questions_num_replace_batch,
    generate_perturbed_questions_var_rename_batch,
    generate_perturbed_questions_distractor_batch,
)
from .utils.make_incomplete_questions_rsi import (
    generate_incomplete_variants,
    generate_incomplete_variants_batch,
    generate_question_variants,
    generate_question_variants_batch,
)

_BLANK_STRATEGIES = frozenset({"important", "guided", "info", "info_type", "guidance", "info_rem"})
_PERTURB_STRATEGIES = frozenset({"num_replace", "var_rename", "distractor", "distractor_insert"})
from .utils import rsm_calculation, dc_calculation, rsi_calculation


class RepStiffDetector(BaseDetector):
    """
    RepStiff detector using utils/* implementations:
      1) Generate similar questions and save to JSON.
      2) Create incomplete question pairs and save to JSON.
      3) Compute RSM (zRSM), Directional Collapse, and RSI (zRSI).
    """

    def __init__(
        self,
        num_similar_questions: int = 5,
        num_rsi_variants: int = 5,
        max_openrouter_workers: int = 4,
        max_incomplete_retries: int = 2,
        min_valid_similars: int = 2,
        device: str = "cuda",
        output_dir: Optional[str] = None,
        reuse_existing: bool = True,
        score_mode: str = "rsi",
        model_name: Optional[str] = None,
        openrouter_model: str = "openai/gpt-4o-mini",
        layer_name: str = "mid",
        incomplete_blank_strategy: str = "important",
        incomplete_num_blanks: int = 1,
    ):
        self.num_similar_questions = num_similar_questions
        self.num_rsi_variants = num_rsi_variants
        self.max_openrouter_workers = max(1, max_openrouter_workers)
        self.max_incomplete_retries = max(0, max_incomplete_retries)
        self.min_valid_similars = max(0, min_valid_similars)
        self.device = device
        self.reuse_existing = reuse_existing
        self.score_mode = score_mode
        self.model_name = (
            model_name
            or os.getenv("REP_STIFF_MODEL_NAME")
            or os.getenv("HF_MODEL_ID")
        )
        self.openrouter_model = openrouter_model
        self.layer_name = layer_name
        strategy = (incomplete_blank_strategy or "important").strip().lower()
        if strategy == "info_rem":
            strategy = "important"
        if strategy == "distractor_insert":
            strategy = "distractor"
        self.incomplete_blank_strategy = strategy
        self.incomplete_num_blanks = max(1, int(incomplete_num_blanks))
        self.output_dir = output_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "rep_stiff_outputs")
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._rsm_ready = False
        self._dc_ready = False
        self._rsi_ready = False
        self._model_bound = False

    _shared_model = None
    _shared_tokenizer = None
    _shared_layer_map = None
    _shared_device = None
    _shared_model_name = None
    _shared_device_name = None

    FIXED_TREND_WEIGHTS = {
        "rsi_slope": 0.6,
        "rsi_mid": 0.2,
        "dc_slope": 0.1,
        "rsm_slope": 0.1,
    }
    FIXED_TREND_V2_WEIGHTS = {
        "rsi_slope": 0.5,
        "rsi_curvature_penalty": 0.3,
        "rsm_slope": 0.1,
        "dc_slope": 0.1,
    }
    FIXED_TREND_V3_WEIGHTS = {
        "rsi_mid": 0.7,
        "rsi_peak_margin": 0.2,
        "rsm_slope": 0.1,
        "dc_slope": 0.1,
    }
    FIXED_TREND_V4_WEIGHTS = {
        "rsi_mid": 0.5,
        "rsi_mid_vs_early": 0.2,
        "rsi_mid_vs_late": 0.2,
        "rsm_slope": 0.1,
        "dc_slope": 0.1,
    }

    @staticmethod
    def compute_combined_score(
        layer_scores: Dict[str, float],
        weights_path: Optional[str],
    ) -> Optional[float]:
        if not weights_path or not os.path.exists(weights_path):
            return None
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        weights = payload.get("weights", {})
        bias = payload.get("bias", 0.0)
        if not isinstance(weights, dict):
            return None
        total = float(bias)
        for key, weight in weights.items():
            val = layer_scores.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            total += float(weight) * float(val)
        return total

    @classmethod
    def compute_fixed_trend_score(cls, layer_scores: Dict[str, float]) -> Optional[float]:
        def _get(name: str) -> Optional[float]:
            val = layer_scores.get(name)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)

        rsi_early = _get("rsi_early")
        rsi_mid = _get("rsi_mid")
        rsi_late = _get("rsi_late")
        rsm_early = _get("rsm_early")
        rsm_late = _get("rsm_late")
        dc_early = _get("directional_collapse_early")
        dc_late = _get("directional_collapse_late")

        if rsi_early is None or rsi_late is None or rsi_mid is None:
            return None

        rsi_slope = rsi_late - rsi_early
        rsm_slope = None if rsm_early is None or rsm_late is None else (rsm_late - rsm_early)
        dc_slope = None if dc_early is None or dc_late is None else (dc_late - dc_early)

        total = 0.0
        total += cls.FIXED_TREND_WEIGHTS["rsi_slope"] * rsi_slope
        total += cls.FIXED_TREND_WEIGHTS["rsi_mid"] * rsi_mid
        if dc_slope is not None:
            total += cls.FIXED_TREND_WEIGHTS["dc_slope"] * dc_slope
        if rsm_slope is not None:
            total += cls.FIXED_TREND_WEIGHTS["rsm_slope"] * rsm_slope
        return total

    @classmethod
    def compute_fixed_trend_v2_score(cls, layer_scores: Dict[str, float]) -> Optional[float]:
        def _get(name: str) -> Optional[float]:
            val = layer_scores.get(name)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)

        rsi_early = _get("rsi_early")
        rsi_mid = _get("rsi_mid")
        rsi_late = _get("rsi_late")
        rsm_early = _get("rsm_early")
        rsm_late = _get("rsm_late")
        dc_early = _get("directional_collapse_early")
        dc_late = _get("directional_collapse_late")

        if rsi_early is None or rsi_mid is None or rsi_late is None:
            return None

        rsi_slope = rsi_late - rsi_early
        rsi_mid_expected = (rsi_early + rsi_late) / 2.0
        rsi_curvature = abs(rsi_mid - rsi_mid_expected)
        rsm_slope = None if rsm_early is None or rsm_late is None else (rsm_late - rsm_early)
        dc_slope = None if dc_early is None or dc_late is None else (dc_late - dc_early)

        total = 0.0
        total += cls.FIXED_TREND_V2_WEIGHTS["rsi_slope"] * rsi_slope
        total -= cls.FIXED_TREND_V2_WEIGHTS["rsi_curvature_penalty"] * rsi_curvature
        if rsm_slope is not None:
            total += cls.FIXED_TREND_V2_WEIGHTS["rsm_slope"] * rsm_slope
        if dc_slope is not None:
            total -= cls.FIXED_TREND_V2_WEIGHTS["dc_slope"] * dc_slope
        return total

    @classmethod
    def compute_fixed_trend_v3_score(cls, layer_scores: Dict[str, float]) -> Optional[float]:
        def _get(name: str) -> Optional[float]:
            val = layer_scores.get(name)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)

        rsi_early = _get("rsi_early")
        rsi_mid = _get("rsi_mid")
        rsi_late = _get("rsi_late")
        rsm_early = _get("rsm_early")
        rsm_late = _get("rsm_late")
        dc_early = _get("directional_collapse_early")
        dc_late = _get("directional_collapse_late")

        if rsi_early is None or rsi_mid is None or rsi_late is None:
            return None

        rsi_peak_margin = max(0.0, rsi_mid - max(rsi_early, rsi_late))
        rsm_slope = None if rsm_early is None or rsm_late is None else (rsm_late - rsm_early)
        dc_slope = None if dc_early is None or dc_late is None else (dc_late - dc_early)

        total = 0.0
        total += cls.FIXED_TREND_V3_WEIGHTS["rsi_mid"] * rsi_mid
        total += cls.FIXED_TREND_V3_WEIGHTS["rsi_peak_margin"] * rsi_peak_margin
        if rsm_slope is not None:
            total += cls.FIXED_TREND_V3_WEIGHTS["rsm_slope"] * rsm_slope
        if dc_slope is not None:
            total -= cls.FIXED_TREND_V3_WEIGHTS["dc_slope"] * dc_slope
        return total

    @classmethod
    def compute_fixed_trend_v4_score(cls, layer_scores: Dict[str, float]) -> Optional[float]:
        def _get(name: str) -> Optional[float]:
            val = layer_scores.get(name)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)

        rsi_early = _get("rsi_early")
        rsi_mid = _get("rsi_mid")
        rsi_late = _get("rsi_late")
        rsm_early = _get("rsm_early")
        rsm_late = _get("rsm_late")
        dc_early = _get("directional_collapse_early")
        dc_late = _get("directional_collapse_late")

        if rsi_early is None or rsi_mid is None or rsi_late is None:
            return None

        rsi_mid_vs_early = rsi_mid - rsi_early
        rsi_mid_vs_late = rsi_mid - rsi_late
        rsm_slope = None if rsm_early is None or rsm_late is None else (rsm_late - rsm_early)
        dc_slope = None if dc_early is None or dc_late is None else (dc_late - dc_early)

        total = 0.0
        total += cls.FIXED_TREND_V4_WEIGHTS["rsi_mid"] * rsi_mid
        total += cls.FIXED_TREND_V4_WEIGHTS["rsi_mid_vs_early"] * rsi_mid_vs_early
        total += cls.FIXED_TREND_V4_WEIGHTS["rsi_mid_vs_late"] * rsi_mid_vs_late
        if rsm_slope is not None:
            total += cls.FIXED_TREND_V4_WEIGHTS["rsm_slope"] * rsm_slope
        if dc_slope is not None:
            total -= cls.FIXED_TREND_V4_WEIGHTS["dc_slope"] * dc_slope
        return total

    def get_name(self):
        return f"rep_stiff_{self.score_mode}_score"

    def calculate_score(self, data_item):
        scores, _paths = self.calculate_scores(data_item)
        return scores.get(self.score_mode, np.nan)

    def calculate_scores(self, data_item) -> Tuple[Dict[str, float], Dict[str, str]]:
        question = data_item.get("original_user_content")
        if not question:
            return {
                "rsm_score": np.nan,
                "directional_collapse_score": np.nan,
                "rsi_score": np.nan,
            }, {}

        similar_entry, similar_path = self._get_or_create_similar_questions(question, data_item)
        pairs, pairs_path = self._get_or_create_incomplete_pairs(similar_entry, data_item)

        rsm_score = self.rsm_score_calculation(similar_entry, pairs)
        directional_collapse_score = self.directional_collapse_score_calculation(similar_entry, pairs)
        rsi_score = self.rsi_score_calculation(similar_entry, pairs)

        return (
            {
                "rsm_score": rsm_score,
                "directional_collapse_score": directional_collapse_score,
                "rsi_score": rsi_score,
            },
            {
                "similar_questions_path": similar_path,
                "incomplete_pairs_path": pairs_path,
                "num_pairs": len(pairs),
            },
        )

    def rsm_score_calculation(self, similar_entry: Dict, pairs: List[Dict]) -> float:
        self._ensure_rsm_ready()
        layer_idx = rsm_calculation.LAYER_MAP[self.layer_name]
        incomplete_mapping = self._build_incomplete_mapping(pairs)
        orig_key = (similar_entry.get("original_question_id", 0), None)
        if orig_key not in incomplete_mapping:
            return float(np.nan)
        try:
            out = rsm_calculation.compute_group_zrsm(similar_entry, incomplete_mapping, layer_idx)
            return float(out.get("zRSM", np.nan))
        except Exception:
            return float(np.nan)

    def directional_collapse_score_calculation(self, similar_entry: Dict, pairs: List[Dict]) -> float:
        self._ensure_dc_ready()
        layer_idx = dc_calculation.LAYER_MAP[self.layer_name]
        incomplete_mapping = self._build_incomplete_mapping(pairs)
        orig_key = (similar_entry.get("original_question_id", 0), None)
        if orig_key not in incomplete_mapping:
            return float(np.nan)
        try:
            out = dc_calculation.compute_group_alignment(similar_entry, incomplete_mapping, layer_idx)
            return float(out.get("alignment", np.nan))
        except Exception:
            return float(np.nan)

    def rsi_score_calculation(self, similar_entry: Dict, pairs: List[Dict]) -> float:
        self._ensure_rsi_ready()
        layer_idx = rsi_calculation.LAYER_MAP[self.layer_name]
        variants_entries, _variants_path = self._get_or_create_rsi_variants(similar_entry, pairs)
        group_data = self._build_rsi_group(variants_entries)
        if group_data["original"] is None or len(group_data["similars"]) == 0:
            return np.nan
        out = rsi_calculation.compute_group_zrsi(group_data, layer_idx)
        return float(out.get("zRSI", np.nan))

    @staticmethod
    def _read_cache(path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

    def _cache_lock(self, question, kind):
        from detectors.utils.shared_cache import SharedCacheLock
        return SharedCacheLock(self._build_output_path(question, suffix=kind + '.lock'))

    @staticmethod
    def _write_cache(path, payload):
        from eval_scripts.collection.checkpoints import AtomicJSON
        AtomicJSON.write(path, payload)

    def _get_or_create_similar_questions(self, question, data_item):
        with self._cache_lock(question, 'similar_questions'):
            return self._get_or_create_similar_questions_unlocked(question, data_item)

    def _get_or_create_incomplete_pairs(self, similar_entry, data_item):
        with self._cache_lock(similar_entry['original_question'], 'incomplete_pairs'):
            return self._get_or_create_incomplete_pairs_unlocked(similar_entry, data_item)

    def _get_or_create_rsi_variants(self, similar_entry, pairs):
        with self._cache_lock(similar_entry['original_question'], 'incomplete_variants'):
            return self._get_or_create_rsi_variants_unlocked(similar_entry, pairs)

    def _get_or_create_similar_questions_unlocked(self, question: str, data_item: Dict) -> Tuple[Dict, str]:
        path = self._build_output_path(question, suffix="similar_questions.json")
        if self.reuse_existing:
            payload = self._read_cache(path)
            entries = payload.get("similar_questions", [])
            if (payload.get("original_question") == question
                    and payload.get("generator_model") == self.openrouter_model
                    and isinstance(entries, list)
                    and len(entries) == self.num_similar_questions
                    and all(isinstance(e, dict) and isinstance(e.get("question"), str)
                            and e["question"].strip() and e.get("id") == i + 1
                            for i, e in enumerate(entries))):
                return payload, path

        similar_questions = generate_similar_questions(
            question,
            num_questions=self.num_similar_questions,
            model=self.openrouter_model,
        )
        similar_entries = [
            {"id": idx + 1, "question": q} for idx, q in enumerate(similar_questions)
        ]
        payload = {
            "dataset": data_item.get("data_source", "unknown"),
            "original_question_id": 0,
            "original_question": question,
            "similar_questions": similar_entries,
            "generator_model": self.openrouter_model,
        }
        self._write_cache(path, payload)
        return payload, path

    def _get_or_create_incomplete_pairs_unlocked(
        self, similar_entry: Dict, data_item: Dict
    ) -> Tuple[List[Dict], str]:
        question = similar_entry["original_question"]
        suffix = f"incomplete_pairs__{self.incomplete_blank_strategy}__k{self.incomplete_num_blanks}.json"
        path = self._build_output_path(question, suffix=suffix)
        all_questions = [question] + [s["question"] for s in similar_entry.get("similar_questions", [])]
        expected_ids = [None] + [s["id"] for s in similar_entry.get("similar_questions", [])]
        cached = self._read_cache(path) if self.reuse_existing else {}
        old_pairs = cached.get("pairs", []) if cached.get("generator_model") == self.openrouter_model else []
        if not isinstance(old_pairs, list):
            old_pairs = []
        incomplete_list = [None] * len(all_questions)
        for i, original in enumerate(all_questions):
            if i < len(old_pairs) and isinstance(old_pairs[i], dict):
                entry = old_pairs[i]
                text = entry.get("incomplete_question")
                if (entry.get("original_question") == original
                        and entry.get("similar_question_id") == expected_ids[i]
                        and isinstance(text, str) and self._is_incomplete_valid(text, original)):
                    incomplete_list[i] = text
        if all(text is not None for text in incomplete_list):
            return old_pairs[:len(all_questions)], path

        info_to_remove = None
        if self.incomplete_blank_strategy in ("guided", "info", "info_type", "guidance"):
            # Keep the same removal instruction when repairing part of a cached group.
            info_to_remove = cached.get("info_removed_type") if any(incomplete_list) else None
            if not info_to_remove:
                info_to_remove = identify_information_to_remove(question, self.openrouter_model)
        elif self.incomplete_blank_strategy in _PERTURB_STRATEGIES:
            info_to_remove = self.incomplete_blank_strategy
        else:
            info_to_remove = f"importance_top_{self.incomplete_num_blanks}"
        pairs: List[Dict] = []
        missing = [i for i, text in enumerate(incomplete_list) if text is None]
        generated = self._generate_incomplete_questions(
            [all_questions[i] for i in missing], info_to_remove,
        )
        for i, text in zip(missing, generated):
            incomplete_list[i] = text

        original_incomplete = incomplete_list[0] if incomplete_list else None
        pairs.append(
            {
                "dataset": data_item.get("data_source", "unknown"),
                "original_question_id": 0,
                "similar_question_id": None,
                "type": "original",
                "original_question": question,
                "incomplete_question": original_incomplete,
                "info_removed": info_to_remove,
            }
        )

        for idx, similar in enumerate(similar_entry.get("similar_questions", []), start=1):
            incomplete = incomplete_list[idx] if idx < len(incomplete_list) else None
            pairs.append(
                {
                    "dataset": data_item.get("data_source", "unknown"),
                    "original_question_id": 0,
                    "similar_question_id": similar["id"],
                    "type": "similar",
                    "original_question": similar["question"],
                    "incomplete_question": incomplete,
                    "info_removed": info_to_remove,
                }
            )

        payload = {
            "original_question": question,
            "info_removed_type": info_to_remove,
            "generator_model": self.openrouter_model,
            "incomplete_blank_strategy": self.incomplete_blank_strategy,
            "incomplete_num_blanks": self.incomplete_num_blanks,
            "num_pairs": len(pairs),
            "pairs": pairs,
            "num_valid_similars": self._count_valid_similars(pairs),
        }
        self._write_cache(path, payload)
        return pairs, path

    def _build_incomplete_mapping(self, pairs: List[Dict]) -> Dict[Tuple[int, int], Dict]:
        mapping: Dict[Tuple[int, int], Dict] = {}
        for entry in pairs:
            orig_id = entry.get("original_question_id", 0)
            similar_id = entry.get("similar_question_id")
            key = (orig_id, similar_id if entry.get("type") == "similar" else None)
            incomplete = entry.get("incomplete_question")
            if not isinstance(incomplete, str) or not incomplete.strip():
                continue
            if not self._is_incomplete_valid(incomplete, entry.get("original_question")):
                continue
            mapping[key] = {
                "complete": entry["original_question"],
                "incomplete": incomplete,
            }
        return mapping

    def _count_valid_similars(self, pairs: List[Dict]) -> int:
        count = 0
        for entry in pairs:
            if entry.get("type") != "similar":
                continue
            incomplete = entry.get("incomplete_question") or ""
            if self._is_incomplete_valid(incomplete, entry.get("original_question")):
                count += 1
        return count

    def _generate_incomplete_questions(
        self, all_questions: List[str], info_to_remove: Optional[str]
    ) -> List[Optional[str]]:
        if not all_questions:
            return []

        incomplete_list = [None] * len(all_questions)
        strategy = self.incomplete_blank_strategy
        for attempt in range(self.max_incomplete_retries + 1):
            try:
                if strategy in _BLANK_STRATEGIES:
                    if strategy in ("guided", "info", "info_type", "guidance"):
                        batch_out = generate_incomplete_questions_with_guidance_batch(
                            all_questions, str(info_to_remove or ""), self.openrouter_model
                        )
                    else:
                        batch_out = generate_incomplete_questions_by_importance_batch(
                            all_questions,
                            num_blanks=self.incomplete_num_blanks,
                            model=self.openrouter_model,
                        )
                elif strategy == "num_replace":
                    batch_out = generate_perturbed_questions_num_replace_batch(
                        all_questions, model=self.openrouter_model
                    )
                elif strategy == "var_rename":
                    batch_out = generate_perturbed_questions_var_rename_batch(
                        all_questions, model=self.openrouter_model
                    )
                elif strategy == "distractor":
                    batch_out = generate_perturbed_questions_distractor_batch(
                        all_questions, model=self.openrouter_model
                    )
                else:
                    raise ValueError(f"Unknown incomplete_blank_strategy: {strategy}")
                if len(batch_out) != len(all_questions):
                    raise ValueError("Batch output length mismatch")
                for i, out in enumerate(batch_out):
                    if out:
                        incomplete_list[i] = out
            except Exception:
                def _work(q: str) -> Optional[str]:
                    if strategy in ("guided", "info", "info_type", "guidance"):
                        return generate_incomplete_question_with_guidance(
                            q,
                            str(info_to_remove or ""),
                            self.openrouter_model,
                        )
                    if strategy == "important":
                        return generate_incomplete_question_by_importance(
                            q,
                            num_blanks=self.incomplete_num_blanks,
                            model=self.openrouter_model,
                        )
                    if strategy == "num_replace":
                        return generate_perturbed_questions_num_replace_batch(
                            [q], model=self.openrouter_model
                        )[0]
                    if strategy == "var_rename":
                        return generate_perturbed_questions_var_rename_batch(
                            [q], model=self.openrouter_model
                        )[0]
                    if strategy == "distractor":
                        return generate_perturbed_questions_distractor_batch(
                            [q], model=self.openrouter_model
                        )[0]
                    return None

                max_workers = min(self.max_openrouter_workers, len(all_questions))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(_work, q): idx
                        for idx, q in enumerate(all_questions)
                    }
                    for future in as_completed(future_map):
                        idx = future_map[future]
                        try:
                            incomplete_list[idx] = future.result()
                        except Exception:
                            incomplete_list[idx] = None

            if all(
                q is not None and self._is_incomplete_valid(q, all_questions[i])
                for i, q in enumerate(incomplete_list)
            ):
                break
        return incomplete_list

    def _has_blank(self, text: str) -> bool:
        return "[BLANK]" in text or "[blank]" in text.lower()

    def _is_incomplete_valid(self, text: str, original: Optional[str] = None) -> bool:
        if not text or not text.strip():
            return False
        strategy = self.incomplete_blank_strategy
        if strategy in ("guided", "info", "info_type", "guidance"):
            return self._has_blank(text)
        if strategy == "important":
            return text.count("[BLANK]") == self.incomplete_num_blanks
        if strategy in _PERTURB_STRATEGIES and original:
            return text.strip() != original.strip()
        if strategy in _PERTURB_STRATEGIES:
            return True
        return self._has_blank(text)

    def _get_or_create_rsi_variants_unlocked(
        self, similar_entry: Dict, pairs: List[Dict]
    ) -> Tuple[List[Dict], str]:
        question = similar_entry["original_question"]
        suffix = f"incomplete_variants__{self.incomplete_blank_strategy}__k{self.incomplete_num_blanks}.json"
        path = self._build_output_path(question, suffix=suffix)
        # Pair contents and generation settings determine variant-cache compatibility.
        fingerprint = hashlib.sha256(json.dumps(
            {"pairs": pairs, "model": self.openrouter_model, "n": self.num_rsi_variants,
             "min_valid_similars": self.min_valid_similars}, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        cached = self._read_cache(path) if self.reuse_existing else {}
        valid_pairs = [p for p in pairs if isinstance(p.get("incomplete_question"), str)
                       and self._is_incomplete_valid(p["incomplete_question"], p.get("original_question"))]
        cached_entries = cached.get("entries", []) if cached.get("fingerprint") == fingerprint else []
        if not isinstance(cached_entries, list):
            cached_entries = []
        reusable = {}
        for entry in cached_entries:
            if not isinstance(entry, dict):
                continue
            variants = entry.get("variants", [])
            if (isinstance(variants, list) and len(variants) == self.num_rsi_variants
                    and all(isinstance(v, dict) and isinstance(v.get("variant"), str)
                            and v["variant"].strip() for v in variants)):
                reusable[(entry.get("type"), entry.get("id"))] = entry
        if valid_pairs and len(reusable) == len(valid_pairs):
            return list(reusable.values()), path

        if self.min_valid_similars and self._count_valid_similars(pairs) < self.min_valid_similars:
            payload = {
                "original_question": question,
                "num_entries": 0,
                "entries": [],
                "skipped": True,
                "reason": "insufficient_valid_similars",
                "min_valid_similars": self.min_valid_similars,
            }
            self._write_cache(path, payload)
            return [], path

        def _build_variants_payload(entry: Dict) -> Optional[Dict]:
            incomplete_question = entry.get("incomplete_question")
            if not isinstance(incomplete_question, str) or not self._is_incomplete_valid(
                    incomplete_question, entry.get("original_question")):
                return None
            try:
                if self._has_blank(incomplete_question):
                    variants = generate_incomplete_variants_batch(
                        incomplete_question,
                        num_variants=self.num_rsi_variants,
                        model=self.openrouter_model,
                    )
                else:
                    variants = generate_question_variants_batch(
                        incomplete_question,
                        num_variants=self.num_rsi_variants,
                        model=self.openrouter_model,
                    )
                strategy = "paraphrase_batch"
            except Exception:
                if self._has_blank(incomplete_question):
                    variants, _ = generate_incomplete_variants(
                        incomplete_question,
                        num_variants=self.num_rsi_variants,
                        model=self.openrouter_model,
                    )
                else:
                    variants = generate_question_variants(
                        incomplete_question,
                        num_variants=self.num_rsi_variants,
                        model=self.openrouter_model,
                    )
                strategy = "mixed_fallback"

            entry_id = entry["similar_question_id"] if entry.get("type") == "similar" else 0
            return {
                "dataset": entry.get("dataset"),
                "original_question_id": entry.get("original_question_id", 0),
                "type": entry.get("type"),
                "id": entry_id,
                "original_incomplete_question": incomplete_question,
                "variants": [
                    {"id": i + 1, "variant": v, "strategy": strategy}
                    for i, v in enumerate(variants)
                ],
            }

        entries: List[Dict] = list(reusable.values())
        pending = [entry for entry in valid_pairs if (
            entry.get("type"), entry.get("similar_question_id") if entry.get("type") == "similar" else 0
        ) not in reusable]
        if pending:
            max_workers = min(self.max_openrouter_workers, len(pending))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_build_variants_payload, entry): entry for entry in pending}
                for future in as_completed(future_map):
                    payload = future.result()
                    if payload:
                        entries.append(payload)

        payload = {
            "original_question": question,
            "fingerprint": fingerprint,
            "num_entries": len(entries),
            "entries": entries,
        }
        self._write_cache(path, payload)
        return entries, path

    def _build_rsi_group(self, variants_entries: List[Dict]) -> Dict:
        if not variants_entries:
            return {"original": None, "similars": {}}
        groups = rsi_calculation.organize_variants_by_group(variants_entries)
        return groups.get(0, {"original": None, "similars": {}})

    def _ensure_rsm_ready(self):
        if not self._rsm_ready:
            cls = type(self)
            if not self.model_name:
                raise ValueError(
                    "RepStiff requires a Hugging Face model id/path. "
                    "Pass --rep_stiff_model_name (e.g., $HF_MODEL_ID) to evaluate_all_methods."
                )
            if (
                cls._shared_model is not None
                and cls._shared_model_name == self.model_name
                and cls._shared_device_name == self.device
            ):
                rsm_calculation.model = cls._shared_model
                rsm_calculation.tokenizer = cls._shared_tokenizer
                rsm_calculation.LAYER_MAP = cls._shared_layer_map
                rsm_calculation.DEVICE = cls._shared_device
            else:
                rsm_calculation.DEVICE = self.device
                rsm_calculation.initialize_model(self.model_name)
                cls._shared_model = rsm_calculation.model
                cls._shared_tokenizer = rsm_calculation.tokenizer
                cls._shared_layer_map = rsm_calculation.LAYER_MAP
                cls._shared_device = rsm_calculation.DEVICE
                cls._shared_model_name = self.model_name
                cls._shared_device_name = self.device
            self._rsm_ready = True

    def _ensure_dc_ready(self):
        if not self._dc_ready:
            cls = type(self)
            if (
                cls._shared_model is not None
                and cls._shared_model_name == self.model_name
                and cls._shared_device_name == self.device
            ):
                dc_calculation.model = cls._shared_model
                dc_calculation.tokenizer = cls._shared_tokenizer
                dc_calculation.LAYER_MAP = cls._shared_layer_map
                dc_calculation.DEVICE = cls._shared_device
                self._dc_ready = True
                return
            if self._rsm_ready and rsm_calculation.model is not None:
                dc_calculation.model = rsm_calculation.model
                dc_calculation.tokenizer = rsm_calculation.tokenizer
                dc_calculation.LAYER_MAP = rsm_calculation.LAYER_MAP
                dc_calculation.DEVICE = rsm_calculation.DEVICE
                self._dc_ready = True
                return
            if not self.model_name:
                raise ValueError(
                    "RepStiff requires a Hugging Face model id/path. "
                    "Pass --rep_stiff_model_name (e.g., $HF_MODEL_ID) to evaluate_all_methods."
                )
            dc_calculation.initialize_model(self.model_name)
            self._dc_ready = True

    def _ensure_rsi_ready(self):
        if not self._rsi_ready:
            cls = type(self)
            if (
                cls._shared_model is not None
                and cls._shared_model_name == self.model_name
                and cls._shared_device_name == self.device
            ):
                rsi_calculation.model = cls._shared_model
                rsi_calculation.tokenizer = cls._shared_tokenizer
                rsi_calculation.LAYER_MAP = cls._shared_layer_map
                rsi_calculation.DEVICE = cls._shared_device
                self._rsi_ready = True
                return
            if self._rsm_ready and rsm_calculation.model is not None:
                rsi_calculation.model = rsm_calculation.model
                rsi_calculation.tokenizer = rsm_calculation.tokenizer
                rsi_calculation.LAYER_MAP = rsm_calculation.LAYER_MAP
                rsi_calculation.DEVICE = rsm_calculation.DEVICE
                self._rsi_ready = True
                return
            if not self.model_name:
                raise ValueError(
                    "RepStiff requires a Hugging Face model id/path. "
                    "Pass --rep_stiff_model_name (e.g., $HF_MODEL_ID) to evaluate_all_methods."
                )
            rsi_calculation.initialize_model(self.model_name)
            self._rsi_ready = True

    def _build_output_path(self, question: str, suffix: str) -> str:
        question_id = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
        filename = f"{question_id}_{suffix}"
        return os.path.join(self.output_dir, filename)