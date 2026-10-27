"""
Hugging Face + vLLM compatibility: expose `all_special_tokens_extended`.

Some tokenizer builds (e.g. Qwen2Tokenizer on older Transformers) do not define
that attribute; vLLM still reads it inside EngineCore worker processes.

`get_cached_tokenizer` patching in the parent alone is insufficient when vLLM
uses multiprocessing spawn for v1 — workers need this fix at Transformers layer.

Applied:
- Explicitly via `generate_full_data.py` imports.
- Automatically in subprocesses via `compat_site/sitecustomize.py` if
  PYTHONPATH includes that directory (`generate_full_data` sets this before LLM startup).
"""

from __future__ import annotations

_applied = False


def apply_transformers_special_tokens_extended_getattr_compat() -> None:
    global _applied
    if _applied:
        return
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return

    _orig_getattr = getattr(PreTrainedTokenizerBase, "__getattr__", None)
    if _orig_getattr is None:
        _applied = True
        return

    def __getattr__compat(self, key):
        if key == "all_special_tokens_extended":
            try:
                return list(self.all_special_tokens)
            except Exception:
                return []
        return _orig_getattr(self, key)

    PreTrainedTokenizerBase.__getattr__ = __getattr__compat  # type: ignore[assignment]
    _applied = True
