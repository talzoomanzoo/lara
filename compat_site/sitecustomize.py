"""
Loaded automatically by Python startup when `compat_site/` is first on PYTHONPATH.

Ensures vLLM EngineCore worker processes get the tokenizer compatibility patch.
See `tokenizer_extended_compat.py`.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokenizer_extended_compat import apply_transformers_special_tokens_extended_getattr_compat

apply_transformers_special_tokens_extended_getattr_compat()
