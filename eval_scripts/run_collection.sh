#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Prefer this workspace's evaluation environment; PYTHON explicitly overrides it.
if [[ -z "${PYTHON:-}" && -x "$HOME/miniconda3/envs/eval/bin/python" ]]; then
    PYTHON="$HOME/miniconda3/envs/eval/bin/python"
fi
exec "${PYTHON:-python}" "$SCRIPT_DIR/collection/run.py" "$@"
