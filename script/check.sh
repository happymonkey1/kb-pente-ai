#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
run_venv="$script_dir/run-venv.sh"

cd -- "$repository_root"

bash -n "$script_dir/run-venv.sh" "$script_dir/check.sh"
git diff --check
UV_CACHE_DIR="${TMPDIR:-/tmp}/kb-pente-ai-uv-check" uv lock --check --offline
"$run_venv" python -m compileall -q main.py src script
"$run_venv" mypy src main.py script/*.py --no-error-summary
"$run_venv" python -m unittest discover -s src -p '*_test.py'
