#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${PYTHON:-$repository_root/.venv/bin/python}"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/kb-pente-native-extension.XXXXXX")"
trap 'rm -rf "$build_root"' EXIT

MAX_JOBS="${MAX_JOBS:-2}" "$python_bin" "$script_dir/setup.py" build_ext \
    --build-temp "$build_root/temp" \
    --build-lib "$build_root/lib"
PYTHONPATH="$build_root/lib${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" "$script_dir/test_binding.py"
