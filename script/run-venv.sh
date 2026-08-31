#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
venv_activate="$repository_root/.venv/bin/activate"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [arguments ...]" >&2
    echo "Example: $0 python -m unittest discover -s src -p '*_test.py'" >&2
    exit 64
fi

if [[ ! -f "$venv_activate" ]]; then
    echo "Virtual environment not found at: $venv_activate" >&2
    echo "Create it with 'uv sync' before running this script." >&2
    exit 1
fi

temp_directory="${TMPDIR:-/tmp}"
if ! log_file="$(mktemp "$temp_directory/kb-pente-ai-venv.XXXXXX")"; then
    echo "Unable to create a temporary log file in: $temp_directory" >&2
    exit 1
fi

if ! source "$venv_activate"; then
    echo "Unable to activate virtual environment: $venv_activate" >&2
    exit 1
fi

cd -- "$repository_root" || exit 1

printf 'Log file: %s\n' "$log_file"
{
    printf 'Started: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'Working directory: %s\n' "$repository_root"
    printf 'Python: %s\n' "$(command -v python)"
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n\n'
} | tee "$log_file"

"$@" 2>&1 | tee -a "$log_file"
command_status=${PIPESTATUS[0]}

{
    printf '\nFinished: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'Exit status: %d\n' "$command_status"
} | tee -a "$log_file"

exit "$command_status"
