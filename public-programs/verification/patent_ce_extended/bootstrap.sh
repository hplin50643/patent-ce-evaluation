#!/usr/bin/env bash
set -euo pipefail
task_script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
task_uv="$HOME/.local/bin/uv"
task_environment="$HOME/.venvs/patent-ce-13methods-py310"
if [[ ! -x "$task_uv" ]]; then
  curl --fail --silent --show-error --location https://astral.sh/uv/0.12.4/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
"$task_uv" python install 3.10.20 --no-bin
if [[ ! -x "$task_environment/bin/python" ]]; then
  "$task_uv" venv "$task_environment" --python 3.10.20 --managed-python --no-python-downloads --no-project
fi
"$task_uv" pip sync --python "$task_environment/bin/python" --require-hashes "$task_script_dir/requirements-lock.txt"
exec "$task_environment/bin/python" "$task_script_dir/assets_setup.py" "$@"
