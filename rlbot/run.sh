#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Equivalent to Windows %LOCALAPPDATA%\RLBot5\bots\torch-archive\torch\lib
export LD_LIBRARY_PATH="${XDG_DATA_HOME:-$HOME/.local/share}/RLBot5/bots/torch-archive/torch/lib"

exec "$SCRIPT_DIR/GGLBot" "$@"