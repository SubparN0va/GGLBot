#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Equivalent to Windows %LOCALAPPDATA%\RLBot5\bots\libtorch_cpu\lib
TORCH_LIB="${XDG_DATA_HOME:-$HOME/.local/share}/RLBot5/bots/libtorch_cpu/lib"

# Wine uses PATH to locate DLLs (closest equivalent to Windows PATH)
export PATH="$TORCH_LIB:$PATH"

cd "$SCRIPT_DIR"
exec wine "$SCRIPT_DIR/GGLBot.exe" "$@"