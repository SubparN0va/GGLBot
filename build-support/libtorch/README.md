This directory supplies Docker with the selected local Windows LibTorch SDK.

`scripts/build_bob.py` reads `rlbot/device.txt` (`cpu` or `gpu`, default `gpu`) and temporarily
creates `local/` containing headers, required import libraries, and version
metadata. CPU uses `torch-archive/torch`; CUDA uses `libtorch`. The helper never
copies runtime DLLs and removes `local/` after bob exits. Run the helper for
both devices; the Docker build requires its staged SDK.

The generated `local/manifest.json` stays visible to bob's Git-aware directory
hasher. It contains the device, version, and content fingerprint, never the
SDK's original path. Do not commit generated files. If the helper was forcibly
terminated, make sure it has stopped before removing a leftover `local/`.
