"""Build CPU or CUDA packages with bob using the selected local Windows LibTorch SDK."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


CPU_LIBRARIES = ("torch", "torch_cpu", "c10")
CUDA_LIBRARIES = (*CPU_LIBRARIES, "torch_cuda", "c10_cuda")
VERSION_HEADER = Path("include/torch/csrc/api/include/torch/version.h")
CPU_HEADERS = (
    VERSION_HEADER,
    Path("include/torch/csrc/api/include/torch/torch.h"),
)
CUDA_HEADERS = (
    *CPU_HEADERS,
    Path("include/c10/cuda/CUDAMacros.h"),
    Path("include/ATen/cuda/CUDAConfig.h"),
)


def read_device(project: Path) -> str:
    path = project / "rlbot/device.txt"
    mode = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    if mode not in ("cpu", "cuda"):
        raise ValueError("rlbot/device.txt must contain cpu or cuda before building.")
    return mode


def sdk_version(root: Path, mode: str) -> str:
    """Check the selected Windows SDK without requiring the other device's archive."""
    libraries = CUDA_LIBRARIES if mode == "cuda" else CPU_LIBRARIES
    required = list(CUDA_HEADERS if mode == "cuda" else CPU_HEADERS)
    if mode == "cuda":
        required.append(Path("build-version"))
    for name in libraries:
        required.extend((Path("lib") / (name + ".lib"), Path("lib") / (name + ".dll")))
    for relative in required:
        if not (root / relative).is_file() or (root / relative).stat().st_size == 0:
            raise ValueError(f"Missing {mode} SDK file: {relative.as_posix()}")
    if mode == "cpu" and (root / "lib/torch_cuda.dll").exists():
        raise ValueError("CPU builds require CPU LibTorch, not a CUDA SDK.")

    header = (root / VERSION_HEADER).read_text(encoding="utf-8")
    components = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        macro = re.search(r"^\s*#define\s+TORCH_VERSION_" + component + r"\s+(\d+)\b", header, re.MULTILINE)
        if not macro:
            raise ValueError("Missing LibTorch version macros.")
        components.append(macro.group(1))
    header_version = ".".join(components)
    if mode == "cpu":
        return header_version + "+cpu"
    version = (root / "build-version").read_text(encoding="utf-8").strip()
    if not re.fullmatch(re.escape(header_version) + r"\+cu\d+", version):
        raise ValueError("Expected a release CUDA build-version matching the LibTorch headers.")
    return version


def sdk_candidates(project: Path, mode: str, explicit: Path | None, env: dict[str, str]):
    # Explicit overrides are exclusive: do not accidentally build against another SDK.
    override = explicit or env.get("LIBTORCH_CUDA_ROOT" if mode == "cuda" else "LIBTORCH_CPU_ROOT")
    if override:
        yield Path(override).expanduser()
        return

    # Match the launcher's nearby-bot search, followed by the standard bot-pack location.
    subdir = "libtorch" if mode == "cuda" else "torch-archive/torch"
    directory = project / "rlbot"
    for _ in range(5):
        yield directory / subdir
        if directory.parent == directory:
            break
        directory = directory.parent
    if env.get("LOCALAPPDATA"):
        yield Path(env["LOCALAPPDATA"]) / "RLBot5/bots" / subdir


def find_sdk(project: Path, mode: str, explicit: Path | None, env: dict[str, str]):
    reason = "No complete local Windows LibTorch SDK was found."
    for root in sdk_candidates(project, mode, explicit, env):
        try:
            return root, sdk_version(root, mode)
        except (OSError, ValueError) as error:
            reason = str(error)
    raise ValueError(f"{mode.upper()} builds require a complete Windows {mode} LibTorch SDK. "
                     "Install it in the bot pack or use --libtorch. " + reason)


@contextmanager
def staged_sdk(project: Path, mode: str, sdk):
    """Keep SDK data private to this build; only a content fingerprint enters bob's hash."""
    project = project.resolve()
    stage = project / "build-support/libtorch/local"
    # Never reuse or recursively remove an existing directory or an external junction.
    if not stage.parent.resolve().is_relative_to(project):
        raise ValueError("The LibTorch staging directory must be inside this project.")
    stage.parent.mkdir(parents=True, exist_ok=True)
    try:
        stage.mkdir()
    except FileExistsError as error:
        raise ValueError("build-support/libtorch/local already exists. Another helper may be running; "
                         "if an earlier build was interrupted, remove that generated directory and retry.") from error
    try:
        source, version = sdk
        shutil.copytree(source / "include", stage / "include")
        (stage / "lib").mkdir()
        libraries = CUDA_LIBRARIES if mode == "cuda" else CPU_LIBRARIES
        for name in libraries:
            shutil.copy2(source / "lib" / (name + ".lib"), stage / "lib" / (name + ".lib"))
        (stage / "build-version").write_text(version + "\n", encoding="utf-8")

        # bob ignores Git-ignored SDK files. Hash their contents into a temporary,
        # non-ignored manifest so changing SDKs invalidates its executable cache.
        digest = hashlib.sha256()
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            digest.update(path.relative_to(stage).as_posix().encode("utf-8") + b"\0")
            file_hash = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    file_hash.update(block)
            digest.update(file_hash.digest())
        manifest = {"format": 1, "device": mode, "version": version, "sha256": digest.hexdigest()}
        (stage / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        yield stage
    finally:
        if stage.resolve() != stage or not stage.resolve().is_relative_to(project):
            raise ValueError("Refusing to clean a LibTorch staging directory that moved outside the project.")
        shutil.rmtree(stage)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bob", default="bob", help="bob executable name or path (default: bob on PATH)")
    parser.add_argument("--libtorch", type=Path, help="Selected Windows LibTorch root; overrides bot-pack discovery")
    parser.add_argument("--out-dir", default="bob_build", help="bob output directory (default: bob_build)")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parent.parent
    try:
        mode = read_device(project)
        sdk = find_sdk(project, mode, args.libtorch, os.environ)
        executable = shutil.which(args.bob)
        if not executable:
            raise ValueError("bob was not found. Supply its executable with --bob.")
        with staged_sdk(project, mode, sdk):
            platforms = "Windows GPU" if mode == "cuda" else "Windows + Linux CPU"
            print(f"bob: {platforms} (Windows LibTorch {sdk[1]})", flush=True)
            return subprocess.run([executable, "build", "bob.toml", "--out-dir", args.out_dir],
                                  cwd=project, check=False).returncode
    except (OSError, ValueError) as error:
        print(f"Build error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
