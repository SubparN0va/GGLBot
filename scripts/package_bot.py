"""Create a source submission zip without building the bot or installing LibTorch."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from build_bob import read_device


PACKAGE_PATHS = (
    ".dockerignore", ".gitignore", "README.md", "bob.toml",
    "CMakeLists.txt", "CMakePresets.json", "cpp.Dockerfile",
    "build-support/libtorch/README.md", "cmake", "cpp-interface", "inc",
    "launcher", "rlbot", "scripts", "src", "tests",
)
EXCLUDED_DIRECTORIES = {
    ".git", ".vs", ".vscode", ".idea", ".venv", "venv", "__pycache__",
    "build", "out", "bob_build", "CMakeFiles", "_deps", "Testing",
    "000-runtime", "libtorch", "torch-archive",
}
EXCLUDED_FILES = {
    "CMakeCache.txt", "CMakeSettings.json", "CMakeUserPresets.json",
    "CMakeLists.txt.user", "build.ninja", ".ninja_deps", ".ninja_log",
    "cmake_install.cmake", "install_manifest.txt", "compile_commands.json",
    "CTestTestfile.cmake",
}
COMPILED_SUFFIXES = {".exe", ".dll", ".lib", ".obj", ".o", ".a", ".so", ".pdb", ".ilk", ".pyc", ".pyd"}


def submission_files(project: Path, output: Path) -> list[Path]:
    for name in PACKAGE_PATHS:
        if not (project / name).exists():
            raise ValueError(f"Missing submission input: {name}")
    for name in ("cpp-interface/library/CMakeLists.txt", "inc/RocketSim/src/RocketSim.h"):
        if not (project / name).is_file():
            raise ValueError("Missing submodule sources. Run git submodule update --init --recursive and retry.")
    read_device(project)

    def check_local(path: Path):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Copy linked files into the project before packaging: {path.relative_to(project)}")

    def walk_error(error):
        raise error

    candidates = []
    for name in PACKAGE_PATHS:
        root = project / name
        check_local(root)
        if root.is_file():
            candidates.append(root)
            continue
        for directory, dirs, files in os.walk(root, onerror=walk_error):
            dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRECTORIES and not name.startswith("cmake-build-")]
            for name in dirs:
                check_local(Path(directory) / name)
            candidates.extend(Path(directory) / name for name in files)

    included = []
    for path in sorted(candidates):
        if (path.name == ".git" or path.name in EXCLUDED_FILES
                or path.suffix.lower() in COMPILED_SUFFIXES or ".so." in path.name):
            continue
        check_local(path)
        if path.resolve() == output:
            continue
        # Linux executables have no required extension; keep them out of source zips too.
        with path.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                continue
        included.append(path)
    if not any(path.is_relative_to(project / "rlbot") and path.suffix == ".lt" for path in included):
        raise ValueError("No .lt models found in rlbot/. Add your trained models before packaging.")
    return included


def package_bot(project: Path, output: Path) -> int:
    project, output = project.resolve(), output.resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("The output filename must end in .zip.")
    files = submission_files(project, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Replace an earlier submission only after the new archive has been written successfully.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip.tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, path.relative_to(project).as_posix())
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return len(files)


def main(argv=None) -> int:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=project / "submission.zip", help="Output zip (default: submission.zip in the project root)")
    args = parser.parse_args(argv)
    try:
        count = package_bot(project, args.output)
        print(f"Created {args.output.resolve()} ({count} files). Send this zip to the host.")
        return 0
    except (OSError, ValueError) as error:
        print(f"Packaging error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
