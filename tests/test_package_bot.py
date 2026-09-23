"""Check that source submissions include models and dependencies without local build artifacts."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
import package_bot


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gglbot package ")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name).resolve()
        self.output = self.project / "submission.zip"
        directories = {"cmake", "cpp-interface", "inc", "launcher", "rlbot", "scripts", "src", "tests"}
        for name in package_bot.PACKAGE_PATHS:
            if name in directories:
                (self.project / name).mkdir()
            else:
                self.write(name)
        self.write("cpp-interface/library/CMakeLists.txt")
        self.write("inc/RocketSim/src/RocketSim.h")
        self.write("rlbot/device.txt", b"gpu\n")
        self.write("rlbot/POLICY.lt", b"trained model")
        self.write("rlbot/bot.toml")
        self.write("rlbot/loadout.toml")

    def write(self, name, content=b"fixture"):
        path = self.project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_packages_current_sources_models_and_submodules_without_git(self):
        self.write(".gitignore", b"*.lt\nrlbot/logo.png\n")
        self.write("inc/CustomObs.h", b"uncommitted custom observation builder")
        self.write("rlbot/logo.png", b"custom logo")
        self.write("rlbot/models/SHARED_HEAD.lt", b"another model")
        count = package_bot.package_bot(self.project, self.output)
        with ZipFile(self.output) as archive:
            self.assertEqual(count, len(archive.namelist()))
            self.assertIsNone(archive.testzip())
            for name in ("bob.toml", "cpp-interface/library/CMakeLists.txt", "inc/RocketSim/src/RocketSim.h",
                         "rlbot/POLICY.lt", "rlbot/models/SHARED_HEAD.lt", "rlbot/logo.png", "inc/CustomObs.h"):
                self.assertEqual(archive.read(name), (self.project / name).read_bytes())

    def test_excludes_build_outputs_metadata_and_torch_even_inside_source_folders(self):
        excluded = [
            ".git/config", "cpp-interface/.git", "inc/RocketSim/.git",
            "cpp-interface/build/nested/cache", "src/cmake-build-debug/generated.h",
            "inc/RocketSim/CMakeFiles/generated.cpp", "inc/RocketSim/CMakeCache.txt",
            "inc/RocketSim/CMakeSettings.json", "rlbot/GGLBot.exe", "rlbot/runtime.dll",
            "rlbot/libtorch_cpu.so.1", "rlbot/000-runtime/GGLBotCoreCPU",
            "rlbot/libtorch/include/torch.h", "rlbot/torch-archive/torch/include/torch.h",
            "build-support/libtorch/local/manifest.json", "build-support/libtorch/local/lib/torch.lib",
            "scripts/__pycache__/build_bob.pyc", "out/build/anything", "bob_build/anything",
            "src/.venv/site-packages/anything",
        ]
        for name in excluded:
            self.write(name)
        self.write("rlbot/GGLBot", b"\x7fELFbinary")
        package_bot.package_bot(self.project, self.output)
        # A second run must not zip its own output.
        package_bot.package_bot(self.project, self.output)
        with ZipFile(self.output) as archive:
            self.assertTrue(set(excluded).isdisjoint(archive.namelist()))
            self.assertNotIn("rlbot/GGLBot", archive.namelist())
            self.assertNotIn("submission.zip", archive.namelist())
            self.assertIn("build-support/libtorch/README.md", archive.namelist())

    def test_missing_model_or_submodule_preserves_previous_zip(self):
        package_bot.package_bot(self.project, self.output)
        previous = self.output.read_bytes()
        for name, message in [("rlbot/POLICY.lt", "No .lt models"),
                              ("inc/RocketSim/src/RocketSim.h", "submodule")]:
            path = self.project / name
            data = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(ValueError, message):
                package_bot.package_bot(self.project, self.output)
            self.assertEqual(self.output.read_bytes(), previous)
            path.write_bytes(data)

    def test_failed_archive_write_preserves_previous_zip_and_cleans_temporary_file(self):
        package_bot.package_bot(self.project, self.output)
        previous = self.output.read_bytes()
        with patch.object(package_bot.ZipFile, "write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                package_bot.package_bot(self.project, self.output)
        self.assertEqual(self.output.read_bytes(), previous)
        self.assertEqual(list(self.project.glob("*.zip.tmp")), [])

    def test_cli_runs_outside_project_without_build_tools(self):
        for name in ("package_bot.py", "build_bob.py"):
            (self.project / "scripts" / name).write_bytes((PROJECT / "scripts" / name).read_bytes())
        result = subprocess.run([sys.executable, str(self.project / "scripts/package_bot.py")],
                                cwd=self.project.parent, env={"PATH": ""},
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.exists())
        self.assertIn("Send this zip to the host", result.stdout)


if __name__ == "__main__":
    unittest.main()
