"""Exercise each compiled launcher with only its selected stand-in core and runtime."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

LAUNCHER, CORE = map(Path, sys.argv[1:3])
DEVICE = sys.argv[3]
CUDA = DEVICE == 'CUDA'
WINDOWS = os.name == 'nt'
SUFFIX = '.exe' if WINDOWS else ''
MARKER = 'torch_cuda.dll' if CUDA else ('torch_cpu.dll' if WINDOWS else 'libtorch_cpu.so')
OVERRIDE = 'LIBTORCH_CUDA_ROOT' if CUDA else 'LIBTORCH_CPU_ROOT'
SUBDIR = 'libtorch' if CUDA else 'torch-archive/torch'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gglbot launcher ')
        self.addCleanup(self.temp.cleanup)
        # Keep all five search roots inside this fixture, away from installed runtimes.
        self.root = Path(self.temp.name) / 'isolated/nested/root'
        self.bot = self.root / 'pack/bot with spaces'
        (self.bot / '000-runtime').mkdir(parents=True)
        self.launcher = self.bot / ('GGLBot' + SUFFIX)
        self.core = self.bot / '000-runtime' / ('GGLBotCore' + DEVICE + SUFFIX)
        shutil.copy2(LAUNCHER, self.launcher)
        shutil.copy2(CORE, self.core)
        self.env = os.environ.copy()
        self.env.pop('LIBTORCH_CPU_ROOT', None)
        self.env.pop('LIBTORCH_CUDA_ROOT', None)
        self.env['LOCALAPPDATA'] = str(self.root / 'data')
        self.env['XDG_DATA_HOME'] = str(self.root / 'data')
        self.env['GGLBOT_TEST_TRACE'] = str(self.root / 'trace.txt')
        self.runtime = self.install_runtime(self.root / 'pack' / SUBDIR)

    def install_runtime(self, root):
        directory = root / 'lib'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MARKER).touch()
        self.env['GGLBOT_TEST_' + DEVICE + '_PATH'] = str(directory)
        return directory

    def run_bot(self, *args):
        return subprocess.run([str(self.launcher), *args], env=self.env, cwd=self.root,
                              text=True, capture_output=True, timeout=15)

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'GGLBot: using ' + ('GPU' if CUDA else 'CPU') + '\n')
        self.assertEqual(result.stderr, '')

    def trace(self):
        path = self.root / 'trace.txt'
        return path.read_text().splitlines() if path.exists() else []

    def test_only_selected_runtime_and_core_are_needed(self):
        self.assert_success(self.run_bot())
        self.assertEqual(self.trace(), [DEVICE])

    def test_selection_is_baked_into_launcher(self):
        (self.bot / 'device.txt').write_text('cpu' if CUDA else 'gpu')
        self.assert_success(self.run_bot())
        self.assertEqual(self.trace(), [DEVICE])

    def test_quoted_arguments(self):
        argument = 'argument with spaces "quoted" trailing\\'
        self.env['GGLBOT_TEST_ARGUMENT'] = argument
        self.assert_success(self.run_bot(argument))

    def test_missing_runtime(self):
        (self.runtime / MARKER).unlink()
        result = self.run_bot()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(OVERRIDE, result.stderr)
        self.assertEqual(self.trace(), [])

    def test_default_bot_pack_directory(self):
        (self.runtime / MARKER).unlink()
        self.install_runtime(self.root / 'data/RLBot5/bots' / SUBDIR)
        self.assert_success(self.run_bot())

    def test_runtime_override(self):
        root = self.root / 'custom runtime with spaces'
        self.install_runtime(root)
        self.env[OVERRIDE] = str(root)
        self.assert_success(self.run_bot())

    def test_invalid_override_does_not_use_bot_pack(self):
        self.env[OVERRIDE] = str(self.root / 'missing')
        result = self.run_bot()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(OVERRIDE, result.stderr)
        self.assertEqual(self.trace(), [])

    def test_other_runtime_override_is_ignored(self):
        self.env['LIBTORCH_CPU_ROOT' if CUDA else 'LIBTORCH_CUDA_ROOT'] = str(self.root / 'missing')
        self.assert_success(self.run_bot())

    def test_missing_core(self):
        self.core.unlink()
        result = self.run_bot()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('core executable not found', result.stderr)

    @unittest.skipUnless(CUDA, 'CUDA launcher only')
    def test_gpu_errors_never_start_cpu(self):
        # A stale CPU executable must not be considered even after a CUDA startup error.
        shutil.copy2(CORE, self.bot / '000-runtime/GGLBotCoreCPU.exe')
        for status in ('75', '0xc0000135', '19'):
            self.env['GGLBOT_TEST_CUDA_EXIT'] = status
            result = self.run_bot()
            self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.trace(), ['CUDA'] * 3)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
