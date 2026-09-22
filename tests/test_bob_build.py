"""Check exclusive SDK discovery, staging, cleanup, cache keys, and build selection."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'scripts'))
import build_bob


class BobBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gglbot bob ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'pack/nested/project'
        (self.project / 'rlbot').mkdir(parents=True)
        self.device = self.project / 'rlbot/device.txt'
        self.env = {'LOCALAPPDATA': str(self.root / 'app data')}
        pack = Path(self.env['LOCALAPPDATA']) / 'RLBot5/bots'
        self.standard = {'cpu': pack / 'torch-archive/torch', 'cuda': pack / 'libtorch'}

    def make_sdk(self, mode='cuda', root=None):
        root = root or self.standard[mode]
        headers = build_bob.CUDA_HEADERS if mode == 'cuda' else build_bob.CPU_HEADERS
        for relative in headers:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('// SDK header\n', encoding='utf-8')
        (root / build_bob.VERSION_HEADER).write_text(
            '#define TORCH_VERSION_MAJOR 2\n#define TORCH_VERSION_MINOR 14\n'
            '#define TORCH_VERSION_PATCH 0\n', encoding='utf-8')
        if mode == 'cuda':
            (root / 'build-version').write_text('2.14.0+cu126\n', encoding='utf-8')
        (root / 'lib').mkdir()
        libraries = build_bob.CUDA_LIBRARIES if mode == 'cuda' else build_bob.CPU_LIBRARIES
        for name in libraries:
            for suffix in ('.lib', '.dll'):
                (root / 'lib' / (name + suffix)).write_bytes(b'fixture')
        (root / 'lib/unused.lib').write_bytes(b'unneeded static library')
        return root

    def select(self, mode='cuda', explicit=None):
        return build_bob.find_sdk(self.project, mode, explicit, self.env)

    def test_each_device_requires_its_own_sdk(self):
        for mode in ('cpu', 'cuda'):
            with self.assertRaisesRegex(ValueError, 'require a complete'):
                self.select(mode)
        self.make_sdk('cpu')
        self.assertEqual(self.select('cpu'), (self.standard['cpu'], '2.14.0+cpu'))
        with self.assertRaises(ValueError):
            self.select('cuda')

    def test_cuda_needs_no_cpu_sdk(self):
        self.make_sdk()
        self.assertEqual(self.select(), (self.standard['cuda'], '2.14.0+cu126'))
        self.assertFalse(self.standard['cpu'].exists())

    def test_nearby_discovery_for_both_modes(self):
        for mode, subdir in [('cpu', 'torch-archive/torch'), ('cuda', 'libtorch')]:
            self.make_sdk(mode)
            nearby = self.make_sdk(mode, self.project.parent / subdir)
            self.assertEqual(self.select(mode)[0], nearby)
            (nearby / 'lib/torch.lib').unlink()
            self.assertEqual(self.select(mode)[0], self.standard[mode])

    def test_overrides_are_exclusive_and_device_specific(self):
        for mode in ('cpu', 'cuda'):
            self.make_sdk(mode)
            with self.assertRaises(ValueError):
                self.select(mode, explicit=self.root / 'missing')
            self.env['LIBTORCH_' + mode.upper() + '_ROOT'] = str(self.root / 'missing')
            with self.assertRaises(ValueError):
                self.select(mode)
            self.assertEqual(self.select(mode, explicit=self.standard[mode])[0], self.standard[mode])

    def test_invalid_cuda_sdk_is_rejected(self):
        root = self.make_sdk()
        for version in ('2.13.0+cu126', '2.14.0+cpu', 'invalid'):
            (root / 'build-version').write_text(version, encoding='utf-8')
            with self.assertRaises(ValueError):
                self.select()
        (root / 'build-version').write_text('2.14.0+cu126', encoding='utf-8')
        (root / 'lib/c10_cuda.dll').unlink()
        with self.assertRaises(ValueError):
            self.select()

    def test_cpu_override_cannot_use_cuda_archive(self):
        root = self.make_sdk()
        with self.assertRaisesRegex(ValueError, 'CPU builds require CPU LibTorch'):
            self.select('cpu', root)

    def test_build_selection_requires_cpu_or_cuda(self):
        for mode in ('cpu', 'cuda'):
            self.device.write_bytes((' \r\n' + mode + '\r\n').encode())
            self.assertEqual(build_bob.read_device(self.project), mode)
        for mode in ('', 'auto', 'gpu', 'auto cpu'):
            self.device.write_text(mode, encoding='utf-8')
            with self.assertRaises(ValueError):
                build_bob.read_device(self.project)
        self.device.unlink()
        with self.assertRaises(ValueError):
            build_bob.read_device(self.project)

    def test_stages_only_selected_build_inputs(self):
        for mode in ('cpu', 'cuda'):
            source = self.make_sdk(mode)
            libraries = build_bob.CUDA_LIBRARIES if mode == 'cuda' else build_bob.CPU_LIBRARIES
            with build_bob.staged_sdk(self.project, mode, self.select(mode)) as stage:
                self.assertEqual({p.stem for p in (stage / 'lib').iterdir()}, set(libraries))
                self.assertEqual(list(stage.rglob('*.dll')), [])
                self.assertEqual((stage / build_bob.VERSION_HEADER).read_bytes(),
                                 (source / build_bob.VERSION_HEADER).read_bytes())
                manifest = (stage / 'manifest.json').read_text(encoding='utf-8')
                self.assertNotIn(str(self.root), manifest)
                self.assertEqual(json.loads(manifest)['device'], mode)
            self.assertFalse(stage.exists())
            self.assertTrue((source / 'lib/torch.dll').exists())

    def fingerprint(self, mode='cuda'):
        with build_bob.staged_sdk(self.project, mode, self.select(mode)) as stage:
            return (stage / 'manifest.json').read_bytes()

    def test_cache_tracks_headers_and_libraries_but_not_dlls(self):
        root = self.make_sdk()
        first = self.fingerprint()
        self.assertEqual(first, self.fingerprint())
        (root / 'lib/torch_cuda.dll').write_bytes(b'new DLL')
        self.assertEqual(first, self.fingerprint())
        (root / 'lib/torch_cuda.lib').write_bytes(b'new import library')
        second = self.fingerprint()
        self.assertNotEqual(first, second)
        (root / 'include/extra.h').write_text('// extra', encoding='utf-8')
        self.assertNotEqual(second, self.fingerprint())
        self.make_sdk('cpu')
        self.assertNotEqual(first, self.fingerprint('cpu'))

    def test_cleanup_after_failure_and_concurrent_build_rejected(self):
        self.make_sdk()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            with build_bob.staged_sdk(self.project, 'cuda', self.select()) as stage:
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    with build_bob.staged_sdk(self.project, 'cuda', self.select()):
                        self.fail('concurrent helper must fail')
                raise RuntimeError('failed')
        self.assertFalse(stage.exists())

    def test_cli_uses_selected_sdk_and_preserves_bob_exit_code(self):
        for mode in ('cpu', 'cuda'):
            self.make_sdk(mode)
            self.device.write_text(mode)
            stage = self.project / 'build-support/libtorch/local'

            def bob(command, **kwargs):
                self.assertEqual(command, ['bob.exe', 'build', 'bob.toml', '--out-dir', 'output with spaces'])
                self.assertEqual(kwargs['cwd'], self.project)
                self.assertEqual((stage / 'lib/torch_cuda.lib').exists(), mode == 'cuda')
                return subprocess.CompletedProcess(command, 7)

            with patch.object(build_bob, '__file__', str(self.project / 'scripts/build_bob.py')), \
                 patch.dict(os.environ, self.env, clear=True), \
                 patch.object(build_bob.shutil, 'which', return_value='bob.exe'), \
                 patch.object(build_bob.subprocess, 'run', side_effect=bob), redirect_stdout(io.StringIO()):
                self.assertEqual(build_bob.main(['--out-dir', 'output with spaces']), 7)
            self.assertFalse(stage.exists())

    def test_missing_sdk_stops_before_bob(self):
        self.device.write_text('cuda')
        with patch.object(build_bob, '__file__', str(self.project / 'scripts/build_bob.py')), \
             patch.dict(os.environ, self.env, clear=True), \
             patch.object(build_bob.subprocess, 'run') as run, redirect_stderr(io.StringIO()):
            self.assertEqual(build_bob.main([]), 1)
            run.assert_not_called()

    @unittest.skipUnless(shutil.which('cmake'), 'CMake needed for Docker selection')
    def test_cmake_and_helper_accept_the_same_build_selection(self):
        output = self.root / 'selection.txt'
        for mode in ('cpu', 'cuda', 'auto', 'invalid', ''):
            self.device.write_text(mode + '\n')
            output.unlink(missing_ok=True)
            result = subprocess.run(['cmake', f'-DDEVICE_FILE={self.device}', f'-DOUTPUT_FILE={output}',
                                     '-P', str(PROJECT / 'cmake/ReadDevice.cmake')],
                                    capture_output=True, text=True, timeout=15)
            if mode in ('cpu', 'cuda'):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output.read_text().strip(), mode)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
