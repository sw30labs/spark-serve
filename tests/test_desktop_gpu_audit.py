"""Offline identity tests; never inspect or terminate actual system processes."""
import inspect
import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spark_serve_controller import Controller, ControllerError, classify_gpu_processes
from test_controller import FakeController, config

EXE = '/usr/libexec/gnome-remote-desktop-daemon'
CGROUP = '0::/user.slice/user-1000.slice/user@1000.service/app.slice/gnome-remote-desktop-handover.service\n'


class DesktopAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        proc = self.root / '123'
        proc.mkdir()
        (proc / 'exe').symlink_to(EXE)
        (proc / 'cgroup').write_text(CGROUP)

    def classify(self, line=None):
        return classify_gpu_processes(line or f'123, {EXE}, 340', self.root)

    def test_verified_small_desktop_stays_visible_without_blocking(self):
        report = self.classify()
        self.assertEqual(1, len(report['gpu_processes']))
        self.assertEqual('verified_desktop_context', report['gpu_processes'][0]['classification'])
        self.assertEqual(EXE, report['desktop_gpu_contexts'][0]['executable'])
        self.assertEqual(340, report['desktop_gpu_contexts'][0]['used_gpu_memory_mib'])
        self.assertEqual([], report['blocking_gpu_processes'])

    def test_reported_name_alone_cannot_spoof_real_executable(self):
        (self.root / '123/exe').unlink()
        (self.root / '123/exe').symlink_to('/tmp/gnome-remote-desktop-daemon')
        self.assertTrue(self.classify()['blocking_gpu_processes'])
        self.assertEqual([], self.classify()['desktop_gpu_contexts'])

    def test_wrong_systemd_service_and_suffix_collision_still_block(self):
        for group in (CGROUP.replace('handover', 'fake'),
                      CGROUP.replace('user@1000', 'user@2000'),
                      '0::/evil/gnome-remote-desktop-handover.service\n'):
            with self.subTest(group=group):
                (self.root / '123/cgroup').write_text(group)
                self.assertTrue(self.classify()['blocking_gpu_processes'])

    def test_large_unknown_or_unsupported_allocations_block(self):
        for memory in ('1024.01', '4096', 'N/A', '[Not Supported]', 'nan', 'inf', '-1'):
            with self.subTest(memory=memory):
                self.assertTrue(self.classify(f'123, {EXE}, {memory}')['blocking_gpu_processes'])
        self.assertFalse(self.classify(f'123, {EXE}, 1024')['blocking_gpu_processes'])

    def test_unknown_compute_process_malformed_rows_and_disappearing_proc_block(self):
        for row in ('123, /usr/bin/python3, 200', 'bad row', f'9999, {EXE}, 0', f'0, {EXE}, 0'):
            with self.subTest(row=row):
                self.assertTrue(self.classify(row)['blocking_gpu_processes'])

    def test_proc_permission_failures_are_blocking(self):
        with patch('os.readlink', side_effect=PermissionError('denied')):
            self.assertTrue(self.classify()['blocking_gpu_processes'])
        with patch.object(Path, 'read_text', side_effect=PermissionError('denied')):
            self.assertTrue(self.classify()['blocking_gpu_processes'])

    def test_mixed_desktop_and_inference_is_still_blocked(self):
        result = self.classify(f'123, {EXE}, 340\n456, /usr/bin/python3, 1024')
        self.assertEqual(2, len(result['gpu_processes']))
        self.assertEqual(1, len(result['desktop_gpu_contexts']))
        self.assertEqual(1, len(result['blocking_gpu_processes']))

    def test_empty_gpu_output_stays_empty(self):
        self.assertEqual([], classify_gpu_processes('')['blocking_gpu_processes'])
        self.assertEqual([], classify_gpu_processes('No running processes found')['blocking_gpu_processes'])

    def test_controller_keeps_gpu_containers_blocking_even_with_verified_desktop(self):
        controller = FakeController(self.root / 'controller')
        classified = self.classify()
        controller.audit = lambda host: {'containers': [], 'listening_ports': [], **classified}
        controller.verify_idle()
        controller.audit = lambda host: {'containers': [{'name':'unknown','running':True,'gpu':True}], 'listening_ports': [], **classified}
        with self.assertRaisesRegex(ControllerError, 'GPU still in use'):
            controller.verify_idle()
        controller.audit = lambda host: {'containers': [], 'listening_ports': [], **self.classify('456, /usr/bin/python3, 100')}
        with self.assertRaisesRegex(ControllerError, 'GPU still in use'):
            controller.verify_idle()

    def test_embedded_remote_audit_uses_exact_same_classifier(self):
        captured = []
        def remote(cfg, host, script, check=False):
            captured.append(script)
            return subprocess.CompletedProcess([], 0, json.dumps({'gpu_processes':[]}), '')
        controller = Controller(config(), remote, directory=self.root / 'controller')
        controller.audit('head')
        argv = shlex.split(captured[0].split('\n', 1)[1])
        source = argv[2]
        self.assertIn(inspect.getsource(classify_gpu_processes), source)
        self.assertIn('--query-compute-apps=pid,process_name,used_gpu_memory', source)
        compile(source, '<remote-audit>', 'exec')
        namespace = {}
        exec(source[:source.index('import json, socket, subprocess, sys')], namespace)
        self.assertEqual(self.classify(), namespace['classify_gpu_processes'](f'123, {EXE}, 340', self.root))
