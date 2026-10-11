#!/usr/bin/env python3
"""UNITTEST fake SDK artifacts, no model inference, device or calibration claim."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import export_rknn as export

class FakeSDK:
    instances = []
    def __init__(self, **kwargs):
        self.calls = []
        self.released = False
        self.instances.append(self)
    def config(self, **kwargs):
        self.calls.append(('config', kwargs)); return 0
    def load_onnx(self, **kwargs):
        self.calls.append(('load', kwargs)); return 0
    def build(self, **kwargs):
        self.calls.append(('build', kwargs)); return 0
    def export_rknn(self, path):
        Path(path).write_bytes(b'UNITTEST not a real RKNN model'); return 0
    def release(self):
        self.released = True

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rknn-export-UNITTEST-')
        self.root = Path(self.temp.name)
        self.onnx = self.root / 'source.onnx'
        self.onnx.write_bytes(b'UNITTEST fake source')
        self.image = self.root / 'image.jpg'
        self.image.write_bytes(b'UNITTEST fake image')
        self.dataset = self.root / 'dataset.txt'
        self.dataset.write_text('image.jpg\n', encoding='utf-8')
        self.output = self.root / 'new' / 'model.rknn'
        FakeSDK.instances.clear()
        self.patches = [patch.object(export, 'HAS_RKNN_TOOLKIT', True),
                        patch.object(export, 'RKNN', FakeSDK),
                        patch.object(export, 'inspect_onnx', return_value={'shape': [1,3,8,8], 'dtype': 'float32'})]
        for p in self.patches: p.start()
    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()
    def run_export(self, **kwargs):
        return export.convert(self.onnx, self.output, self.dataset, **kwargs)
    def test_int8_missing_dataset_fails_before_sdk(self):
        self.dataset.unlink()
        result = self.run_export()
        self.assertFalse(result['success'])
        self.assertTrue(result['quantized'])
        self.assertIn('fallback is forbidden', result['error'])
        self.assertFalse(FakeSDK.instances)
        self.assertFalse(self.output.exists())
    def test_empty_and_missing_image_calibration_refused(self):
        for text in ('# comment\n', 'absent.jpg\n'):
            self.dataset.write_text(text)
            self.assertFalse(self.run_export()['success'])
        self.assertFalse(FakeSDK.instances)
    def test_int8_recipe_and_completion_marker(self):
        result = self.run_export()
        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 'exported_requires_runtime_validation')
        self.assertEqual(result['runtime_validation'], 'not_run')
        self.assertEqual(result['operator_placement'], 'not_measured')
        saved = json.loads(self.output.with_suffix('.rknn.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['rknn_sha256'], export.sha256(self.output))
        self.assertEqual(saved['recipe']['source_sha256'], export.sha256(self.onnx))
        self.assertEqual(saved['recipe']['calibration']['count'], 1)
        sdk = FakeSDK.instances[0]
        self.assertEqual(sdk.calls[0][1]['std_values'], [[255,255,255]])
        self.assertTrue(sdk.calls[2][1]['do_quantization'])
        self.assertTrue(sdk.released)
    def test_fp16_has_explicit_recipe_without_calibration(self):
        self.dataset.unlink()
        result = self.run_export(do_quant=False, geometry='letterbox', optimization_level=0)
        self.assertTrue(result['success'])
        self.assertFalse(result['quantized'])
        self.assertEqual(result['recipe']['sdk_config']['float_dtype'], 'float16')
        self.assertEqual(result['recipe']['sdk_config']['optimization_level'], 0)
        self.assertEqual(result['recipe']['external_input']['geometry'], 'letterbox')
        self.assertFalse(FakeSDK.instances[0].calls[2][1]['do_quantization'])
    def test_existing_outputs_are_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b'old model')
        result = self.run_export()
        self.assertFalse(result['success'])
        self.assertEqual(self.output.read_bytes(), b'old model')
        self.assertFalse(FakeSDK.instances)
    def test_source_cannot_be_output(self):
        result = export.convert(self.onnx, self.onnx, self.dataset)
        self.assertFalse(result['success'])
        self.assertEqual(self.onnx.read_bytes(), b'UNITTEST fake source')
    def test_invalid_parameters_are_not_guessed(self):
        for kwargs in ({'optimization_level': True}, {'optimization_level': 4}, {'algo':'made_up'},
                       {'target_platform':'other'}, {'do_quant':1}, {'geometry':'other'}):
            self.assertFalse(self.run_export(**kwargs)['success'])
        self.assertFalse(FakeSDK.instances)
    def test_build_failure_releases_and_does_not_publish(self):
        with patch.object(FakeSDK, 'build', return_value=-1):
            result = self.run_export()
        self.assertFalse(result['success'])
        self.assertTrue(FakeSDK.instances[0].released)
        self.assertFalse(self.output.exists())
    def test_algorithm_rejection_is_visible_without_substitution(self):
        with patch.object(FakeSDK, 'config', side_effect=ValueError('UNITTEST unsupported KL')):
            result = self.run_export(algo='kl_divergence')
        self.assertFalse(result['success'])
        self.assertIn('unsupported KL', result['error'])
        self.assertTrue(FakeSDK.instances[0].released)
    def test_no_bytes_after_successful_sdk_status_is_failure(self):
        with patch.object(FakeSDK, 'export_rknn', return_value=0):
            self.assertFalse(self.run_export()['success'])
        self.assertFalse(self.output.exists())
    def test_changed_source_blocks_publication(self):
        original = FakeSDK.export_rknn
        def changed(sdk, path):
            original(sdk, path); self.onnx.write_bytes(b'changed source'); return 0
        with patch.object(FakeSDK, 'export_rknn', changed):
            result = self.run_export()
        self.assertFalse(result['success'])
        self.assertFalse(self.output.exists())
    def test_log_hints_never_prove_100_percent(self):
        result = export.parse_op_distribution('target: npu\ntarget: cpu\n')
        self.assertIsNone(result['npu_passthrough_rate'])
        self.assertEqual(result['cpu_fallback_count'], 1)

if __name__ == '__main__': unittest.main(verbosity=2)
