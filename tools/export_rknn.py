#!/usr/bin/env python3
"""Traceable static YOLO RKNN export. Missing calibration never becomes FP16.

Conversion is not runtime validation, operator placement proof, or mAP evidence.
The original ONNX and existing RKNN files are kept by default.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import time

try:
    from rknn.api import RKNN
    HAS_RKNN_TOOLKIT = True
except ImportError:
    RKNN = None
    HAS_RKNN_TOOLKIT = False

class ExportError(ValueError):
    pass

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def parse_op_distribution(log_text):
    """Legacy textual hints only; absence of CPU messages is not placement proof."""
    cpu = [line.strip() for line in log_text.splitlines()
           if 'target: cpu' in line.lower() or 'run on cpu' in line.lower()]
    npu = [line.strip() for line in log_text.splitlines() if 'target: npu' in line.lower()]
    return {'npu_op_count': len(npu), 'cpu_fallback_count': len(cpu),
            'npu_passthrough_rate': None, 'cpu_fallback_ops': cpu[:20],
            'evidence_kind': 'unverified_textual_hints',
            'note': 'Actual operator placement requires a separate runtime profile; no 100% claim.'}

def inspect_onnx(path):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    if any(value.data_location == onnx.TensorProto.EXTERNAL for value in model.graph.initializer):
        raise ExportError('External tensor files are not supported by this bounded exporter')
    onnx.checker.check_model(model)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    if len(inputs) != 1:
        raise ExportError('Expected one image input')
    value = inputs[0]
    shape = [dim.dim_value if dim.HasField('dim_value') else None
             for dim in value.type.tensor_type.shape.dim]
    if (value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or len(shape) != 4
            or shape[0:2] != [1, 3] or any(type(x) is not int or x <= 0 for x in shape)):
        raise ExportError('Expected static float32 NCHW batch1 RGB ONNX input')
    if any(node.op_type in ('If', 'Loop', 'Scan') for node in model.graph.node):
        raise ExportError('Dynamic control flow must be explicitly reconstructed before conversion')
    return {'name': value.name, 'dtype': 'float32', 'layout': 'NCHW', 'shape': shape,
            'output_names': [value.name for value in model.graph.output],
            'opset': {value.domain or 'ai.onnx': value.version for value in model.opset_import}}

def calibration_entries(dataset_path):
    path = Path(dataset_path).resolve()
    if not path.is_file():
        raise ExportError('INT8 calibration list does not exist; FP16 fallback is forbidden')
    entries = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            # Existing project lists use project-relative paths. Prefer explicit
            # list-relative files, then the caller's project cwd; record resolution.
            relative = path.parent / candidate
            candidate = relative if relative.is_file() else Path.cwd() / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise ExportError(f'Calibration image missing: {line}')
        entries.append({'path': str(candidate), 'sha256': sha256(candidate)})
    if not entries:
        raise ExportError('INT8 calibration list is empty')
    return entries

def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')

def _check_result(result, operation):
    # Some Toolkit Python wrappers return None on successful config.
    if result is not None and result != 0:
        raise ExportError(f'{operation} failed with return code {result}')

def convert(onnx_path, output_path, dataset_txt, target_platform='rk3588',
            do_quant=True, algo='normal', *, overwrite=False, optimization_level=3,
            geometry='resize'):
    """Publish artifact and recipe only after successful SDK export; no mock path."""
    result = {'schema_version': 2, 'success': False, 'status': 'failed',
              'quantized': do_quant, 'algorithm': algo if do_quant else None,
              'runtime_validation': 'not_run', 'accuracy_validation': 'not_run',
              'operator_placement': 'not_measured'}
    runtime = None
    try:
        if type(do_quant) is not bool or type(overwrite) is not bool:
            raise ExportError('do_quant and overwrite must be bool')
        if type(optimization_level) is not int or not 0 <= optimization_level <= 3:
            raise ExportError('optimization_level must be 0..3')
        if geometry not in ('resize', 'letterbox'):
            raise ExportError('Unsupported preprocessing geometry')
        if target_platform != 'rk3588':
            raise ExportError('This recipe is pinned to rk3588')
        if algo not in ('normal', 'mmse', 'kl_divergence'):
            raise ExportError('Unknown calibration algorithm; no silent substitution')
        source, output = Path(onnx_path).resolve(), Path(output_path).resolve()
        metadata = output.with_suffix(output.suffix + '.json')
        if not source.is_file():
            raise ExportError('ONNX source is missing')
        if source in (output, metadata):
            raise ExportError('Cannot overwrite an input source')
        if not overwrite and (output.exists() or metadata.exists()):
            raise ExportError('Output exists; choose a new version directory or --overwrite')
        entries = calibration_entries(dataset_txt) if do_quant else []
        if do_quant and Path(dataset_txt).resolve() in (output, metadata):
            raise ExportError('Cannot overwrite calibration list')
        contract = inspect_onnx(source)
        source_sha = sha256(source)
        if not HAS_RKNN_TOOLKIT or RKNN is None:
            raise ExportError('rknn-toolkit2 is unavailable; conversion was not executed')
        try:
            version = importlib.metadata.version('rknn-toolkit2')
        except importlib.metadata.PackageNotFoundError:
            version = 'unverified'
        config = {'target_platform': target_platform, 'mean_values': [[0, 0, 0]],
                  'std_values': [[255, 255, 255]], 'float_dtype': 'float16',
                  'optimization_level': optimization_level}
        if do_quant:
            config.update(quantized_algorithm=algo, quantized_method='channel',
                          quantized_dtype='w8a8')
        recipe = {'source_onnx': str(source), 'source_sha256': source_sha,
                  'onnx_contract': contract, 'toolkit_version': version,
                  'sdk_config': config, 'do_quantization': do_quant,
                  'calibration': {'count': len(entries), 'entries': entries,
                                  'list_sha256': sha256(dataset_txt) if do_quant else None},
                  'external_input': {'dtype': 'uint8', 'layout': 'NHWC', 'color': 'RGB',
                                     'range': [0, 255], 'batch': 1, 'geometry': geometry,
                                     'normalization': 'Once inside RKNN using mean0/std255; never divide input twice'},
                  'limits': ['Export success is not finite-output/runtime/mAP acceptance',
                             'Operator placement is not inferred from missing CPU log strings',
                             'Calibration images are not an independent accuracy validation set']}
        result.update(onnx=str(source), output=str(output), recipe=recipe,
                      recipe_sha256=hashlib.sha256(_json_bytes(recipe)).hexdigest())
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.rknn-export-', dir=output.parent) as directory:
            staged = Path(directory)
            dataset = None
            if do_quant:
                dataset = staged / 'calibration.txt'
                dataset.write_text('\n'.join(entry['path'] for entry in entries) + '\n', encoding='utf-8')
            runtime = RKNN(verbose=True)
            _check_result(runtime.config(**config), 'config (algorithm unsupported is an error)')
            _check_result(runtime.load_onnx(model=str(source)), 'load_onnx')
            started = time.perf_counter()
            _check_result(runtime.build(do_quantization=do_quant,
                                        dataset=str(dataset) if dataset else None), 'build')
            result['build_seconds'] = time.perf_counter() - started
            artifact = staged / 'model.rknn'
            _check_result(runtime.export_rknn(str(artifact)), 'export_rknn')
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise ExportError('SDK returned success but produced no model bytes')
            if sha256(source) != source_sha:
                raise ExportError('ONNX changed while building; refusing publication')
            if do_quant and any(sha256(entry['path']) != entry['sha256'] for entry in entries):
                raise ExportError('Calibration image changed while building')
            result.update(success=True, status='exported_requires_runtime_validation',
                          rknn_sha256=sha256(artifact), size_bytes=artifact.stat().st_size,
                          size_mb=artifact.stat().st_size / (1024 * 1024), metadata_path=str(metadata))
            staged_metadata = staged / 'model.json'
            staged_metadata.write_bytes(_json_bytes(result))
            if not overwrite and (output.exists() or metadata.exists()):
                raise ExportError('Output appeared before publication; refusing overwrite')
            # The sidecar is the final completion marker. Old inputs are never touched.
            os.replace(artifact, output)
            os.replace(staged_metadata, metadata)
        return result
    except Exception as error:
        result.update(success=False, status='failed', error=f'{type(error).__name__}: {error}')
        return result
    finally:
        if runtime is not None:
            runtime.release()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', default='models/weights/yolov8n.onnx')
    parser.add_argument('--output', default='')
    parser.add_argument('--dataset', default='data/calibration/dataset.txt')
    parser.add_argument('--target-platform', default='rk3588')
    parser.add_argument('--algorithm', default='normal', choices=['normal', 'mmse', 'kl_divergence'])
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--optimization-level', type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument('--geometry', choices=['resize', 'letterbox'], default='resize')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--report', default='')
    args = parser.parse_args(argv)
    output = args.output or str(Path(args.onnx).with_suffix('')) + ('_fp16.rknn' if args.fp16 else f'_int8_{args.algorithm}.rknn')
    if args.report:
        destination = Path(args.report).resolve()
        forbidden = {Path(args.onnx).resolve(), Path(output).resolve(), Path(output).resolve().with_suffix('.rknn.json')}
        if destination in forbidden or (not args.fp16 and destination == Path(args.dataset).resolve()):
            parser.error('Report cannot overwrite an input or model/recipe')
        if destination.exists() and not args.overwrite:
            parser.error('Report exists; choose a new path')
    result = convert(args.onnx, output, args.dataset, args.target_platform,
                     not args.fp16, args.algorithm, overwrite=args.overwrite,
                     optimization_level=args.optimization_level, geometry=args.geometry)
    if args.report:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.rknn-report-', dir=destination.parent) as temporary:
            staged = Path(temporary) / 'report.json'
            staged.write_bytes(_json_bytes(result))
            os.replace(staged, destination)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result['success'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
