"""E8 batch=1, FP32, LR=160x160 cost; Depth estimation measured separately.

MAC/FLOP counts are a documented arithmetic model, not a kernel instruction
count. In particular fused selective scan is explicitly counted, never zero.
Latency includes existing diagnostic reductions; profiler hooks are removed
before latency/memory measurement.
"""
import argparse
import importlib
from pathlib import Path
import statistics as st
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.udr.e8_common import ROOT, load_yaml, record, validate_plan, write
from scripts.udr.run_e8_training import verify_completion


def operation_count(model, inputs):
    dense, scan = {}, [0, 0]
    handles = []
    def add(name, value):
        dense[name] = dense.get(name, 0) + int(value)
    def hook(module, args, output):
        kind = type(module).__name__
        if isinstance(module, torch.nn.Conv2d):
            add('conv2d', output.numel() * module.in_channels // module.groups * np.prod(module.kernel_size))
        elif isinstance(module, torch.nn.Linear):
            add('linear', output.numel() * module.in_features)
        elif kind == 'WindowAttention':
            b, n, c3 = args[0].shape
            add('attention_qk_av', 2 * b * n * n * (c3 // 3))
        elif kind == 'ASSM':
            b, length, _ = args[0].shape
            token = args[2]
            k, rank = module.embeddingB.weight.shape
            state = token.weight.shape[1]
            add('route_embedding_prompt', k * rank * state + b * length * k * state)
        elif kind == 'Selective_Scan':
            b, length, channels = args[0].shape
            n, r = module.d_state, module.dt_rank
            add('scan_projection_einsums', b * length * channels * (2 * r + 2 * n))
            # Real recurrence: delta*A, delta*B*u, state update and C output;
            # count exponentials as one modeled scalar operation, plus D*u.
            scan[0] += b * length * channels * (4 * n + 1)
            scan[1] += b * length * channels * n
        elif kind == 'SpatialGradient':
            add('functional_sobel', output.numel() * 2 * 9)
    for module in model.modules():
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    macs = sum(dense.values()) + scan[0]
    return dict(modeled_macs=macs, modeled_flops=2 * macs + scan[1],
        dense_macs_by_component=dense, scan_recurrence_macs=scan[0],
        scan_exponential_ops=scan[1],
        convention='multiply-add=2 FLOPs; real scan recurrence included; exp=1 modeled op',
        excluded='bias, norm/activation/softmax/Gumbel/quantile, sort/index/memory and other elementwise ops; '
                 'these are not zero-cost and remain included in measured latency. Counts are arithmetic estimates.')


def timed(call, device, warmup, repeats):
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    with torch.no_grad():
        for _ in range(warmup):
            value = call()
            del value
        sync()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        values = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            value = call()
            sync()
            values.append((time.perf_counter() - start) * 1000)
            del value
    return dict(latency_ms_mean=st.mean(values), latency_ms_std=st.stdev(values),
        latency_ms_median=st.median(values), latency_ms_p95=float(np.quantile(values, .95)),
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        repeats=repeats, warmup=warmup)


def load_depth_adapter(factory, checkpoint, device):
    """Use the SAME upstream estimator that produced the experiment's Depth.

    Factory signature: build(checkpoint=<str>, device=<torch.device>).
    Object: model (nn.Module), preprocess(uint8 RGB HWC)->input tensor,
    predict(input)->float tensor [1,1,160,160], name (str).
    Adapter owns the estimator's original resize/normalization/postprocessing.
    """
    module, function = factory.split(':', 1)
    adapter = getattr(importlib.import_module(module), function)(checkpoint=checkpoint, device=device)
    if not isinstance(adapter.model, torch.nn.Module) or not adapter.name:
        raise ValueError('Depth adapter needs .model and .name.')
    adapter.model.to(device).float().eval()
    return adapter


def profile(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Formal E8 profiling needs the production CUDA selective-scan kernels.')
    plan = validate_plan(args.plan)
    seed = plan['seeds'][0] if args.seed is None else args.seed
    run = next(item for item in plan['runs'] if item['seed'] == seed)
    device = torch.device('cuda:0')
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = ROOT/f'results/E8_scaling/x{plan["scale"]}'
    report = dict(scale=plan['scale'], batch=1, lr_shape=[160, 160], dtype='FP32',
                  gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, tf32=False, models={})
    # Host preprocessing measures actual RGB conversion, full-image P2/P98
    # scalar depth normalization, tensor conversion and H2D; disk IO excluded.
    from basicsr.data.rgb_depth_paired_image_dataset import RGBDepthPairedImageDataset
    normalizer = RGBDepthPairedImageDataset.__new__(RGBDepthPairedImageDataset)
    ds = load_yaml(plan['configs'][f's{seed}_udrv2_test']['path'])['datasets']['test_1']
    normalizer.depth_percentile_low = ds['depth_percentile_low']
    normalizer.depth_percentile_high = ds['depth_percentile_high']
    normalizer.depth_normalize_eps = ds['depth_normalize_eps']
    rng = np.random.default_rng(seed)
    rgb_host = rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
    depth_host = rng.random((160, 160), dtype=np.float32)
    def preprocess(with_depth):
        rgb = torch.from_numpy(rgb_host.astype(np.float32).transpose(2, 0, 1).copy()).unsqueeze(0).to(device) / 255
        if not with_depth:
            return (rgb,)
        depth = normalizer._normalize_depth(depth_host)
        depth = torch.from_numpy(depth.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        return rgb, depth
    from basicsr.archs import build_network
    from scripts.udr.prepare_e4_integration import parameters
    for label in ('rgb', 'udrv2'):
        checkpoint = verify_completion(args.plan, run, label)
        config = load_yaml(plan['configs'][f's{seed}_{label}_test']['path'])
        model = build_network(config['network_g']).to(device).float().eval()
        model.load_state_dict(parameters(checkpoint['path']), strict=True)
        pre = timed(lambda: preprocess(label == 'udrv2'), device, args.warmup, args.repeats)
        inputs = preprocess(label == 'udrv2')
        count = operation_count(model, inputs)
        measured = timed(lambda: model(*inputs), device, args.warmup, args.repeats)
        report['models'][label] = dict(params=sum(p.numel() for p in model.parameters()),
            checkpoint=checkpoint, **count, **measured,
            preprocessing_latency_ms=pre['latency_ms_mean'],
            input='seeded synthetic fixture; direct single 160x160 forward; no tiling or disk IO')
        del model, inputs
        torch.cuda.empty_cache()
    if not args.depth_factory or not args.depth_checkpoint:
        raise ValueError('E8 requires --depth-factory and --depth-checkpoint to report the actual separate estimator cost.')
    adapter = load_depth_adapter(args.depth_factory, args.depth_checkpoint, device)
    pre = timed(lambda: adapter.preprocess(rgb_host), device, args.warmup, args.repeats)
    depth_input = adapter.preprocess(rgb_host)
    with torch.no_grad():
        prediction = adapter.predict(depth_input)
    if prediction.shape != (1, 1, 160, 160) or prediction.dtype != torch.float32 or not torch.isfinite(prediction).all():
        raise ValueError('Depth adapter must return finite FP32 aligned LR depth [1,1,160,160].')
    del prediction
    measured = timed(lambda: adapter.predict(depth_input), device, args.warmup, args.repeats)
    report['depth_estimator'] = dict(name=adapter.name,
        checkpoint=record(args.depth_checkpoint), factory=args.depth_factory,
        input_tensor_shape=list(depth_input.shape),
        params=sum(p.numel() for p in adapter.model.parameters()),
        preprocessing_latency_ms=pre['latency_ms_mean'], **measured,
        note='Estimator inference plus its adapter postprocessing; report separately, even when Depth is cached offline.')
    report['online_total_latency_ms'] = (report['models']['udrv2']['latency_ms_mean'] +
        report['models']['udrv2']['preprocessing_latency_ms'] +
        measured['latency_ms_mean'] + pre['latency_ms_mean'])
    report['online_total_note'] = 'Sum of separately measured serial stages, not a measured pipelined end-to-end latency.'
    write(output/'E8_complexity.json', report)
    lines = ['# E8 complexity', '', 'LR 160×160; batch 1; FP32; TF32 off. ' + report['gpu'], '',
        '| Model | Params | Modeled MACs | Modeled FLOPs | Latency ms | Peak allocated VRAM bytes | Preprocess ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label, row in report['models'].items():
        lines.append(f'| {label} | {row["params"]} | {row["modeled_macs"]} | {row["modeled_flops"]} | '
                     f'{row["latency_ms_mean"]:.3f} | {row["peak_allocated_vram_bytes"]} | '
                     f'{row["preprocessing_latency_ms"]:.3f} |')
    lines += ['', 'MAC/FLOP counting exclusions: ' + count['excluded'], '',
              'Depth estimator (separate): ' + str(report['depth_estimator']), '',
              'Serial sum latency ms: ' + str(report['online_total_latency_ms']),
              report['online_total_note'], '']
    (output/'E8_complexity.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved E8 measured complexity:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--depth-factory', help='module:function adapter for the original Depth estimator')
    parser.add_argument('--depth-checkpoint')
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error('Use warmup>=1 and repeats>=2.')
    profile(args)


if __name__ == '__main__':
    main()
