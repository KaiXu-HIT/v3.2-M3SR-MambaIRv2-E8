"""Evaluate public/faithfully reproduced SR weights under E8's exact protocol.

Official models outside this repository use a small local factory adapter;
there is no guessed architecture, automatic weight conversion, or paper-number
mixing. DIV2K and DF2K rows are explicitly separated, never jointly ranked.
"""
import argparse
import importlib
import inspect
from pathlib import Path
import statistics as st
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.udr.e8_common import ROOT, DATASETS, load_yaml, read, record, write

MODELS = ('MambaIR', 'MambaIRv2', 'SwinIR', 'HAT', 'SRFormer', 'RCAN', 'EDSR')


def basicsr_factory(checkpoint, config, device):
    """Default adapter for architectures actually registered in this checkout.

    Other official repositories supply a factory with this same signature.
    Return a float RGB [0,1] -> RGB [0,1] nn.Module; its .upscale is required.
    Factory must load all requested weights strictly and call eval().
    """
    from basicsr.archs import build_network
    from scripts.udr.prepare_e4_integration import parameters
    model = build_network(load_yaml(config)['network_g']).to(device).float().eval()
    model.load_state_dict(parameters(checkpoint), strict=True)
    return model


def template():
    return dict(note='Fill source URL, training data, checkpoint, original architecture config and local factory. '
                     'enable only measured public-weight/unified-reproduction entries; all seven are needed for full comparison.',
        models=[dict(name=name, enabled=False, training_data=None, scale=4,
                     source_url=None, checkpoint=None, config=None,
                     factory='scripts.udr.compare_e8_public:basicsr_factory',
                     method='public_weights', training_recipe=None) for name in MODELS])


def validate_entry(entry, scale):
    for field in ('name', 'training_data', 'checkpoint', 'config', 'factory', 'source_url'):
        if not entry.get(field):
            raise ValueError('Missing public comparison field: ' + field)
    if entry['training_data'] not in ('DIV2K', 'DF2K') or entry['scale'] != scale:
        raise ValueError('Public comparison needs an explicit DIV2K/DF2K label and matching scale.')
    if not entry['source_url'].startswith('https://'):
        raise ValueError('Use a verifiable original public weight/reproduction source URL.')
    if entry['method'] not in ('public_weights', 'unified_reproduction'):
        raise ValueError('Unsupported comparison provenance.')
    if entry['method'] == 'unified_reproduction' and not entry.get('training_recipe'):
        raise ValueError('Unified reproduction must record its complete training recipe.')


def worker(entry_path, test_config, seeds, output):
    import torch
    from basicsr.data import build_dataset, build_dataloader
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.utils import set_random_seed
    from basicsr.utils.img_util import tensor2img
    from scripts.udr.e0_mechanism_audit import infer_partitioned
    entry = read(entry_path)
    config = load_yaml(test_config)
    validate_entry(entry, config['scale'])
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    module, name = entry['factory'].split(':', 1)
    factory = getattr(importlib.import_module(module), name)
    model = factory(checkpoint=entry['checkpoint'], config=entry['config'], device=device).float().eval()
    if model.upscale != config['scale']:
        raise ValueError('Official model .upscale differs from E8 test scale.')
    runs = {str(seed): {} for seed in seeds}
    with torch.no_grad():
        for _, source in sorted(config['datasets'].items()):
            source.update(phase='test', scale=config['scale'])
            dataset = build_dataset(source)
            dataset.paths.sort(key=lambda item: item['lq_path'])
            loader = build_dataloader(dataset, source, num_gpu=1, dist=False,
                                      sampler=None, seed=seeds[0])
            for seed in seeds:
                set_random_seed(seed)
                psnr, ssim = [], []
                # Match E7/E8 DataLoader iterator creation after RNG reset;
                # iterating Dataset directly would shift stochastic routing.
                for item in loader:
                    sr, _, _ = infer_partitioned(model, item['lq'].to(device))
                    if sr.shape != item['gt'].shape:
                        raise ValueError('Public model output shape differs from GT.')
                    image, gt = tensor2img(sr), tensor2img(item['gt'])
                    kwargs = dict(crop_border=config['scale'], test_y_channel=True)
                    psnr.append(float(calculate_psnr(image, gt, **kwargs)))
                    ssim.append(float(calculate_ssim(image, gt, **kwargs)))
                runs[str(seed)][source['name']] = dict(psnr=st.mean(psnr), ssim=st.mean(ssim))
    write(output, dict(runs=runs, checkpoint=record(entry['checkpoint']),
                      config=record(entry['config']), adapter=record(inspect.getfile(factory))))


def compare(args):
    metrics = read(args.metrics)
    plan = read(metrics['plan']['path'])
    if record(metrics['plan']['path']) != metrics['plan']:
        raise ValueError('E8 metric plan changed.')
    selected = [entry for entry in read(args.manifest)['models'] if entry.get('enabled')]
    if not selected:
        raise ValueError('Enable supplied public checkpoints in the comparison manifest first.')
    names = [entry['name'] for entry in selected]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate public model name; use distinct variants.')
    output = ROOT/f'results/E8_scaling/x{metrics["scale"]}/public_comparison'
    output.mkdir(parents=True, exist_ok=True)
    test = plan['configs'][f's{metrics["seeds"][0]}_rgb_test']['path']
    results = []
    for index, entry in enumerate(selected):
        validate_entry(entry, metrics['scale'])
        entrypath, result = output/f'model{index}.json', output/f'model{index}_measured.json'
        write(entrypath, entry)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                        str(entrypath), test, str(result), '--seeds', *map(str, metrics['seeds'])], cwd=ROOT, check=True)
        values = read(result)
        per = {ds: {metric: dict(mean=st.mean(values['runs'][str(seed)][ds][metric] for seed in metrics['seeds']),
                                 std=st.stdev(values['runs'][str(seed)][ds][metric] for seed in metrics['seeds']))
                    for metric in ('psnr', 'ssim')} for ds in DATASETS}
        results.append(dict(model=entry['name'], training_data=entry['training_data'],
            source_url=entry['source_url'], method=entry['method'], training_recipe=entry.get('training_recipe'),
            per_dataset=per, provenance=values,
            variability='Repeated inference of one public checkpoint; not independent training-seed variation.'))
    report = dict(scale=metrics['scale'], seeds=metrics['seeds'], models=results,
        e8_metrics=record(args.metrics), manifest=record(args.manifest),
        required_models=list(MODELS), missing_models=[name for name in MODELS if name not in names],
        complete_comparison=all(name in names for name in MODELS),
        note='DIV2K and DF2K are separate groups; compare E8 DF2K only against DF2K peers. '
             'Public-checkpoint inference std and E8 independent-training std have different meanings.')
    write(output/'E8_public_comparison.json', report)
    lines = ['# E8 public-model comparison', '', report['note'], '',
             'Complete seven-model comparison: ' + str(report['complete_comparison'])]
    for data in ('DIV2K', 'DF2K'):
        lines += ['', '## ' + data, '', '| Model | ' + ' | '.join(DATASETS) + ' | Source |',
                  '|---|' + '|'.join(['---:'] * len(DATASETS)) + '|---|']
        if data == 'DF2K':
            for label in ('rgb', 'udrv2'):
                values = [metrics['per_dataset'][ds][label] for ds in DATASETS]
                lines.append('| E8 ' + label + ' | ' + ' | '.join(
                    f'{v["psnr"]["mean"]:.4f}/{v["ssim"]["mean"]:.4f}' for v in values) + ' | This run |')
        for item in results:
            if item['training_data'] == data:
                lines.append('| ' + item['model'] + ' | ' + ' | '.join(
                    f'{item["per_dataset"][ds]["psnr"]["mean"]:.4f}/'
                    f'{item["per_dataset"][ds]["ssim"]["mean"]:.4f}' for ds in DATASETS) +
                    ' | ' + item['source_url'] + ' |')
    lines += ['', 'Missing: ' + ', '.join(report['missing_models']), '']
    (output/'E8_public_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved measured public comparison:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-template')
    parser.add_argument('--manifest', default=str(ROOT/'options/E8_public_models.json'))
    parser.add_argument('--metrics', default=str(ROOT/'results/E8_scaling/x4/E8_metrics.json'))
    parser.add_argument('--worker', nargs=3, metavar=('ENTRY', 'TEST_CONFIG', 'OUTPUT'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 11, 12])
    args = parser.parse_args()
    if args.write_template:
        if Path(args.write_template).exists():
            raise FileExistsError(args.write_template)
        write(args.write_template, template())
    elif args.worker:
        worker(args.worker[0], args.worker[1], args.seeds, args.worker[2])
    else:
        compare(args)


if __name__ == '__main__':
    main()
