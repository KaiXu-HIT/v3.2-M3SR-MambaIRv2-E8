"""E8 DF2K warm-start plans, sequential x4 -> x2 -> x3; no new mechanism."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.udr.e8_common import (ROOT, DATASETS, SCALES, gate_reports, load_yaml,
    read, record, rgb_dataset, scaled_network, write, verify)
import yaml


def data_template():
    train = load_yaml(ROOT/'options/train/mambairv2/train_E3_a1e4_phaseB_x4.yml')['datasets']
    test = load_yaml(ROOT/'options/test/mambairv2/test_UDR_MambaSR_x4.yml')['datasets']
    result = dict(note='Fill null paths only; x4 DIV2K/test paths are inherited. '
                       'All LR/depth files must use each dataset filename template. '
                       'Depth must be generated from that LR input, never HR.', scales={})
    for scale in SCALES:
        def paths(source):
            item = {key: deepcopy(value) for key, value in source.items()
                    if key.startswith('dataroot_') or key.startswith('filename_tmpl')}
            if scale != 4:
                item['dataroot_lq'] = None
                item['dataroot_depth'] = None
                for key in ('filename_tmpl', 'filename_tmpl_depth'):
                    item[key] = item[key].replace('x4', f'x{scale}')
            return item
        div = paths(train['train'])
        flickr = deepcopy(div)
        for key in ('dataroot_gt', 'dataroot_lq', 'dataroot_depth'):
            flickr[key] = None
        result['scales'][str(scale)] = dict(div2k=div, flickr2k=flickr,
            validation=paths(train['val']),
            tests={item['name']: paths(item) for item in test.values()})
    return result


def dataset_options(manifest, scale):
    sources = load_yaml(ROOT/'options/train/mambairv2/train_E3_a1e4_phaseB_x4.yml')['datasets']
    tests = load_yaml(ROOT/'options/test/mambairv2/test_UDR_MambaSR_x4.yml')['datasets']
    entry = manifest['scales'][str(scale)]
    for item in [entry['div2k'], entry['flickr2k'], entry['validation']] + list(entry['tests'].values()):
        for key in ('dataroot_gt', 'dataroot_lq', 'dataroot_depth'):
            value = item.get(key)
            if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str) or not value[0]:
                raise ValueError(f'E8 x{scale} requires one explicit folder in {key}; fill the data manifest.')
    if set(entry['tests']) != set(DATASETS):
        raise ValueError('E8 requires exactly the original five test datasets.')
    # x4 roots are immutable. New scales retain GT roots, while LR/depth
    # roots are explicit inputs rather than unverified string substitutions.
    old = data_template()['scales']['4']
    for part in ('div2k', 'validation'):
        for key in ('dataroot_gt', 'dataroot_lq', 'dataroot_depth'):
            if (scale == 4 or key == 'dataroot_gt') and entry[part][key] != old[part][key]:
                raise ValueError('E8 changed an existing DIV2K path: ' + part + '/' + key)
    for ds in DATASETS:
        for key in ('dataroot_gt', 'dataroot_lq', 'dataroot_depth'):
            if (scale == 4 or key == 'dataroot_gt') and entry['tests'][ds][key] != old['tests'][ds][key]:
                raise ValueError('E8 changed an existing benchmark path: ' + ds + '/' + key)
    training = deepcopy(sources['train'])
    training.update(entry['div2k'])
    training['name'] = 'DF2K-Train-RGBDepth'
    # Both collections are sampled uniformly by image by the existing
    # multi-root dataset. Do not duplicate DIV2K or use a different RGB set.
    for key in ('dataroot_gt', 'dataroot_lq', 'dataroot_depth'):
        if entry['div2k'][key] == entry['flickr2k'][key]:
            raise ValueError('Flickr2K must be a separate collection: ' + key)
        training[key] = entry['div2k'][key] + entry['flickr2k'][key]
    for key in ('filename_tmpl', 'filename_tmpl_depth'):
        if entry['div2k'][key] != entry['flickr2k'][key]:
            raise ValueError('Existing multi-root loader requires equal DIV2K/Flickr2K filename templates.')
    # Keep the LR training patch 48x48 for a comparable per-step budget.
    training['gt_size'] = 48 * scale
    validation = deepcopy(sources['val'])
    validation.update(entry['validation'])
    for item in tests.values():
        item.update(entry['tests'][item['name']])
    return dict(train=training, val=validation), tests


def validate_data(datasets, tests, scale):
    """Read every triple once to reject absent/misaligned/new HR-derived data."""
    from basicsr.data import build_dataset
    counts = {}
    for name, source in list(datasets.items()) + list(tests.items()):
        opt = deepcopy(source)
        opt.update(phase='val', scale=scale)
        dataset = build_dataset(opt)
        for index in range(len(dataset)):
            sample = dataset[index]
            h, w = sample['lq'].shape[-2:]
            if sample['gt'].shape[-2:] != (h * scale, w * scale):
                raise ValueError('GT/LR scale mismatch: ' + sample['gt_path'])
        counts[name] = len(dataset)
    expected_counts = dict(val=100, test_1=5, test_2=14, test_3=100, test_4=100, test_5=109)
    for key, expected in expected_counts.items():
        if counts[key] != expected:
            raise ValueError(f'E8 incomplete validation/benchmark set {key}: expected {expected}, got {counts[key]}.')
    # Audit train collections independently: a mislabeled DIV2K-only plan
    # must not silently claim DF2K. The official full counts are fixed here.
    for index, (label, expected) in enumerate((('DIV2K', 800), ('Flickr2K', 2650))):
        source = deepcopy(datasets['train'])
        for key in ('dataroot_gt', 'dataroot_lq', 'dataroot_depth'):
            source[key] = [source[key][index]]
        source.update(phase='val', scale=scale)
        count = len(build_dataset(source))
        if count != expected:
            raise ValueError(f'E8 full DF2K requires {expected} {label} images, got {count}.')
        counts[label] = count
    return counts


def make_configs(output, scale, seeds, rgb_source, e4_source, datasets, tests,
                 phase_a=30000, phase_b=450000):
    """Both models receive 480k additional DF2K updates by default.

    RGB teacher is the finished same-scale RGB control, immutable throughout
    UDR A/B. Each model retains its own validated RGB/UDR ancestor.
    """
    configs, runs = {}, []
    outdir = output/'configs'
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        prefix = f'E8_DF2K_x{scale}_s{seed}'
        paths = dict(rgb_initial=str(output/f'rgb_initial_s{seed}.pth'),
                     udr_initial=str(output/f'udr_initial_s{seed}.pth'))
        for key, steps in (('rgb', phase_a + phase_b), ('udr_a', phase_a), ('udrv2', phase_b)):
            paths[key] = str(ROOT/f'experiments/{prefix}_{key}/models/net_g_{steps}.pth')
        for label in ('rgb', 'udr_a', 'udrv2'):
            config = deepcopy(rgb_source if label == 'rgb' else e4_source)
            config.update(name=prefix + '_' + label, scale=scale, manual_seed=seed)
            config['datasets'] = deepcopy(datasets)
            if label == 'rgb':
                config['datasets'] = {key: rgb_dataset(value) for key, value in datasets.items()}
            config['network_g'] = scaled_network(config['network_g'], scale)
            config['path'].update(pretrain_network_g=paths['rgb_initial' if label == 'rgb' else
                ('udr_initial' if label == 'udr_a' else 'udr_a')], strict_load_g=True,
                resume_state=None, param_key_g='params')
            train = config['train']
            train['total_iter'] = phase_a + phase_b if label == 'rgb' else (phase_a if label == 'udr_a' else phase_b)
            # Preserve the original low RGB / higher adaptive-module rates;
            # hold them constant in warm-start adaptation for both controls.
            train['scheduler'] = dict(type='MultiStepLR', milestones=[], gamma=1.)
            train['warmup_iter'] = -1
            if label == 'rgb':
                train['optim_g']['lr'] = 1e-5
            else:
                config['path']['pretrain_kind'] = 'e4'
                train['phase'] = 'A' if label == 'udr_a' else 'B'
                train['rgb_teacher_checkpoint'] = paths['rgb']
            for opt in config['val']['metrics'].values():
                opt.update(crop_border=scale, test_y_channel=True)
            path = outdir/f's{seed}_{label}.yml'
            path.write_text('# E8: DF2K; fixed E4 mechanism; original x4 paths retained.\n' +
                            yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            configs[f's{seed}_{label}'] = record(path)
        for label, source in (('rgb', rgb_source), ('udrv2', e4_source)):
            cfg = dict(name=f'{prefix}_{label}_test', model_type=source['model_type'],
                       scale=scale, num_gpu=1, manual_seed=seed,
                       network_g=scaled_network(source['network_g'], scale),
                       path=dict(pretrain_network_g=paths[label], strict_load_g=True, param_key_g='params'),
                       datasets={key: rgb_dataset(value) if label == 'rgb' else deepcopy(value)
                                 for key, value in tests.items()},
                       val=dict(save_img=False, suffix=None, metrics=deepcopy(source['val']['metrics'])))
            for opt in cfg['val']['metrics'].values():
                opt.update(crop_border=scale, test_y_channel=True)
            path = outdir/f's{seed}_{label}_test.yml'
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            configs[f's{seed}_{label}_test'] = record(path)
        runs.append(dict(seed=seed, checkpoints=paths))
    return configs, runs


def prepare(args):
    e6 = read(args.e6_report)
    e7 = read(args.e7_report)
    checks = gate_reports(read(args.e4_report), read(args.e5_report), e6, e7)
    # Check the E6 lineage and E7 checkpoint hashes, rather than trusting a
    # pass boolean detached from the actual weights.
    from scripts.udr.evaluate_e6_multiseed import read_plan, checkpoint_provenance
    from scripts.udr.e7_image_statistics import validate_e6_report
    e6plan = read_plan(args.e6_plan)
    provenance = checkpoint_provenance(e6plan, ['rgb', 'udrv1', 'udrv2'])
    validate_e6_report(e6plan, e6, provenance)
    for label in ('rgb', 'udrv2'):
        for seed in e6plan['seeds']:
            if e7['checkpoint_sha256'][label][str(seed)] != provenance[label][seed]['sha256']:
                raise ValueError('E7 and E6 checkpoint lineage differs.')
    scale = args.scale
    inputs = {name: record(getattr(args, name)) for name in
              ('e4_report', 'e5_report', 'e6_report', 'e7_report', 'e6_plan')}
    source_runs = {s: {label: provenance[label][s] for label in ('rgb', 'udrv2')}
                   for s in e6plan['seeds']}
    if scale != 4:
        previous_scale = 4 if scale == 2 else 2
        previous = ROOT/f'results/E8_scaling/x{previous_scale}/E8_metrics.json'
        previous_report = read(previous)
        if previous_report['scale'] != previous_scale or previous_report['training_data'] != 'DF2K':
            raise ValueError('Finish and evaluate the preceding scale before preparing this scale.')
        verify(previous_report['plan'])
        if previous_report['seeds'] != e6plan['seeds']:
            raise ValueError('Previous E8 scale used a different seed list.')
        inputs['previous_scale_metrics'] = record(previous)
        # Always transfer x4 -> new scale; x3 waits until x2 is evaluated,
        # but does not inherit the different x2 reconstruction head.
        parent = read(ROOT/'results/E8_scaling/x4/E8_metrics.json')
        verify(parent['plan'])
        if parent['seeds'] != e6plan['seeds']:
            raise ValueError('E8 changed matched seeds between scales.')
        source_runs = {s: {label: parent['checkpoints'][label][str(s)] for label in ('rgb', 'udrv2')}
                       for s in e6plan['seeds']}
        inputs['x4_metrics'] = record(ROOT/'results/E8_scaling/x4/E8_metrics.json')
    manifest = read(args.data_manifest)
    datasets, tests = dataset_options(manifest, scale)
    counts = validate_data(datasets, tests, scale)
    from scripts.udr.prepare_e4_integration import MODES, TAGS
    u, a = e6plan['selected_uncertainty'], e6plan['selected_alpha']
    if e6['selected_uncertainty'] != u or e6['selected_alpha'] != a:
        raise ValueError('E6 results and plan selected different mechanisms.')
    ablation = next(row for row in read(args.e4_report)['ablation'] if row['model'] == 'U + A')
    if ablation['uncertainty'] != u or ablation['local_alpha'] != a:
        raise ValueError('E4 ablation does not describe the mechanism confirmed by E6.')
    e5 = read(args.e5_report)
    if e5['provenance']['e4_uncertainty_mode'] != MODES[u]:
        raise ValueError('E5 tested a different E4 uncertainty mechanism.')
    e4_source = load_yaml(ROOT/f'options/train/mambairv2/train_E3_{a}_phaseB_x4.yml')
    e4_source['model_type'] = 'E4UDRMambaIRv2Model'
    e4_source['network_g'].update(type='E4UDRMambaIRv2', uncertainty_mode=MODES[u])
    e4_source['train'].update(lambda_u=.01 if u == 'U3' else 0., lambda_a=TAGS[a])
    rgb_source = load_yaml(ROOT/'options/train/mambairv2/train_S0_RGB_MambaIRv2_x4.yml')
    output = ROOT/f'experiments/E8_scaling/x{scale}'
    if (output/'E8_plan.json').exists():
        raise FileExistsError('E8 plan already exists; retain it for reproducibility: ' + str(output))
    # Freeze only this scale's data input. Filling x2/x3 paths later must not
    # invalidate an already trained x4 plan or its cost measurement.
    snapshot = output/'E8_data_snapshot.json'
    write(snapshot, dict(scale=scale, data=manifest['scales'][str(scale)],
                         source_manifest=str(Path(args.data_manifest).resolve())))
    inputs['data_snapshot'] = record(snapshot)
    configs, runs = make_configs(output, scale, e6plan['seeds'], rgb_source, e4_source,
                                datasets, tests, args.phase_a, args.phase_b)
    for run in runs:
        run['sources'] = {label: record(source_runs[run['seed']][label]['path'])
                          for label in ('rgb', 'udrv2')}
        for label in ('rgb', 'udrv2'):
            if run['sources'][label]['sha256'] != source_runs[run['seed']][label]['sha256']:
                raise ValueError('E8 parent checkpoint changed.')
    plan = dict(schema='E8-v1', scale=scale, seeds=e6plan['seeds'], training_data='DF2K',
        data_counts=counts, inputs=inputs, entry_checks=checks, configs=configs, runs=runs,
        selected_uncertainty=u, selected_alpha=a, phase_a=args.phase_a, phase_b=args.phase_b,
        added_updates_per_model=args.phase_a + args.phase_b,
        parent_scale=4, reset_upsample=scale != 4,
        teacher='same-scale same-seed finished DF2K RGB control, frozen throughout UDR A/B',
        baseline_note='E8 DF2K control; E0-E7 fixed DIV2K reference is retained separately. '
                      'Inherited RGB/UDR training costs differ and must be disclosed.')
    write(output/'E8_plan.json', plan)
    print('Saved verified E8 plan:', output/'E8_plan.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-data-template', metavar='JSON')
    parser.add_argument('--data-manifest', default=str(ROOT/'options/E8_data.json'))
    parser.add_argument('--scale', type=int, choices=SCALES, default=4)
    parser.add_argument('--phase-a', type=int, default=30000)
    parser.add_argument('--phase-b', type=int, default=450000)
    defaults = dict(e4_report='results/E4_udrv2/ablation/E4_ablation.json',
        e5_report='results/E5_depth_causality/E5_causality.json',
        e6_report='results/E6_multiseed/E6_multiseed.json',
        e7_report='results/E7_image_statistics/E7_statistics.json',
        e6_plan='experiments/E6_plan/E6_plan.json')
    for key, default in defaults.items():
        parser.add_argument('--' + key.replace('_', '-'), default=str(ROOT/default))
    args = parser.parse_args()
    if args.write_data_template:
        if Path(args.write_data_template).exists():
            raise FileExistsError(args.write_data_template)
        write(args.write_data_template, data_template())
        return
    if not (30000 <= args.phase_a <= 50000 and 250000 <= args.phase_b <= 450000):
        parser.error('E8 warm-start requires A 30k-50k and B 250k-450k.')
    if args.phase_a % 5000 or args.phase_b % 5000:
        parser.error('Use 5000-step budget increments so the exact final checkpoints and optimizer states are saved.')
    prepare(args)


if __name__ == '__main__':
    main()
