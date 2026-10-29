"""Offline E8 contracts; CPU recurrence is not a production CUDA benchmark."""
import ast
from copy import deepcopy
import logging
from pathlib import Path
import sys
import tempfile
import types

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.udr.e8_common import (ROOT, DATASETS, gate_reports, load_yaml,
                                   transfer_state, summarize_runs)
from scripts.udr.prepare_e8_scaling import data_template, dataset_options, make_configs
from scripts.udr.profile_e8_scaling import operation_count


def reject(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError('Invalid E8 input was accepted.')


def contracts():
    seeds = [10, 11, 12]
    e4 = dict(e4_criteria=dict(minimum_success=True))
    e5 = dict(complete_five_set=True, criteria=dict(minimum_ordering=True,
        correct_gt_shuffle_all_seeds=True, shuffle_near_tie_001db=False))
    e6 = dict(seeds=seeds, five_set=dict(udrv2=dict(evidence_vs_rgb=dict(level='strong'))),
        per_dataset={ds: dict(udrv2=dict(delta_psnr_vs_rgb=dict(mean=.06))) for ds in DATASETS})
    e7 = dict(seeds=seeds, bootstrap_replicates=10000,
        datasets={ds: dict(ci95_psnr=[.01, .08], mean_delta_psnr=.06) for ds in DATASETS})
    assert all(gate_reports(e4, e5, e6, e7).values())
    bad = deepcopy(e7)
    bad['datasets']['Urban100']['ci95_psnr'] = [-.01, .08]
    reject(lambda: gate_reports(e4, e5, e6, bad))
    bad = deepcopy(e5)
    bad['complete_five_set'] = False
    reject(lambda: gate_reports(e4, bad, e6, e7))
    for scale in (4, 2, 3):
        manifest = data_template()
        item = manifest['scales'][str(scale)]
        item['flickr2k'].update(dataroot_gt=['/flickr/hr'], dataroot_lq=[f'/flickr/x{scale}'],
                                dataroot_depth=[f'/flickr/depth/x{scale}'])
        if scale != 4:
            for index, data in enumerate([item['div2k'], item['validation']] + list(item['tests'].values())):
                data.update(dataroot_lq=[f'/new/{index}/x{scale}'], dataroot_depth=[f'/new/{index}/depth/x{scale}'])
        datasets, tests = dataset_options(manifest, scale)
        assert len(datasets['train']['dataroot_gt']) == 2
        assert datasets['train']['gt_size'] == 48 * scale
        rgb = load_yaml(ROOT/'options/train/mambairv2/train_S0_RGB_MambaIRv2_x4.yml')
        e4config = load_yaml(ROOT/'options/train/mambairv2/train_E3_a1e4_phaseB_x4.yml')
        e4config['model_type'] = 'E4UDRMambaIRv2Model'
        e4config['network_g'].update(type='E4UDRMambaIRv2', uncertainty_mode='learned_error')
        e4config['train'].update(lambda_u=.01, lambda_a=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            configs, runs = make_configs(Path(tmp), scale, seeds, rgb, e4config, datasets, tests)
            for run in runs:
                seed = run['seed']
                baseline = load_yaml(configs[f's{seed}_rgb']['path'])
                a = load_yaml(configs[f's{seed}_udr_a']['path'])
                b = load_yaml(configs[f's{seed}_udrv2']['path'])
                assert baseline['train']['total_iter'] == a['train']['total_iter'] + b['train']['total_iter'] == 480000
                assert a['train']['phase'] == 'A' and b['train']['phase'] == 'B'
                assert a['train']['rgb_teacher_checkpoint'] == b['train']['rgb_teacher_checkpoint'] == run['checkpoints']['rgb']
                assert b['train']['lambda_u'] == .01 and b['train']['lambda_a'] == 1e-4
                for field in ('dataroot_gt', 'dataroot_lq', 'filename_tmpl', 'gt_size'):
                    assert baseline['datasets']['train'][field] == b['datasets']['train'][field]
                assert 'dataroot_depth' not in baseline['datasets']['train']
                for label in ('rgb', 'udrv2'):
                    cfg = load_yaml(configs[f's{seed}_{label}_test']['path'])
                    assert cfg['scale'] == cfg['network_g']['upscale'] == scale
                    assert cfg['val']['metrics']['psnr']['crop_border'] == scale
                    assert cfg['val']['metrics']['ssim']['test_y_channel'] is True
        # Existing x4 GT/LR/depth settings cannot be replaced by the manifest.
        if scale == 4:
            manifest['scales']['4']['div2k']['dataroot_lq'] = ['/changed']
            reject(lambda: dataset_options(manifest, scale))
    values = {label: {seed: {ds: dict(psnr=30 + delta, ssim=.9 + delta / 100)
                             for ds in DATASETS} for seed in seeds}
              for label, delta in (('rgb', 0), ('udrv2', .06))}
    report = summarize_runs(values, seeds)
    assert report['criteria']['minimum_gain_005'] and not report['criteria']['stronger_gain_008']
    toy = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, bias=False), torch.nn.Flatten(),
                              torch.nn.Linear(8, 3, bias=False))
    arithmetic = operation_count(toy, (torch.ones(1, 1, 4, 4),))
    assert arithmetic['modeled_macs'] == 72 + 24
    assert arithmetic['modeled_flops'] == 192
    print('PASS: E4-E7 entry gates, DF2K path protection, matched budgets/teachers and all-scale metrics')


def architectures():
    from scripts.udr.check_e4_udrv2 import load_architectures
    base, e0, _, _, e4 = load_architectures()
    tree = ast.parse((ROOT/'basicsr/models/e2_udr_mambairv2_model.py').read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'normalized_rgb_error']
    env = dict(torch=torch, F=F)
    exec(compile(tree, '<teacher target>', 'exec'), env)
    target = env['normalized_rgb_error']
    kw = dict(img_size=8, embed_dim=12, depths=(1,) * 6, num_heads=(3,) * 6,
              window_size=4, d_state=2, inner_rank=4, num_tokens=4, mlp_ratio=1.,
              upsampler='pixelshuffle')
    rgb, depth = torch.rand(1, 3, 8, 8), torch.rand(1, 1, 8, 8)
    torch.manual_seed(10)
    parent_rgb = base.MambaIRv2(upscale=4, **kw)
    parent_udr = e4.E4UDRMambaIRv2(upscale=4, **kw)
    reject(lambda: e0.UDRMambaIRv2(upscale=2, **kw))
    for scale in (4, 2, 3):
        baseline = base.MambaIRv2(upscale=scale, **kw)
        rs, reset = transfer_state(baseline.state_dict(), parent_rgb.state_dict(), scale != 4)
        baseline.load_state_dict(rs, strict=True)
        for mode in ('route_concentration', 'feature_variance', 'learned_error'):
            parent = e4.E4UDRMambaIRv2(upscale=4, uncertainty_mode=mode, **kw)
            model = e4.E4UDRMambaIRv2(upscale=scale, performance_scaling=True, uncertainty_mode=mode, **kw)
            state, keys = transfer_state(model.state_dict(), parent.state_dict(), scale != 4, rs)
            model.load_state_dict(state, strict=True)
            assert keys == reset
            for key in keys:
                assert torch.equal(state[key], rs[key])
            for phase in ('A', 'B'):
                model.configure_phase(phase)
                out = model(rgb, depth)
                assert out.shape == (1, 3, 8 * scale, 8 * scale)
                loss = out.abs().mean()
                if mode == 'learned_error':
                    truth = target(out.detach(), torch.zeros_like(out), scale=scale)
                    assert truth.shape == model.uncertainty_map.shape == (1, 1, 8, 8)
                    loss = loss + F.l1_loss(model.uncertainty_map, truth)
                loss.backward()
                model.zero_grad(set_to_none=True)
            if mode == 'learned_error':
                count = operation_count(model.eval(), (rgb, depth))
                assert count['scan_recurrence_macs'] > 0
                assert count['dense_macs_by_component']['scan_projection_einsums'] > 0
        bad = dict(rs)
        bad['conv_first.weight'] = bad['conv_first.weight'][:1]
        reject(lambda: transfer_state(baseline.state_dict(), bad, scale != 4))
    print('PASS: x4/x2/x3 U1/U2/U3 forward/backward, strict transfer, shared fresh PixelShuffle and scan cost')
    optimizer_steps(base, e4, kw)


def optimizer_steps(base, e4, kw):
    from scripts.udr.check_e4_udrv2 import extract_e2_functions
    from scripts.udr.check_udr import load_model_class
    from scripts.grs.check_grs import load_isolated
    parent = load_model_class(types.SimpleNamespace(UDRMambaIRv2=e4.E4UDRMambaIRv2), True)
    class Registry:
        def register(self):
            return lambda cls: cls
    helpers = extract_e2_functions()
    mod = load_isolated((ROOT/'basicsr/models/e4_udrv2_model.py').read_text(encoding='utf-8'),
        dict(UDRMambaIRv2Model=parent, MambaIRv2=base.MambaIRv2, SRModel=parent.__mro__[1],
             MODEL_REGISTRY=Registry(), deepcopy=deepcopy,
             normalized_rgb_error=helpers['normalized_rgb_error'], pearson_map=helpers['pearson_map'],
             get_root_logger=lambda: logging.getLogger('e8_check')))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for scale in (2, 3):
            options = dict(kw, upscale=scale)
            teacher_path, initial = folder/f'rgb{scale}.pth', folder/f'udr{scale}.pth'
            torch.save(dict(params=base.MambaIRv2(**options).state_dict()), teacher_path)
            architecture = dict(type='E4UDRMambaIRv2', uncertainty_mode='learned_error',
                                performance_scaling=True, **options)
            net_options = dict(architecture)
            net_options.pop('type')
            torch.save(dict(params=e4.E4UDRMambaIRv2(**net_options).state_dict()), initial)
            config = dict(is_train=True, scale=scale, dist=False, num_gpu=0,
                network_g=architecture,
                path=dict(pretrain_network_g=str(initial), pretrain_kind='e4', strict_load_g=True),
                train=dict(phase='A', lambda_a=1e-4, lambda_u=.01,
                    rgb_teacher_checkpoint=str(teacher_path), optim_g=dict(type='Adam', lr=1e-4),
                    scheduler=dict(type='MultiStepLR', milestones=[], gamma=1.),
                    pixel_opt=dict(type='L1Loss', loss_weight=1., reduction='mean')))
            for phase in ('A', 'B'):
                config['train'].update(phase=phase, rgb_lr=1e-5)
                model = mod.E4UDRMambaIRv2Model(deepcopy(config))
                model.feed_data(dict(lq=torch.rand(1, 3, 4, 4), depth=torch.rand(1, 1, 4, 4),
                                     gt=torch.rand(1, 3, 4 * scale, 4 * scale)))
                model.optimize_parameters(1)
                net = model.get_bare_model(model.net_g)
                assert net.uncertainty_map.shape == (1, 1, 4, 4)
                assert net.udr.projection.weight.grad is not None
                assert (net.conv_first.weight.grad is None) == (phase == 'A')
                assert all(parameter.grad is None for parameter in model.rgb_teacher.parameters())
                torch.save(dict(params=net.state_dict()), initial)
    print('PASS: actual x2/x3 E4 A/B optimizer steps, frozen same-scale U3 teacher and loss shapes')


if __name__ == '__main__':
    torch.set_num_threads(1)
    contracts()
    architectures()
