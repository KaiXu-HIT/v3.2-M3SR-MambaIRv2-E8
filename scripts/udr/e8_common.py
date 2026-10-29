"""E8 contracts shared by planning, training, evaluation and profiling.

Only scaling/data/budget changes are introduced. E4's selected U/A mechanism,
loss coefficients and existing x4 dataset paths remain the source of truth.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
SCALES = (4, 2, 3)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def verify(recorded):
    if sha256(recorded['path']) != recorded['sha256']:
        raise ValueError('E8 source changed: ' + recorded['path'])


def gate_reports(e4, e5, e6, e7):
    """Conservative operational meaning of 'E4-E7 passed' (no override).

    The proposal gives no global E7 pass flag. Require positive intervals on
    both priority datasets and no negative mean/interval on another dataset;
    do not relabel crossing-zero intervals as statistically stable gains.
    """
    seeds = e6['seeds']
    if len(seeds) not in (3, 5) or len(set(seeds)) != len(seeds):
        raise ValueError('E8 needs the formal independent E6 seed experiment.')
    tests = dict(
        e4_minimum_success=e4['e4_criteria']['minimum_success'] is True,
        e5_complete=e5['complete_five_set'] is True,
        e5_causal_order=e5['criteria']['minimum_ordering'] is True,
        e5_shuffle_above_noise=(e5['criteria']['correct_gt_shuffle_all_seeds'] is True
                               and not e5['criteria']['shuffle_near_tie_001db']),
        e6_same_e7_seeds=e7['seeds'] == seeds,
        e6_strong=e6['five_set']['udrv2']['evidence_vs_rgb']['level'] == 'strong',
        e6_urban_nonnegative=e6['per_dataset']['Urban100']['udrv2']['delta_psnr_vs_rgb']['mean'] >= 0,
        e6_manga_positive=e6['per_dataset']['Manga109']['udrv2']['delta_psnr_vs_rgb']['mean'] > 0,
        e7_10000=e7['bootstrap_replicates'] == 10000,
        e7_complete=set(e7['datasets']) == set(DATASETS))
    for ds in DATASETS:
        item = e7['datasets'][ds]
        low, high = item['ci95_psnr']
        tests['e7_' + ds] = (all(math.isfinite(v) for v in (low, high, item['mean_delta_psnr']))
                             and low <= high and item['mean_delta_psnr'] >= 0 and high > 0)
        if ds in ('Urban100', 'Manga109'):
            tests['e7_' + ds] = tests['e7_' + ds] and low > 0
    failed = [name for name, passed in tests.items() if not passed]
    if failed:
        raise ValueError('E8 entry criteria failed: ' + ', '.join(failed))
    return tests


def rgb_dataset(source):
    source = deepcopy(source)
    source['type'] = 'PairedImageDataset'
    for key in list(source):
        if key == 'filename_tmpl_depth' or 'depth' in key:
            source.pop(key)
    return source


def scaled_network(source, scale):
    network = deepcopy(source)
    network['upscale'] = scale
    if network['type'] == 'E4UDRMambaIRv2':
        network['performance_scaling'] = True
    return network


def transfer_state(target, source, reset_upsample=False, shared_head=None):
    """Whitelist only the scale-dependent PixelShuffle layers for resetting.

    Even x4/x2 layer 0 has the same shape but a different role. Reset ALL
    upsample.* at a scale change, with the identical RGB/UDR random tensors.
    Any other missing/extra/mismatched key is a hard error, never strict=False.
    """
    target, source = dict(target), dict(source)
    allowed = lambda key: reset_upsample and key.startswith('upsample.')
    if any(not allowed(key) for key in set(target) ^ set(source)):
        raise ValueError('E8 transfer has missing or unexpected non-upsampler tensors.')
    resets = []
    for key in target:
        if allowed(key):
            resets.append(key)
            if shared_head is not None:
                if key not in shared_head or shared_head[key].shape != target[key].shape:
                    raise ValueError('RGB/UDR scale-specific upsamplers differ.')
                target[key] = shared_head[key].clone()
        elif target[key].shape != source[key].shape:
            raise ValueError('E8 non-upsampler tensor changed shape: ' + key)
        else:
            target[key] = source[key]
    return target, resets


def validate_plan(path):
    plan = read(path)
    if plan['schema'] != 'E8-v1' or plan['scale'] not in SCALES:
        raise ValueError('Unknown E8 plan.')
    for item in plan['inputs'].values():
        verify(item)
    for item in plan['configs'].values():
        verify(item)
    for run in plan['runs']:
        for item in run['sources'].values():
            verify(item)
    return plan


def summarize_runs(runs, seeds):
    import statistics as st
    result = dict(seeds=seeds, per_dataset={})
    for ds in DATASETS:
        per = {}
        for label in ('rgb', 'udrv2'):
            per[label] = {m: dict(mean=st.mean(runs[label][s][ds][m] for s in seeds),
                                  std=st.stdev(runs[label][s][ds][m] for s in seeds))
                          for m in ('psnr', 'ssim')}
        gain = [runs['udrv2'][s][ds]['psnr'] - runs['rgb'][s][ds]['psnr'] for s in seeds]
        per['delta_psnr'] = dict(mean=st.mean(gain), std=st.stdev(gain), by_seed=gain)
        result['per_dataset'][ds] = per
    gains = [st.mean(runs['udrv2'][s][ds]['psnr'] - runs['rgb'][s][ds]['psnr']
                     for ds in DATASETS) for s in seeds]
    mean = st.mean(gains)
    result['five_set_delta_psnr'] = dict(mean=mean, std=st.stdev(gains), by_seed=gains)
    result['criteria'] = dict(minimum_gain_005=mean >= .05, stronger_gain_008=mean >= .08,
        urban100_positive=result['per_dataset']['Urban100']['delta_psnr']['mean'] > 0,
        manga109_positive=result['per_dataset']['Manga109']['delta_psnr']['mean'] > 0)
    # 'Clearly positive' has no numeric threshold in the proposal. Give its
    # actual interval later instead of inventing a Manga109 cutoff here.
    return result
