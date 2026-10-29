"""E8 same-data/same-scale matched-seed PSNR/SSIM and image bootstrap."""
import argparse
import csv
from pathlib import Path
import statistics as st
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.udr.e8_common import (ROOT, DATASETS, read, record, sha256,
                                   summarize_runs, validate_plan, write)
from scripts.udr.run_e8_training import verify_completion
from scripts.udr.e7_image_statistics import analyze, pair_model_rows


def evaluate(args):
    plan = validate_plan(args.plan)
    output = ROOT/f'results/E8_scaling/x{plan["scale"]}'
    output.mkdir(parents=True, exist_ok=True)
    raw = dict(rgb=[], udrv2=[])
    runs = dict(rgb={}, udrv2={})
    provenance = dict(rgb={}, udrv2={})
    for run in plan['runs']:
        for label in raw:
            seed = run['seed']
            provenance[label][str(seed)] = verify_completion(args.plan, run, label)
            config = plan['configs'][f's{seed}_{label}_test']['path']
            result = output/f'{label}_s{seed}_images.json'
            # The E7 worker is already scale-generic. It preserves the E6
            # partition/overlap, sorted images and RNG reset per dataset.
            subprocess.run([sys.executable, str(ROOT/'scripts/udr/e7_image_statistics.py'),
                            '--worker', config, str(result)], cwd=ROOT, check=True)
            values = read(result)
            raw[label].extend(values)
            runs[label][seed] = {ds: {metric: st.mean(row[metric] for row in values if row['dataset'] == ds)
                                       for metric in ('psnr', 'ssim')} for ds in DATASETS}
    rows = pair_model_rows(raw['rgb'], raw['udrv2'], plan['seeds'])
    report = summarize_runs(runs, plan['seeds'])
    bootstrap = analyze(rows, plan['seeds'], args.bootstrap_seed)
    report.update(scale=plan['scale'], training_data='DF2K', checkpoints=provenance,
        plan=record(args.plan), per_image_bootstrap=bootstrap,
        protocol=f'uint8 Y-channel, bicubic x{plan["scale"]}, crop={plan["scale"]}; matched independent training/inference seeds',
        added_updates_per_model=plan['added_updates_per_model'], baseline_note=plan['baseline_note'])
    report['criteria']['manga109_ci_positive'] = bootstrap['datasets']['Manga109']['ci95_psnr'][0] > 0
    write(output/'E8_metrics.json', report)
    with (output/'E8_per_image.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# E8 DF2K x{plan["scale"]}', '', report['protocol'], '',
             '| Dataset | RGB PSNR | UDR PSNR | RGB SSIM | UDR SSIM | paired ΔPSNR | 95% image CI |',
             '|---|---:|---:|---:|---:|---:|---|']
    for ds in DATASETS:
        item = report['per_dataset'][ds]
        values = [item[label][metric] for metric in ('psnr', 'ssim') for label in ('rgb', 'udrv2')]
        text = ' | '.join(f'{v["mean"]:.4f}±{v["std"]:.4f}' for v in values)
        delta = item['delta_psnr']
        lines.append(f'| {ds} | {text} | {delta["mean"]:+.4f}±{delta["std"]:.4f} | '
                     f'{bootstrap["datasets"][ds]["ci95_psnr"]} |')
    lines += ['', 'Five-set equal-weight ΔPSNR: ' + str(report['five_set_delta_psnr']),
              'Criteria: ' + str(report['criteria']), '', plan['baseline_note'],
              'Image bootstrap is conditional on the observed trained seeds. Seed std is reported separately.', '']
    (output/'E8_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved E8 measured scale results:', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--bootstrap-seed', type=int, default=2026)
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
