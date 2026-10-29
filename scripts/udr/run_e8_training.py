"""E8 train matched RGB / UDR-v2 chains, with verified same-phase recovery."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.udr.e8_common import (ROOT, load_yaml, read, record, sha256,
                                   transfer_state, validate_plan, write)


def initialize(plan, run, folder):
    import torch
    from basicsr.archs import build_network
    from scripts.udr.prepare_e4_integration import parameters
    path = folder/f'initial_s{run["seed"]}.json'
    identity = dict(seed=run['seed'], scale=plan['scale'], sources=run['sources'],
                    plan_sha256=sha256(folder/'E8_plan.json'))
    if path.exists():
        saved = read(path)
        if saved['identity'] != identity:
            raise ValueError('E8 initialization belongs to a different plan.')
        for label in ('rgb', 'udr'):
            if sha256(saved[label]['path']) != saved[label]['sha256']:
                raise ValueError('E8 initialized weights changed.')
        return
    # A new scale resets only PixelShuffle convolutions. The same initialized
    # head is copied into both controls even when equal-shaped x4 keys exist.
    torch.manual_seed(run['seed'])
    rgb_config = load_yaml(plan['configs'][f's{run["seed"]}_rgb']['path'])
    rgb_model = build_network(rgb_config['network_g'])
    rgb, reset_keys = transfer_state(rgb_model.state_dict(),
        parameters(run['sources']['rgb']['path']), plan['reset_upsample'])
    rgb_model.load_state_dict(rgb, strict=True)
    udr_config = load_yaml(plan['configs'][f's{run["seed"]}_udr_a']['path'])
    udr_model = build_network(udr_config['network_g'])
    udr, udr_resets = transfer_state(udr_model.state_dict(),
        parameters(run['sources']['udrv2']['path']), plan['reset_upsample'], rgb)
    udr_model.load_state_dict(udr, strict=True)
    if reset_keys != udr_resets:
        raise ValueError('E8 RGB/UDR reset different scale-dependent layers.')
    for label, state in (('rgb', rgb), ('udr', udr)):
        destination = Path(run['checkpoints'][label + '_initial'])
        if destination.exists():
            raise FileExistsError('Unrecorded E8 initialization: ' + str(destination))
        torch.save(dict(params=state, e8_initialization=identity), destination)
    write(path, dict(identity=identity, reset_keys=reset_keys,
        rgb=record(run['checkpoints']['rgb_initial']),
        udr=record(run['checkpoints']['udr_initial'])))


def receipt_path(folder, seed, label):
    return folder/f'completed_s{seed}_{label}.json'


def verify_completion(plan_path, run, label):
    folder = Path(plan_path).parent
    item = read(receipt_path(folder, run['seed'], label))
    if item['plan_sha256'] != sha256(plan_path):
        raise ValueError('E8 completed job belongs to a different plan.')
    if item['checkpoint'] != record(run['checkpoints'][label]):
        raise ValueError('E8 completed checkpoint changed: ' + label)
    return item['checkpoint']


def run_training(args):
    path = Path(args.plan).resolve()
    plan = validate_plan(path)
    for run in plan['runs']:
        if not args.dry_run:
            initialize(plan, run, path.parent)
        for label in ('rgb', 'udr_a', 'udrv2'):
            cfgpath = plan['configs'][f's{run["seed"]}_{label}']['path']
            destination = Path(run['checkpoints'][label])
            print(f'E8 x{plan["scale"]} seed {run["seed"]} {label}: {cfgpath} -> {destination}', flush=True)
            if args.dry_run:
                continue
            receipt = receipt_path(path.parent, run['seed'], label)
            if receipt.exists():
                verify_completion(path, run, label)
                if not args.resume:
                    raise FileExistsError('Completed E8 stage; use --resume to verify and skip it.')
                continue
            config = load_yaml(cfgpath)
            states = ROOT/'experiments'/config['name']/'training_states'
            candidates = sorted((p for p in states.glob('*.state') if p.stem.isdigit()),
                                key=lambda p: int(p.stem))
            if destination.exists() or candidates:
                if not args.resume:
                    raise FileExistsError('E8 stage has training output; use --resume for same-phase recovery.')
                if not candidates:
                    raise ValueError('E8 output has no matching optimizer state; cannot recover faithfully.')
                import torch
                latest = candidates[-1]
                state = torch.load(latest, map_location='cpu')
                step = int(state['iter'])
                if step != int(latest.stem) or step > config['train']['total_iter']:
                    raise ValueError('E8 optimizer state step differs from this phase.')
                if step == config['train']['total_iter']:
                    if not destination.is_file():
                        raise FileNotFoundError(destination)
                else:
                    import yaml
                    # BasicSR check_resume loads this same phase's network and
                    # optimizer/scheduler. Never --auto_resume across A and B.
                    config['path']['resume_state'] = str(latest)
                    cfgpath = str(path.parent/f'resume_s{run["seed"]}_{label}.yml')
                    Path(cfgpath).write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
                    subprocess.run([sys.executable, str(ROOT/'basicsr/train.py'), '-opt', cfgpath], cwd=ROOT, check=True)
            else:
                subprocess.run([sys.executable, str(ROOT/'basicsr/train.py'), '-opt', cfgpath], cwd=ROOT, check=True)
            if not destination.is_file():
                raise FileNotFoundError('E8 stage did not finish: ' + str(destination))
            write(receipt, dict(plan_sha256=sha256(path), checkpoint=record(destination),
                                seed=run['seed'], stage=label))
    print('E8 training dry-run complete.' if args.dry_run else 'E8 scale training completed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Verify finished jobs; restore latest state only within the current phase.')
    run_training(parser.parse_args())


if __name__ == '__main__':
    main()
