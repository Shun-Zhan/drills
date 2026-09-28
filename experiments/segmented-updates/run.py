"""Execute the preregistered study; outputs are never silently overwritten."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

from filelock import FileLock

from study import HERE, GROUPS, SEEDS, config, prepare, execute, train_task, evaluate_task, dump

BUNDLE_PYTHON = '/Users/zhan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'train', 'evaluate', 'verify', 'analyze', 'package', 'all'))
    parser.add_argument('--task', nargs=2, metavar=('GROUP', 'SEED'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plot-python', default=BUNDLE_PYTHON,
                        help='Separate Python with ReportLab and PyMuPDF.')
    args = parser.parse_args()
    cfg = config()
    root = Path(cfg['runtime']['output_dir'])
    if args.task:
        group, seed_text = args.task
        seed = int(seed_text)
        if args.phase not in ('train', 'evaluate') or group not in GROUPS or seed not in SEEDS:
            parser.error('Task is outside the fixed study.')
        if args.phase == 'evaluate' and group == 'U' and seed != 10:
            parser.error('U has one physical evaluation bank, shared across training seeds.')
        folder = root / ('training' if args.phase == 'train' else 'evaluation') / group / f'seed-{seed}'
        try:
            (train_task if args.phase == 'train' else evaluate_task)(cfg, group, seed, args.resume)
        except Exception as error:
            dump(folder / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
            raise
        return
    root.mkdir(parents=True, exist_ok=True)
    phases = ('prepare', 'train', 'evaluate', 'verify', 'analyze', 'package') if args.phase == 'all' else (args.phase,)
    with FileLock(str(root / '.lock'), timeout=0):
        for phase in phases:
            try:
                if phase == 'prepare':
                    prepare(cfg)
                elif phase in ('train', 'evaluate'):
                    execute(cfg, phase, args.resume)
                else:
                    from results import verify, analyze, package
                    {'verify': verify, 'analyze': analyze, 'package': package}[phase](cfg)
                    if phase == 'analyze':
                        subprocess.run([args.plot_python, '-B', str(HERE / 'plot.py')], check=True)
            except Exception as error:
                dump(root / (phase + '-failure.json'), dict(error=repr(error), traceback=traceback.format_exc()))
                if phase in ('train', 'evaluate'):
                    from results import analyze
                    analyze(cfg)
                raise


if __name__ == '__main__':
    main()
