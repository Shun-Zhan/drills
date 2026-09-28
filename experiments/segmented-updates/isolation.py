"""Check the original worktree, refs and historical evidence without changing it."""
import json
from pathlib import Path
import subprocess
import time

from study import HERE, config, dump, sha, now


def validate(cfg):
    root = Path(cfg['runtime']['output_dir'])
    saved = json.loads((root / 'isolation-before.json').read_text())
    original = Path(saved['original_repository'])
    started = time.perf_counter()
    changed = [name for name, digest in saved['files'].items()
               if not (original / name).is_file() or sha(original / name) != digest]
    refs = subprocess.check_output(['git', 'for-each-ref', '--format=%(refname) %(objectname)',
        'refs/heads', 'refs/remotes'], cwd=original, text=True)
    refs = '\n'.join(row for row in refs.splitlines()
                     if not row.startswith('refs/heads/codex/segmented-updates ')) + '\n'
    status = subprocess.check_output(['git', 'status', '--porcelain=v1', '--branch'],
                                    cwd=original, text=True)
    record = dict(status='complete' if not changed and refs == saved['refs'] and status == saved['status'] else 'failed',
        historical_files_checked=len(saved['files']), changed_or_missing_files=changed,
        original_refs_unchanged=refs == saved['refs'], original_worktree_status_unchanged=status == saved['status'],
        original_repository=str(original), new_worktree=str(HERE.parents[1]),
        worktree_mode=saved['worktree_mode'], validated_at=now(), elapsed_seconds=time.perf_counter() - started)
    dump(root / 'isolation-validation.json', record)
    dump(HERE / 'isolation-validation.json', record)
    print('Isolation:', record['status'], 'files:', len(saved['files']), flush=True)
    if record['status'] != 'complete':
        raise RuntimeError('Original worktree, refs or historical evidence changed.')
    return record


if __name__ == '__main__':
    validate(config())
