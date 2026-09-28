"""Persist meaningful semantic and real-tool regression results."""
import contextlib
import io
import sys
import time
import unittest

from study import HERE, ROOT, now, dump, sources, sha


def main():
    started, timer = now(), time.perf_counter()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_segmented_updates.py')
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (HERE / 'tests.log').write_text(stream.getvalue())
    print(stream.getvalue(), end='')
    dump(HERE / 'test-results.json', dict(command=[sys.executable, '-B', str(HERE / 'check.py')],
        started_at=started, finished_at=now(), elapsed_seconds=time.perf_counter() - timer,
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), successful=result.wasSuccessful(), sources=sources(),
        log_sha256=sha(HERE / 'tests.log')))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
