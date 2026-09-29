# Learning rate, reward, and four-step length: two-stage experiment

The fixed protocol is `protocol.yml`. Stage 1 compares five ten-step groups at
100 episodes x 10 actions on seeds 0, 1, 2. Stage 2 extends the audited
four-step candidate from 250 to 1000 episodes on seeds 0–9. Both stages use
the existing source files without modifying their recorded fingerprints.

Run from the repository root, using the pinned local environment:

```sh
.tools/conda-env/bin/python -B experiments/lr-reward-two-stage/test_suite.py
.tools/conda-env/bin/python -B experiments/ten-step-reward-validation/test_suite.py
.tools/conda-env/bin/python -B experiments/four-step-reward/test_suite.py
.tools/conda-env/bin/python -B experiments/lr-reward-two-stage/stage1.py --phase preflight
.tools/conda-env/bin/python -B -u experiments/lr-reward-two-stage/stage1.py --phase train
.tools/conda-env/bin/python -B -u experiments/lr-reward-two-stage/stage1.py --phase evaluate
.tools/conda-env/bin/python -B experiments/lr-reward-two-stage/stage2.py --phase preflight
.tools/conda-env/bin/python -B -u experiments/lr-reward-two-stage/stage2.py --phase train
.tools/conda-env/bin/python -B experiments/lr-reward-two-stage/stage2.py --phase exact
.tools/conda-env/bin/python -B experiments/lr-reward-two-stage/analyze.py
```

Each stage writes checkpoints, per-step logs, manifests, and CEC proofs to the
Git-ignored `results/lr-reward-two-stage/`. Completed tasks are reused only
when their fingerprints and source checkpoint hashes still match. The final
report, compact CSVs, SVG charts, and evidence hash manifest are tracked here.

The three-seed ten-step comparison is exploratory. The four-step extension is
an independent question about training length and cannot replace the 100x10
same-budget comparison.
