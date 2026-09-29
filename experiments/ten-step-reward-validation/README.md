# Ten-step reward validation

This suite compares the original reward and `best_feasible_v1` on i2c and max.
The fixed protocol is in `protocol.yml`. Its primary outcome is independent,
frozen-policy evaluation, not the best mapping encountered during training.
The ten training seeds are 20–29; historical seeds 0–4 are background only.

The prior ignored results were archived, read-only, at
`/Users/zhan/Downloads/test/experiment-archives/four-step-reward-682ee92.tar.zst`.
The archive hash is in the adjacent `.sha256` file. This suite writes only to
`results/ten-step-reward-validation/` and this directory's derived report/CSVs.

From the repository root:

```sh
.tools/conda-env/bin/python -B experiments/ten-step-reward-validation/test_suite.py
.tools/conda-env/bin/python -B experiments/ten-step-reward-validation/run.py --phase preflight
.tools/conda-env/bin/python -B -u experiments/ten-step-reward-validation/run.py --phase train
.tools/conda-env/bin/python -B -u experiments/ten-step-reward-validation/evaluate.py
.tools/conda-env/bin/python -B -u experiments/ten-step-reward-validation/analyze.py
```

Training and evaluation use three workers with a 30-minute hard timeout per
task. Re-running their main commands resumes only incomplete tasks with the
identical source/protocol/tool/circuit fingerprint. Completed results are never
replaced. An interrupted task resumes at the last atomic checkpoint; a partial
episode is regenerated from the saved RNG state. Analysis refuses missing or
duplicate trajectories and does not publish a passing report from partial data.

The final report, four derived CSVs, `preflight.json`, `statistics.json`, and
`evidence-hashes.json` live here. Full logs, checkpoints, CEC proofs, and run
manifests remain under the ignored results directory.
