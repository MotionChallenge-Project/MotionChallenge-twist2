# Motion Challenge · twist2

[twist2](https://github.com/amazon-far/TWIST2) checkpoint adapter. G1 29DoF; LAFAN40 is the first suite.

## Install and run

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-twist2.git
cd motion-challenge-twist2
uv sync
uv run evaluate.py
```

Reuses local `checkpoints/policy.onnx` or the external cache; missing artifacts are
fetched and hash-checked. Use `--weights PATH --data PATH --offline` for local files.
Results and traces are written outside Git. `--help` lists options.

The root `motion-challenge` submodule owns data, physics, metrics and verification.
This repository only adapts model inference; no `sim2real` dependency.

[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.


## Checkpoint source

[amazon-far/TWIST2](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/README.md) · [twist2_1017_20k.onnx](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/assets/ckpts/twist2_1017_20k.onnx). Author-released checkpoint.

The adapter pins this published artifact by SHA256. Artifact identity is separate from inference parity.

One `uv run evaluate.py` produces one verified `result.json`: success, completion,
three MPBPE alignments through first failure and the same three over the full motion.
Both scopes use the same checkpoint and source revision; no branch switching is needed.
