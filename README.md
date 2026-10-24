# Motion Challenge · twist2

[Author project](https://github.com/amazon-far/TWIST2) · [Checkpoint](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/assets/ckpts/twist2_1017_20k.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/motion-challenge-twist2.git
cd motion-challenge-twist2
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets rollout and
verification parallelism. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `motion-challenge`
submodule owns datasets, physics, metrics and verification. See `--help` for options.
