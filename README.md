# MotionChallenge · twist2

[Author project](https://github.com/amazon-far/TWIST2) · [Checkpoint](https://github.com/amazon-far/TWIST2/blob/b06178f19a22f2138cbd31f60c6d494bc263f67d/assets/ckpts/twist2_1017_20k.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-twist2.git
cd MotionChallenge-twist2
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. Tuned inference defaults; CPU slots scale with
available cores. `--jobs N` overrides the slots. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `MotionChallenge-Core`
submodule owns datasets, physics, metrics and verification. See `--help` for options.

## Deployment

Original author weights, using [sim2real’s deployment configuration](https://github.com/egalahad/sim2real/tree/287ac157e6a7143340da7a9fc2fa5ff552031bc8): **50 Hz**, first-observation history fill, wrist **Kp 40 / Kd 5**, and unclipped network actions. These differ from the author’s original 100 Hz sim2sim setup; benchmark physics and scoring remain shared. [Numerical comparison](validation/deployment-parity.json) · [Source pins](deployment-sources.json).

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](https://github.com/MotionChallenge-Project/MotionChallenge-Core/blob/b08ef12c6898ee866a81bdce055099ee2d0cee4b/docs/OBSERVATIONS.md).
