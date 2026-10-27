"""Check per-slot history, reordered calls and partial resets against scalar inference."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from motion_challenge.adapters.parallel import ParallelControllers
from motion_challenge.resources import lafan_paths
from motion_challenge.data.dataset import load_csv_motion
from motion_challenge.simulation.physics import World
from motion_challenge.runtime.observations import PROPRIOCEPTIVE, controller_state
from restricted import create

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--data', type=Path, default=Path('~/.cache/MotionChallenge/lafan').expanduser())
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
runtime = json.loads(args.runtime.read_text())
paths = lafan_paths(); manifest = json.loads(paths['suite'].read_text())
world = World(paths['robot']); states = []; references = []
for item in manifest['files']:
    motion = load_csv_motion(args.data / item['path'], manifest['joint_names'], manifest['source_fps'])
    world.reset(motion)
    state = controller_state(world.state(), PROPRIOCEPTIVE)
    states.append([replace(state, time_s=i * .02) for i in range(12)])
    references.append([(world.reference(i * .02),) for i in range(12)])
scalar = [create(root, runtime) for _ in states]
parallel = ParallelControllers(root, runtime, create)
slots = list(range(40)); seeds = list(range(40))
try:
    for i, p in enumerate(scalar): p.reset(states[i][0], references[i][0], seeds[i])
    parallel.reset_batch(slots, [s[0] for s in states], [r[0] for r in references], seeds)
    count = 0
    for step in range(12):
        if step == 6:
            ids = [1, 7, 21]
            for i in ids: scalar[i].reset(states[i][step], references[i][step], seeds[i])
            parallel.reset_batch(ids, [states[i][step] for i in ids], [references[i][step] for i in ids], [seeds[i] for i in ids])
        order = slots[::-1] if step % 2 else slots
        results = parallel.act_batch(order, [states[i][step] for i in order], [references[i][step] for i in order])
        for i, actual in zip(order, results):
            expected = scalar[i].act(states[i][step], references[i][step])
            assert asdict(expected) == asdict(actual), (i, step)
            count += 1
finally:
    parallel.close()
    for p in scalar: p.close()
report = dict(passed=True, cases=count, slots=40, workers=runtime['inference_workers'],
              max_abs_error=0., checkpoint_sha256=runtime['weights_sha256'],
              scope='Scalar versus threaded inference on 40 LAFAN initial states and advancing references, reversed slot order and partial resets. Not a full rollout.')
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
