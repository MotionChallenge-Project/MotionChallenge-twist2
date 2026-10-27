"""Compare against pinned sim2real observation/reset/action code and its wrapped ONNX.

Validation only: uv run --with onnx --with pyyaml validate_inference.py --help
Runtime keeps the original author ONNX and has no sim2real dependency.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R

from adapter import TWIST2
from motion_challenge.adapters.onnx import vector
from motion_challenge.sdk import ReferenceFrame, StatePacket


def execute(nodes, scope, path):
    tree = ast.Module(ast.parse('from __future__ import annotations').body + nodes, [])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'weights', 'deployment-config', 'deployment-weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root / 'deployment-sources.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, digest in pins['files'].items():
        assert sha(args.source / name) == digest, name
    assert sha(args.deployment_config) == pins['policy_yaml']['sha256']
    assert sha(args.deployment_weights) == pins['wrapped_onnx']['sha256']
    runtime = json.loads((root / 'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    policy = TWIST2(root, runtime)
    cfg = yaml.safe_load(args.deployment_config.read_text())
    for attr, key in [('names', 'policy_joint_names'), ('kp', 'joint_kp'),
                      ('kd', 'joint_kd'), ('default', 'default_joint_pos'), ('scale', 'action_scale')]:
        expected = cfg[key] if attr == 'names' else vector(cfg[key], policy.names)
        assert np.array_equal(getattr(policy, attr), expected), attr
    assert policy.cfg.get('action_clip') is None
    assert runtime['reference_offsets_s'] == [0.0]
    sim_path = args.source / 'sim2real/sim_env/integrated_sim2sim.py'
    sim_tree = ast.parse(sim_path.read_text())
    env_dt = [n.value.value for n in ast.walk(sim_tree) if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Name) and n.target.id == 'env_dt']
    assert env_dt == [runtime['control_period_s']] == [.02]
    setup = next(n for n in ast.walk(sim_tree) if isinstance(n, ast.FunctionDef) and n.name == 'setup_policy')
    conversion = next(n for n in setup.body if isinstance(n, ast.FunctionDef) and n.name == 'policy')
    target_nodes = [n for n in conversion.body if isinstance(n, (ast.Assign, ast.AugAssign))
                    and ('q_target' in ast.unparse(n))]

    # Execute the public methods themselves, with a minimal reference/state provider.
    class Motion:
        def reset(self): pass
        def update(self, data): pass
        def _motion_step_index(self, step): return step
    scope = {'np': np, 'motion_obs': Motion}
    math_path = args.source / 'sim2real/utils/math.py'
    execute([n for n in ast.parse(math_path.read_text()).body if isinstance(n, ast.FunctionDef)
             and n.name == 'quat_rotate_inverse_numpy'], scope, math_path)
    obs_path = args.source / 'sim2real/rl_policy/observations/twist2.py'
    nodes = ast.parse(obs_path.read_text()).body
    cls = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == 'twist2_input')
    cls.keywords = []
    execute([n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == '_quat_to_roll_pitch'] + [cls], scope, obs_path)
    native = scope['twist2_input'].__new__(scope['twist2_input'])
    native.env = NS(num_actions=29)
    native.history = np.zeros((10, 127), np.float32)
    native.input = np.zeros((1, 1432), np.float32)
    native.current_step = native.future_step = 0
    native.joint_ids = list(range(29))
    native.ankle_indices = [i for i, n in enumerate(policy.names) if 'ankle' in n]
    native.default_joint_pos = policy.default.astype(np.float32)
    native.ang_vel_scale, native.dof_pos_scale, native.dof_vel_scale = .25, 1., .05
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    wrapped = ort.InferenceSession(str(args.deployment_weights), options, providers=['CPUExecutionProvider'])
    arrays = lambda p: [onnx.numpy_helper.to_array(t) for t in onnx.load(p).graph.initializer]
    original, other = arrays(args.weights), arrays(args.deployment_weights)
    assert len(original) == len(other) == 32
    assert all(np.array_equal(a, b) for a, b in zip(original, other))
    rng = np.random.default_rng(271026)
    names = tuple(np.asarray(policy.names)[rng.permutation(29)])
    order = [names.index(n) for n in policy.names]
    errors = {k: 0. for k in ('reset_history', 'observation', 'same_input_network', 'network', 'motor_target', 'synthetic_motor_target')}
    def compare(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a, dtype=np.float64) - b))))
    for reset in range(4):
        previous = np.zeros(29, np.float32)
        for step in range(40):
            rot, rr = R.from_euler('xyz', rng.uniform(-.7, .7, (2, 3)))
            q, qd, omega = rng.normal(0, .35, 29), rng.normal(0, .5, 29), rng.normal(0, .3, 3)
            state = StatePacket(step * .02, names, tuple(q), tuple(qd), (0., 0., .8), tuple(rot.as_quat()),
                                (0.,) * 3, tuple(rot.apply(omega)))
            ref = ReferenceFrame(step * .02, step * .02, tuple(rng.normal(0, .35, 29)),
                                 tuple(rng.normal(0, .3, 3)), tuple(rr.as_quat()), (0.,) * 29,
                                 tuple(rng.normal(0, .3, 3)), tuple(rng.normal(0, .3, 3)))
            native.state_processor = NS(joint_pos=q[order].astype(np.float32), joint_vel=qd[order].astype(np.float32),
                                        root_ang_vel_b=omega.astype(np.float32), root_quat_w=rot.as_quat()[[3, 0, 1, 2]].astype(np.float32))
            for field, value in [('root_pos_future_w', ref.root_position_world_m),
                                 ('root_quat_future_w', rr.as_quat()[[3, 0, 1, 2]]),
                                 ('root_lin_vel_future_w', ref.root_linear_velocity_world_m_s),
                                 ('root_ang_vel_future_w', ref.root_angular_velocity_world_rad_s),
                                 ('joint_pos_future', np.asarray(ref.joint_position_rad)[order])]:
                setattr(native, 'ref_' + field, np.asarray(value, np.float32)[None, None])
            if step == 0:
                policy.reset(state, (ref,), reset)
                native.reset()
                compare('reset_history', policy.history, native.history)
            # Feed the same previous action to isolate observation arithmetic error.
            policy.last = previous.copy()
            native.update({'action': previous})
            inputs = policy.observations(state, (ref,))
            compare('observation', inputs['input'], native.input)
            feed = {k: v[None] for k, v in native._input_parts.items()}
            raw = wrapped.run(None, feed)[0].reshape(-1)
            compare('same_input_network', policy.infer({'input': native.input})['output'].reshape(-1), raw)
            compare('network', policy.infer(inputs)['output'].reshape(-1), raw)
            fn = policy.observations
            policy.observations = lambda *_: inputs
            result = policy.act(state, (ref,))
            policy.observations = fn
            scope.update(self=NS(default_dof_angles=policy.default, controlled_joint_indices=np.arange(29), action_scale=policy.scale), action=raw)
            execute(target_nodes, scope, sim_path)
            compare('motor_target', np.asarray(result.joint_position_target_rad)[order], scope['q_target'])
            previous = raw
        # Exercise outputs beyond the former clipping range explicitly.
        raw = np.linspace(-20, 20, 29, dtype=np.float32)
        fn = policy.infer; policy.infer = lambda _: {'output': raw}
        result = policy.act(state, (ref,)); policy.infer = fn
        scope['action'] = raw; execute(target_nodes, scope, sim_path)
        compare('synthetic_motor_target', np.asarray(result.joint_position_target_rad)[order], scope['q_target'])
        assert np.array_equal(policy.last, raw)
    limits = {k: 2e-4 for k in errors}
    limits.update(same_input_network=0., synthetic_motor_target=0.)
    report = dict(passed=all(errors[k] <= limits[k] for k in errors), cases=160, resets=4,
                  max_abs_error=errors, limits=limits, checkpoint_sha256=runtime['weights_sha256'],
                  deployment_source_commit=pins['commit'], control_hz=50, identical_initializer_tensors=32,
                  scope='Pinned sim2real reset/history, observation methods, ONNX wrapper, action-to-target statements and PD config. Original author network unchanged.',
                  differences='Reference preparation, initial state, 2ms physics, safety/torque limits and metrics remain owned by MotionChallenge Core. Float64 Core state versus float32 deployment state is compared within the recorded tolerance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
