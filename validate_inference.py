"""Native TWIST2 parity check; run with `uv run --with torch validate_inference.py ...`."""
import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from adapter import TWIST2
from motion_challenge.sdk import ReferenceFrame, StatePacket


def execute(nodes, scope, path):
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body + nodes, []), str(path), 'exec'), scope)


def target(node):
    return ast.unparse(node.targets[0]) if isinstance(node, ast.Assign) else ''


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root / 'native-sources.json').read_text())
    for name, sha in pins['files'].items():
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == sha, name
    runtime = json.loads((root / 'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    policy = TWIST2(root, runtime)
    path = args.source / 'deploy_real/server_low_level_g1_sim.py'
    tree = ast.parse(path.read_text())
    scope = {'np': np, 'torch': torch, 'deque': deque}
    execute([n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OnnxPolicyWrapper'], scope, path)
    rotation_path = args.source / 'deploy_real/data_utils/rot_utils.py'
    execute([n for n in ast.parse(rotation_path.read_text()).body if isinstance(n, ast.FunctionDef)], scope, rotation_path)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RealTimePolicyController')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    fields = ('num_actions', 'last_action', 'default_dof_pos', 'stiffness', 'damping', 'action_scale',
              'ankle_idx', 'n_mimic_obs', 'n_proprio', 'n_obs_single', 'history_len', 'total_obs_size', 'proprio_history_buf')
    initialization = [n for n in init.body if target(n) in {'self.' + name for name in fields}
                      or isinstance(n, ast.For) and 'self.proprio_history_buf.append' in ast.unparse(n)]
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    branch = next(n for n in ast.walk(run) if isinstance(n, ast.If) and 'i % self.sim_decimation' in ast.unparse(n.test))
    obs_names = {'rpy', 'obs_body_dof_vel', 'obs_body_dof_vel[self.ankle_idx]', 'obs_proprio', 'obs_full', 'obs_hist', 'future_obs', 'obs_buf'}
    observation = [n for n in branch.body if target(n) in obs_names
                   or isinstance(n, ast.Expr) and ast.unparse(n).startswith('self.proprio_history_buf.append(')]
    action_names = {'self.last_action', 'raw_action', 'scaled_actions', 'pd_target'}
    conversion = [n for n in branch.body if target(n) in action_names]
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    frequency = next(n for n in ast.walk(main_fn) if isinstance(n, ast.Call) and n.args
                     and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--policy_frequency')
    hz = next(k.value.value for k in frequency.keywords if k.arg == 'default')
    assert runtime['control_period_s'] == 1 / hz
    motion_path = args.source / 'deploy_real/server_motion_lib.py'
    build = next(n for n in ast.parse(motion_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'build_mimic_obs')
    start = next(i for i, n in enumerate(build.body) if target(n) == '(roll, pitch, yaw)')
    motion_nodes = [n for n in build.body[start:] if not isinstance(n, ast.Return)]
    native_model = scope['OnnxPolicyWrapper'](policy.session, 'input')
    rng = np.random.default_rng(241026)
    names = tuple(np.array(policy.names)[rng.permutation(29)])
    order = [names.index(name) for name in policy.names]
    errors = {key: 0. for key in ('default', 'scale', 'kp', 'kd', 'reference', 'observation', 'network', 'action', 'motor_target', 'synthetic_motor_target')}
    def compare(key, a, b):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    for reset in range(4):
        native = NS()
        scope['self'] = native
        execute(initialization, scope, path)
        for key, actual, expected in [('default', policy.default, native.default_dof_pos),
                                     ('scale', policy.scale, native.action_scale),
                                     ('kp', policy.kp, native.stiffness), ('kd', policy.kd, native.damping)]:
            compare(key, actual, expected)
        for step in range(40):
            actual, ref_rot = R.random(random_state=rng), R.random(random_state=rng)
            q, qd, omega = rng.normal(0, .35, 29), rng.normal(0, .5, 29), rng.normal(0, .3, 3)
            state = StatePacket(step / hz, names, tuple(q), tuple(qd), (0., 0., .8), tuple(actual.as_quat()),
                                (0.,) * 3, tuple(actual.apply(omega)))
            ref = ReferenceFrame(step / hz, step / hz, tuple(rng.normal(0, .35, 29)),
                                 tuple(rng.normal(0, .3, 3)), tuple(ref_rot.as_quat()),
                                 (0.,) * 29, tuple(rng.normal(0, .3, 3)), tuple(rng.normal(0, .3, 3)))
            for name, value in [('root_pos', ref.root_position_world_m), ('root_rot', ref.root_orientation_xyzw),
                                ('root_vel', ref.root_linear_velocity_world_m_s), ('root_ang_vel', ref.root_angular_velocity_world_rad_s),
                                ('dof_pos', np.array(ref.joint_position_rad)[order])]:
                scope[name] = torch.tensor(np.array(value)[None], dtype=torch.float64)
            scope.update(mask_indicator=False, device=torch.device('cpu'))
            execute(motion_nodes, scope, motion_path)
            motion = scope['mimic_obs_buf'].numpy().reshape(-1)
            scope.update(dof_pos=q[order], dof_vel=qd[order], quat=actual.as_quat()[[3, 0, 1, 2]],
                         ang_vel=omega, action_mimic=motion)
            if step == 0:
                policy.reset(state, (ref,), reset)
            execute(observation, scope, path)
            expected = scope['obs_buf'][None].astype(np.float32)
            inputs = policy.observations(state, (ref,))
            compare('reference', inputs['input'][0, :35], motion)
            compare('observation', inputs['input'], expected)
            raw = native_model(torch.from_numpy(expected)).numpy().reshape(-1)
            compare('network', policy.infer({'input': expected})['output'].reshape(-1), raw)
            scope['raw_action'] = raw
            execute(conversion, scope, path)
            fn = policy.observations
            policy.observations = lambda *_: inputs
            result = policy.act(state, (ref,))
            policy.observations = fn
            compare('action', policy.last, native.last_action)
            compare('motor_target', np.array(result.joint_position_target_rad)[order], scope['pd_target'])
        synthetic = np.linspace(-20, 20, 29, dtype=np.float32)
        scope['raw_action'] = synthetic
        execute(conversion, scope, path)
        fn = policy.infer
        policy.infer = lambda _: {'output': synthetic}
        result = policy.act(state, (ref,))
        policy.infer = fn
        compare('synthetic_motor_target', np.array(result.joint_position_target_rad)[order], scope['pd_target'])
        compare('action', policy.last, native.last_action)
    limits = {key: 2e-4 for key in errors}
    limits.update(network=0., default=1e-7, scale=0., kp=0., kd=0.)
    report = {'passed': all(errors[k] <= limits[k] for k in errors), 'cases': 160, 'resets': 4,
              'max_abs_error': errors, 'limits': limits, 'checkpoint_sha256': runtime['weights_sha256'],
              'source_commit': pins['source_commit'], 'control_hz': hz,
              'scope': 'Original author initialization, motion-command tensor operations, observation/history statements, '
                       'ONNX wrapper, raw last action, clipped motor target and PD gains. Transport, viewer and '
                       'motion-file loading excluded; reference tensors use CPU, unchanged math. Benchmark owns physical torque limits.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
