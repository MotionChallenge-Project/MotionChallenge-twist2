"""Author TWIST2 20k network with the pinned sim2real deployment configuration."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController,roll_pitch


class TWIST2(ONNXController):
    action_output='output'
    raw_action_history=True

    def reset(self,state,reference,seed):
        super().reset(state,reference,seed)
        # The pinned sim2real deployment fills history with the first observation.
        self.ankles=[i for i,n in enumerate(self.names) if 'ankle' in n]
        _,current=self.current_observation(state,reference)
        self.history=np.repeat(current[None],10,axis=0)

    def current_observation(self,state,reference):
        q,qd,omega,gravity,rotation=self.proprio(state)
        qd=qd.copy();qd[self.ankles]=0
        f=reference[0];r=R.from_quat(f.root_orientation_xyzw)
        motion=np.r_[r.inv().apply(f.root_linear_velocity_world_m_s)[:2],f.root_position_world_m[2],
            roll_pitch(r),r.inv().apply(f.root_angular_velocity_world_rad_s)[2],np.asarray(f.joint_position_rad)[self.order]]
        proprio=np.r_[.25*omega,roll_pitch(rotation),q-self.default,.05*qd,self.last]
        current=np.r_[motion,proprio].astype(np.float32)
        return motion,current

    def observations(self,state,reference):
        motion,current=self.current_observation(state,reference)
        inputs={'input':np.r_[current,self.history.reshape(-1),motion][None]}
        self.history=np.concatenate([self.history[1:],current[None]],axis=0)
        return inputs


def create(root,runtime):return TWIST2(root,runtime)
