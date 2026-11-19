"""TWIST2's author observation layout with batched history and inference."""
from tensor_common import TensorPD,roll_pitch,normalize,conjugate,rotate,torch


class TensorTWIST2(TensorPD):
    action_output='output'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ankles=[i for i,n in enumerate(self.names) if 'ankle' in n]
        self.history=torch.zeros((self.slots,10,127),device=self.device)

    def observations(self,state,reference):
        q,qd,omega,_,rotation=self.proprio(state);qd=qd.clone();qd[:,self.ankles]=0
        r=normalize(reference.root_orientation_xyzw[:,0].double())
        motion=torch.cat((rotate(conjugate(r),reference.root_linear_velocity_world_m_s[:,0].double())[:,:2],
            reference.root_position_world_m[:,0,2:3].double(),roll_pitch(r),
            rotate(conjugate(r),reference.root_angular_velocity_world_rad_s[:,0].double())[:,2:3],
            reference.joint_position_rad[:,0,self.order].double()),dim=-1)
        proprio=torch.cat((.25*omega,roll_pitch(rotation),q-self.default,.05*qd,self.last),dim=-1)
        current=torch.cat((motion,proprio),dim=-1).float()
        self.history.copy_(torch.where(self.fresh[:,None,None],current[:,None],self.history));self.fresh.zero_()
        inputs={'input':torch.cat((current,self.history.flatten(1),motion),dim=-1).float()}
        self.history[:,:-1]=self.history[:,1:].clone();self.history[:,-1]=current
        return inputs

ADAPTER_CLASS='TensorTWIST2'
NATIVE_CLASS='TWIST2'
