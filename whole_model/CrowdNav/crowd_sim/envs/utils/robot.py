from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
import torch

class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob,imitation_learning = True,recorded_unrotated_state=None,occ=None):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        if(imitation_learning):
            state = JointState(self.get_full_state(), ob)
            action = self.policy.predict(state)
        else:  #RL

            human_xy = recorded_unrotated_state[:, :, 9:11]  # [hist_len, n-human ,2(x,y)]
            robot_xy = recorded_unrotated_state[:, 0:1, 0:2]  # [hist_len, 1 ,2(x,y)]
            joint_xy = torch.cat([robot_xy, human_xy], dim=-2)

            state = JointState(self.get_full_state(), ob)
            action = self.policy.predict(state,joint_xy,occ)#.joint_xy = [hist_len ,n-human+1, 2(xy)]=[5 3 2]   ,  occ =[4,32] =[hist_len-1 ,32 ]
        return action
