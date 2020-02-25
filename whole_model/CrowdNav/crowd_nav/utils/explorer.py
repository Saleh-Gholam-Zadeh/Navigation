import logging
import copy
import torch
from crowd_sim.envs.utils.info import *

import time

class Explorer(object):
    def __init__(self,traj_model, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None


        self.traj_model = traj_model
        self.history_max_len = 9
        self.pool = Pooling(type_='directional', hidden_dim=128, cell_side=2)

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)
    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []

            jj = 0
            my_history = []             # similar to  history_state=[]
            my_history_pooling = []     # similar to  history_occ=[]

            t0 = time.clock()
            while not done:
                tt = jj + self.history_max_len

                if(not(imitation_learning)):      #havaset bashe tu halate rl state ha rotate shodan


                    robot_s = self.robot.get_full_state()
                    robot_state = torch.tensor([robot_s.px, robot_s.py ,robot_s.vx ,robot_s.vy ,robot_s.radius ,robot_s.gx ,robot_s.gy ,robot_s.v_pref,robot_s.theta])
                    full_st = torch.zeros(len(ob),14)

                    for kk ,human in enumerate(ob):    #unrotated ezafe kon harman !!! be last state
                        human_s = torch.tensor([human.px, human.py,human.vx,human.vy,human.radius])
                        full_st[kk,:] =  torch.cat([robot_state,human_s])
                    my_history.append(full_st)


                    human_xy = full_st[:,9:11]
                    robot_xy = full_st[0:1,0:2]
                    joint_xy = torch.cat([robot_xy,human_xy],dim=-2)

                    if(jj==0):
                        my_history = my_history*(self.history_max_len)
                        my_history_pooling.append(torch.zeros(1, 32))
                        my_history_pooling=my_history_pooling*(self.history_max_len-1)
                        recorded_occ = torch.zeros(self.history_max_len - 1, 32)
                    recorded_unrotated_state = torch.stack(   [my_history[k] for k in range(tt- self.history_max_len, tt)], dim=0)  # [5 2 14] or [history_len n-agent 14]]

                    if (jj > 0):
                        state_tensorized_prev = my_history[-2]  # [num-human 14]   -2because we already appended current state into the history

                        human_xy_prev = state_tensorized_prev[:, 9:11]  # [  n-human ,2(x,y)]
                        robot_xy_prev = state_tensorized_prev[0:1, 0:2]  # [ 1 ,2(x,y)]
                        joint_xy_prev = torch.cat([robot_xy_prev, human_xy_prev], dim=-2)  # [n-human+1, 2(x,y)]
                        occ_map = self.pool(None, joint_xy_prev, joint_xy)  # None=hidden_state   we dont really need obs1 since we have velocity (or obs1=obs2-v*dt)
                        my_history_pooling.append(occ_map)
                        recorded_occ = torch.stack( [my_history_pooling[k] for k in range(tt - self.history_max_len, tt - 1)], dim=0).squeeze()  # [4  32] or [history_len-1 , n*n*2]] //



                if(imitation_learning):
                    action = self.robot.act(ob, imitation_learning, None,None)
                    ob, reward, done, info = self.env.step(action)
                    states.append(self.robot.policy.last_state)  #contatins JointState object
                    jj = jj + 1
                else:
                    t2=time.clock()
                    action = self.robot.act(ob,imitation_learning,recorded_unrotated_state,recorded_occ)
                    #print("RL action time",time.clock()-t2)
                    ob, reward, done, info = self.env.step(action)
                    jj = jj + 1
                    states.append(self.robot.policy.last_state_unrotated)








                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            #print('1 experiment_time:',time.clock()-t0)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        #history_max_len = 3
        history_state=[]
        history_occ=[]

        history_state_plus=[]
        history_occ_plus = []
        for i, state in enumerate(states):
            #print(i)
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning: #(IL phase)
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state_rotated = self.target_policy.transform(state)
                state_tensorized = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)  #[num-human 14]
                                          for human_state in state.human_states], dim=0)

                human_xy = state_tensorized[:, 9:11]  # [n-human ,2(x,y)]
                robot_xy = state_tensorized[0:1, 0:2]  # [1 ,2(x,y)]
                joint_xy = torch.cat([robot_xy, human_xy], dim=-2)  # [n-human+1, 2(x,y)]
                state_simple = state_tensorized

                ii = i + self.history_max_len

                history_state.append(state_simple)                                          # i=[                         0    1    2   3   4     5
                if(i==0):                                                                  # ii=[ 0    1    2   3   4     5     6       ]
                    history_state = history_state*self.history_max_len       #  history_state = [ s0  s0   s0   s0  s0    s1   s2      ]
                    history_occ.append(torch.zeros(1,32))                    # history_occ    =      [ 0    0    0   0   c01  c12           ]
                    history_occ = history_occ*(self.history_max_len-1)
                    recorded_occ = torch.zeros(self.history_max_len - 1, 32)  #[4 32] =[hist-1,32]

                recorded_unrotated_state = torch.stack( [history_state[k] for k in range( ii-self.history_max_len, ii)],   dim=0)  # [5 2 14] or [history_len n-agent 14]]

                if(i>0):
                    state_tensorized_prev = torch.cat([torch.Tensor([states[i-1].self_state + human_state]).to(self.device)  #[num-human 14]
                                              for human_state in states[i-1].human_states], dim=0)
                    human_xy_prev = state_tensorized_prev[:, 9:11]  # [  n-human ,2(x,y)]
                    robot_xy_prev = state_tensorized_prev[0:1, 0:2]  # [ 1 ,2(x,y)]
                    joint_xy_prev = torch.cat([robot_xy_prev , human_xy_prev], dim=-2)  # [n-human+1, 2(x,y)]
                    occ_map = self.pool(None ,joint_xy_prev, joint_xy) #None=hidden_state   we dont really need obs1 since we have velocity (or obs1=obs2-v*dt)
                    history_occ.append(occ_map)

                    recorded_occ = torch.stack([history_occ[k] for k in range(ii - self.history_max_len, ii - 1)],dim=0).squeeze()  # [4  32] or [history_len n*n*2]]
                    #check kon hatman



                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
                value = torch.Tensor([value]).to(self.device)




                self.memory.push((state_rotated,recorded_unrotated_state,recorded_occ,value))  # state_rotated --> same as hat we had (which changan used to use)
                # recorded_urotated_state --> history of(at most 3 step) states before rotatation (I added)
                # value --> same as what we had before



            # recorded occ ra vbaraye inja takmil kon::
            else: #RL phase
                state_unrotated = state   # ! tooye run_k_episode unrotated mifrestim inja
                state_rotated = self.target_policy.rotate(state_unrotated) #dorostesh kardam inja

                #added from here:

                human_xy = state_unrotated[:, 9:11]  # [n-human ,2(x,y)]
                robot_xy = state_unrotated[0:1, 0:2]  # [1 ,2(x,y)]
                joint_xy = torch.cat([robot_xy, human_xy], dim=-2)  # [n-human+1, 2(x,y)]

                history_state.append(state_unrotated)
                if(i==0):
                    history_state = history_state*self.history_max_len
                    history_occ.append(torch.zeros(1, 32))
                    history_occ = history_occ * (self.history_max_len - 1)
                    recorded_occ = torch.zeros(self.history_max_len - 1, 32)
                ii=i+self.history_max_len

                recorded_unrotated_state = torch.stack([history_state[k] for k in range(ii - self.history_max_len , ii)],dim=0)  # [5 2 14] or [history_len n-agent 14]]

                if(i>0):
                    state_tensorized_prev = states[i-1]  #[num-human 14]
                    human_xy_prev = state_tensorized_prev[:, 9:11]  # [  n-human ,2(x,y)]
                    robot_xy_prev = state_tensorized_prev[0:1, 0:2]  # [ 1 ,2(x,y)]
                    joint_xy_prev = torch.cat([robot_xy_prev , human_xy_prev], dim=-2)  # [n-human+1, 2(x,y)]
                    occ_map = self.pool(None ,joint_xy_prev, joint_xy) #None=hidden_state   we dont really need obs1 since we have velocity (or obs1=obs2-v*dt)
                    history_occ.append(occ_map)


                    recorded_occ = torch.stack([history_occ[k] for k in range(ii - self.history_max_len, ii - 1)], dim=0).squeeze()  # [4  32] or [history_len-1 , n*n*2]]
                    # ii          =    [ 0    1    2   3   4     5     6      ]
                    #  history_state = [ s0  s0   s0   s0  s0    s1   s2      ]
                    # history_occ    = [ 0    0    0   0   c01  c12           ]





                # next , _plus:
                mm=i+self.history_max_len
                if i == len(states) - 1:  #
                    # terminal state
                    value = reward
                else:
                    next_state_unrotated = states[i + 1] #[n-human 14]
                    next_state_rotated = self.target_policy.rotate(next_state_unrotated) #[n-human ,13]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)

                    human_xy_next = next_state_unrotated[:, 9:11]  # [  n-human ,2(x,y)]
                    robot_xy_next = next_state_unrotated[0:1, 0:2]  # [ 1 ,2(x,y)]
                    joint_xy_next = torch.cat([robot_xy_next, human_xy_next], dim=-2)  # [n-human+1, 2(x,y)]
                    occ_map_next = self.pool(None, joint_xy, joint_xy_next)  # [1 32]

                    if(i==0):
                        history_state_plus.append(state_unrotated) # when i=0 then we append both state[0] and state[1] afterward
                        history_state_plus = history_state_plus*(self.history_max_len-1) # -1 because we will append state[1] afterward
                        history_occ_plus.append(torch.zeros(1,32))
                        history_occ_plus = history_occ_plus*(self.history_max_len-2) #-2 because we will append c01 afterward

                                                                  #      e.g for history_max_len=5:   history_state_plus=[s0   s0   s0   s0    s1     s2   s3]
                                                                                                 #    history_occ_plus  =[ 0    0    0   occ1  occ2  occ3]
                    history_state_plus.append(next_state_unrotated)
                    history_occ_plus.append(occ_map_next)


                    recorded_unrotated_state_plus = torch.stack([history_state_plus[k] for k in range( mm - self.history_max_len , mm )], dim=0)  # [5 2 14] or [history_len n-agent 14]]
                    recorded_occ_plus = torch.stack( [history_occ_plus[k] for k in range(mm - self.history_max_len , mm-1)], dim=0).squeeze()  # [4  32] or [history_len-1 , n*n*2]] // akharesh range(...,i+1) nist mesle balaee, chon ooc[0] nadarim dar lahze avval

                    human_xy_history_plus = recorded_unrotated_state_plus[ :, :, 9:11]  # [ hist_len, n-human ,2(x,y)]
                    robot_xy_history_plus = recorded_unrotated_state_plus[ :, 0:1, 0:2]  # [hist_len, 1 ,2(x,y)]
                    joint_xy_history_plus = torch.cat([robot_xy_history_plus, human_xy_history_plus], dim=-2).unsqueeze(0)   #[1 ,hist-len, n-human+1,2]


                    representation_env_plus = self.traj_model(joint_xy_history_plus, recorded_occ_plus.unsqueeze(0))[0].data
                    value = reward + gamma_bar * self.target_model(next_state_rotated.unsqueeze(0),representation_env_plus).data.item()

                value = torch.Tensor([value]).to(self.device)

                self.memory.push((state_rotated, recorded_unrotated_state,recorded_occ, value))  # state_rotated --> same as what we had (which changan used to use)
                # recorded_unrotated_state --> history of(at most 3 step) states before rotatation (I added)
                # value --> same as what we had before
                #( [2,13] / [5 2 14] / [4 32] / [1]


            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])



def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0

# def states_to_tensor(states,self):
#     state_tensorized = torch.cat([torch.Tensor([states.self_state + human_state]).to(self.device)  # [num-human 13]
#                                   for human_state in states.human_states], dim=0)
#     return  state_tensorized



def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.bool)
    x[i] = 0
    return x


class Pooling(torch.nn.Module):
    ## Default S-LSTM Parameters
    def __init__(self, cell_side=2.0, n=4, hidden_dim=128, out_dim=None,
                 type_='occupancy', pool_size=8, blur_size=0):
        super(Pooling, self).__init__()
        self.cell_side = cell_side
        self.n = n
        self.type_ = type_
        self.pool_size = pool_size
        self.blur_size = blur_size

        self.pooling_dim = 1
        if self.type_ == 'directional':
            self.pooling_dim = 2
        if self.type_ == 'social':
            self.pooling_dim = hidden_dim

        if out_dim is None:
            out_dim = hidden_dim
        self.out_dim = out_dim

        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(n * n * self.pooling_dim, out_dim),
            torch.nn.ReLU(),
        )

    def forward(self, hidden_state, obs1, obs2):
        if self.type_ == 'occupancy':
            grid = self.occupancies(obs2)
        elif self.type_ == 'directional':
            grid = self.directional(obs1, obs2)
        elif self.type_ == 'social':
            grid = self.social(hidden_state, obs2)
        #return self.embedding(grid)
        return grid

    # def occupancies(self, obs):
    #     n = obs.size(0)
    #     return torch.stack([
    #         self.occupancy(obs[i], obs[one_cold(i, n)])
    #         for i in range(n)
    #     ], dim=0)

    def directional(self, obs1, obs2):
        n = obs2.size(0)
        if n == 1:
            return self.occupancy(obs2[0], None).unsqueeze(0)

        return torch.stack([
            self.occupancy(
                obs2[i],
                obs2[one_cold(i, n)],
                (obs2 - obs1)[one_cold(i, n)] - (obs2 - obs1)[i],
            )
            for i in range(1) #only w.r.t Robot
        ], dim=0)
    #
    # def social(self, hidden_state, obs):
    #     n = obs.size(0)
    #     return torch.stack([
    #         self.occupancy(obs[i], obs[one_cold(i, n)], hidden_state[one_cold(i, n)])
    #         for i in range(n)
    #     ], dim=0)

    def occupancy(self, xy, other_xy, other_values=None):
        """Returns the occupancy."""
        if other_xy is None or \
           xy[0] != xy[0] or \
           other_xy.size(0) == 0:
            return torch.zeros(self.n * self.n * self.pooling_dim, device=xy.device)

        if other_values is None:
            other_values = torch.ones(other_xy.size(0), 1, device=xy.device)

        mask = torch.isnan(other_xy[:, 0]) == 0
        oxy = other_xy[mask]
        other_values = other_values[mask]
        if not oxy.size(0):
            return torch.zeros(self.n * self.n * self.pooling_dim, device=xy.device)

        oij = ((oxy - xy) / (self.cell_side / self.pool_size) + self.n * self.pool_size / 2)
        range_violations = torch.sum((oij < 0) + (oij >= self.n * self.pool_size), dim=1)
        range_mask = range_violations == 0
        oij = oij[range_mask].long()
        other_values = other_values[range_mask]
        if oij.size(0) == 0:
            return torch.zeros(self.n * self.n * self.pooling_dim, device=xy.device)
        oi = oij[:, 0] * self.n * self.pool_size + oij[:, 1]

        # slow implementation of occupancy
        # occ = torch.zeros(self.n * self.n, self.pooling_dim, device=xy.device)
        # for oii, v in zip(oi, other_values):
        #     occ[oii, :] += v

        # faster occupancy
        occ = torch.zeros(self.n**2 * self.pool_size**2, self.pooling_dim, device=xy.device)
        occ[oi] = other_values
        occ = torch.transpose(occ, 0, 1)
        occ_2d = occ.view(1, -1, self.n * self.pool_size, self.n * self.pool_size)

        # optional, blurring (avg with stride 1) has similar effect to bilinear interpolation
        if self.blur_size:
            occ_blurred = torch.nn.functional.avg_pool2d(
                occ_2d, self.blur_size, 1, int(self.blur_size / 2), count_include_pad=True)
        else:
            occ_blurred = occ_2d

        occ_summed = torch.nn.functional.lp_pool2d(occ_blurred, 1, self.pool_size)
        # occ_summed = torch.nn.functional.avg_pool2d(occ_blurred, self.pool_size)  # faster?

        return occ_summed.view(-1)
