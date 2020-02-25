import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL

import time


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__() #input_dim=13
        self.self_state_dim = self_state_dim  #6
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = 128 + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state,env_rep): #state =[100 n-human 13]
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        t0 = time.clock()
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]

        #
        # mlp1_output = self.mlp1(state.view((-1, size[2])))
        # mlp2_output = self.mlp2(mlp1_output)
        #
        # if self.with_global_state:
        #     # compute attention scores
        #     global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
        #     global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
        #         contiguous().view(-1, self.global_state_dim)
        #     attention_input = torch.cat([mlp1_output, global_state], dim=1)
        # else:
        #     attention_input = mlp1_output
        # scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        #
        # # masked softmax
        # # weights = softmax(scores, dim=1).unsqueeze(2)
        # scores_exp = torch.exp(scores) * (scores != 0).float()
        # weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        # self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        #
        # # output feature is a linear combination of input features
        # features = mlp2_output.view(size[0], size[1], -1)
        # # for converting to onnx
        # # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        # weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        #
        # # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, env_rep], dim=1)  #[100,6] + [100,128]
        value = self.mlp3(joint_state)

        #print('forward_time_sarl:',time.clock()-t0)
        return value


class SARL_modified(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config,traj_model): # traj_model is added by saleh
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.traj_model = traj_model #added by saleh
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

        self.pool = Pooling(type_='directional', hidden_dim=128, cell_side=2)

    def get_attention_weights(self):
        return self.model.attention_weights





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
