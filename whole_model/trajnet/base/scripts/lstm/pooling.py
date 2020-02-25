from collections import defaultdict

import torch
import time

def one_cold(i, n):
    """Inverse one-hot encoding."""
    x = torch.ones(n, dtype=torch.bool)
    x[i] = 0
    return x


class HiddenStateMLPPooling(torch.nn.Module):
    def __init__(self, hidden_dim=128, mlp_dim=128, mlp_dim_spatial=16, out_dim=None):
        super(HiddenStateMLPPooling, self).__init__()
        self.out_dim = out_dim or hidden_dim
        self.spatial_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, mlp_dim_spatial),
            torch.nn.ReLU(),
        )
        self.hidden_embedding = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, mlp_dim - mlp_dim_spatial),
            torch.nn.ReLU(),
        )
        self.out_projection = torch.nn.Linear(mlp_dim, self.out_dim)

    @staticmethod
    def rel_obs(obs):
        unfolded = obs.unsqueeze(0).repeat(obs.size(0), 1, 1)
        relative = unfolded - obs.unsqueeze(1)
        return relative

    def forward(self, hidden_states, _, obs):
        rel_obs = self.rel_obs(obs)
        spatial = self.spatial_embedding(rel_obs)
        hidden = self.hidden_embedding(hidden_states)
        hidden_unfolded = hidden.unsqueeze(0).repeat(hidden.size(0), 1, 1)
        embedded = torch.cat([spatial, hidden_unfolded], dim=2)
        pooled, _ = torch.max(embedded, dim=1)
        return self.out_projection(pooled)


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

    def forward(self, hidden_state, obs1, obs2,occ):
        # if self.type_ == 'occupancy':
        #     grid = self.occupancies(obs2)
        # elif self.type_ == 'directional':
        #     grid = self.directional(obs1, obs2)
        # elif self.type_ == 'social':
        #     grid = self.social(hidden_state, obs2)
        #return self.embedding(grid.squeeze())  # check kardam occ ba grid yeki bood !
        return self.embedding(occ)   #self.embeddinf 32-->128    occ=[100 32] ---> ans=[100 128]

    def occupancies(self, obs):
        n = obs.size(0)
        return torch.stack([
            self.occupancy(obs[i], obs[one_cold(i, n)])
            for i in range(n)
        ], dim=0)

    def directional(self, obs1, obs2):  #obs1,2 = [n-agents ,2]
        n = obs2.size(-2)
        if n == 1:
            return self.occupancy(obs2[0], None).unsqueeze(0)

        return torch.stack([ #  ":" is  used because we need these operations for all batches
            self.occupancy(
                obs2[:,i:i+1,:],
                obs2[:,one_cold(i, n),:],
                (obs2 - obs1)[:,one_cold(i, n),:] - (obs2 - obs1)[:,i:i+1,:],  # i:i+1  eq  0:1 which is 0
            )
            for i in range(1)
        ], dim=0)

    def social(self, hidden_state, obs):
        n = obs.size(0)
        return torch.stack([
            self.occupancy(obs[i], obs[one_cold(i, n)], hidden_state[one_cold(i, n)])
            for i in range(n)
        ], dim=0)

    def occupancy(self, xy, other_xy, other_values=None):  #[?batch? timesteps n-agent 2(x,y)]
        """Returns the occupancy."""
        # if other_xy is None or \
        #         xy[0] != xy[0] or \
        #         other_xy.size(0) == 0:
        #     return torch.zeros(self.n * self.n * self.pooling_dim, device=xy.device)
        #
        # if other_values is None:
        #     other_values = torch.ones(other_xy.size(0), 1, device=xy.device)

        t0 = time.clock()
        num_batches = other_xy.size(0)
        occ_summed = torch.zeros(num_batches,self.pooling_dim,self.n,self.n)

        for batch_num in range(xy.size(0)):
            mask = torch.isnan(other_xy[batch_num,:, 0]) == 0
            oxy = other_xy[batch_num,mask]
            other_value = other_values[batch_num,mask]
            if not oxy.size(0):
                print('oxy.size(0)=0')

                return torch.zeros(self.n * self.n * self.pooling_dim, device=xy.device)


            oij = (oxy - xy[batch_num]) / (self.cell_side / self.pool_size) + self.n * self.pool_size / 2
            range_violations = torch.sum((oij < 0) + (oij >= self.n * self.pool_size), dim=1)
            range_mask = range_violations == 0
            oij = oij[range_mask].long() #how to do it paralel???
            other_value = other_value[range_mask]
            if oij.size(0) == 0:
                occ_summed[batch_num,:,:,:]= torch.zeros(self.pooling_dim, self.n , self.n , device=xy.device)
                continue
            oi = oij[:, 0] * self.n * self.pool_size + oij[:, 1]

            # slow implementation of occupancy
            # occ = torch.zeros(self.n * self.n, self.pooling_dim, device=xy.device)
            # for oii, v in zip(oi, other_values):
            #     occ[oii, :] += v

            # faster occupancy
            occ = torch.zeros(self.n ** 2 * self.pool_size ** 2, self.pooling_dim, device=xy.device)
            occ[oi] = other_value
            occ = torch.transpose(occ, 0, 1)
            occ_2d = occ.view(1, -1, self.n * self.pool_size, self.n * self.pool_size)

            # optional, blurring (avg with stride 1) has similar effect to bilinear interpolation
            if self.blur_size: #false
                occ_blurred = torch.nn.functional.avg_pool2d(
                    occ_2d, self.blur_size, 1, int(self.blur_size / 2), count_include_pad=True)
            else: #true
                occ_blurred = occ_2d

            occ_summed[batch_num,:,:,:] = torch.nn.functional.lp_pool2d(occ_blurred, 1, self.pool_size)
            # occ_summed = torch.nn.functional.avg_pool2d(occ_blurred, self.pool_size)  # faster?
        print('occupancy time with for loop:',time.clock()-t0)
        return occ_summed.view(num_batches,-1)   #[100,32]
