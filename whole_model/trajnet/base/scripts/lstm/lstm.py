from collections import defaultdict
import itertools

import numpy as np
import torch
import trajnettools

#from .modules import Hidden2Normal, InputEmbedding
from modules import Hidden2Normal, InputEmbedding
import time

NAN = float('nan')

n_track = 0


def drop_distant(xy, r=10.0):
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    if not all(any(e == e for e in column) for column in distance_2.T):
        print(distance_2.tolist())
        print(np.nanmin(distance_2, axis=0))
        raise Exception
    mask = np.nanmin(distance_2, axis=0) < r ** 2
    return xy[:, mask]


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.pool_to_input = pool_to_input

        self.input_embedding = InputEmbedding(2, self.embedding_dim, 4.0)
        if self.pool is not None and self.pool_to_input:
            self.input_embedding = InputEmbedding(2 + self.pool.out_dim, self.embedding_dim, 4.0)

        self.encoder = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

    def step(self, lstm, hidden_cell_state, obs1, obs2,occ): #obs1,obs2=[100 n-human 2(xy)]
        """Do one step: two inputs to one normal prediction."""
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene  -->saleh: always present for my data
        #track_mask = (torch.isnan(obs1[:,:, 0]) + torch.isnan(obs2[:,:, 0])) == 0  in my dataset mask is always true
        #obs1, obs2 = obs1[track_mask], obs2[track_mask]
        t0=time.clock()
        hidden_cell_stacked = [                             # h,c =[100 128],[100 128]  // tooye normal-trajnet h,c =[n-1gent 128]
            torch.stack([h for  h in hidden_cell_state[0] ], dim=0),
            torch.stack([c for  c in hidden_cell_state[1] ], dim=0),
        ]
        #print("stacking-time=" , time.clock()-t0)
        # input embedding and optional pooling
        if self.pool is None:
            input_emb = self.input_embedding(obs2 - obs1)
        elif self.pool_to_input:#true
            hidden_states_to_pool = hidden_cell_stacked[0].detach() #dar kol faghat baraye social lstm estefade mishe na directional//ans=h = [100  128]  //tooye trajnet normal  h ra migire ke [n-agent 128] hast sizesh// dar kol mohem nist faghat baraye social lstm
            pooled = self.pool(hidden_states_to_pool, obs1, obs2,occ)  #0cc=[100 32] , pooled [100 128] as expected
            input_emb = self.input_embedding(torch.cat([(obs2 - obs1)[:,0], pooled], dim=-1))#self.input_embedding (130--->62+2)
        else:                                           #[100  2(x,y]    , [100 128] // concat=[100 130] --->ans=[100 64]
            input_emb = self.input_embedding(obs2 - obs1)
            hidden_states_to_pool = hidden_cell_stacked[0].detach()
            hidden_cell_stacked[0] += self.pool(hidden_states_to_pool, obs1, obs2)

        # step
        hidden_cell_stacked = lstm(input_emb, [hidden_cell_stacked[0] ,hidden_cell_stacked[1]]) # lstm = LSTMcell
                                    #( [100 64] , [h=[100 128],c=[100 128]])                  // toye normal-trajnet = ([n-agent 64], [h=[n-agent 128],c=[n-agent 128])

        # unmask
        #inja moshkel dare::
        #kollan i bayad hazf beshe

        #mask_index = [i for i in range(obs1.size(-2)) ] chon inja afrad ra hazf kardim va faghat baraye robot mohasebat ra anjam midim pas dige mask_index nemikhaym
        # for  h, c in zip(
        #                       hidden_cell_stacked[0],
        #                       hidden_cell_stacked[1],
        #                       ):
        #     hidden_cell_state[0] = h
        #     hidden_cell_state[1] = c
        #
        # return hidden_cell_state
        #print(time.clock()-t0,"step-time")
        return list([hidden_cell_stacked[0],hidden_cell_stacked[1]])

    def tag_step(self, lstm, hidden_cell_state, tag):
        """Update step for all LSTMs with a start tag."""
        hidden_cell_state = (
            hidden_cell_state[0][:,0,:],  # h of robot [100 128]
            hidden_cell_state[1][:,0,:],   # C of robot [100  128]
        )
        hidden_cell_state = lstm(tag[:,0,:], hidden_cell_state)   # tag of robot  for all 100 batches
        return (
            list(hidden_cell_state[0]),  # h of robot= [100 128]
            list(hidden_cell_state[1]),  # c of robot= [100 128]
        )

    def forward(self, observed,pooled, prediction_truth=None, n_predict=None): #pooled added by saleh which is occ_map
        """forward

        observed shape is (seq, n_tracks, observables)
        """
        assert len(observed.size())==4
        assert len(pooled.size()) == 3
        # assert ((prediction_truth is None) + (n_predict is None)) == 1
        # if n_predict is not None:
        #     # -1 because one prediction is done by the encoder already
        #     prediction_truth = [None for _ in range(n_predict - 1)]

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        batch_size = observed.size(-4) #observed = [(?n-batch) , time ,n-agents ,2]
        n_tracks = observed.size(-2)
        # hidden_cell_state = (     #[  [7,128] , [7,128]  ]
        #     [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(n_tracks)],
        #     [torch.zeros( self.hidden_dim, device=observed.device) for _ in range(n_tracks)],
        # )

        hidden_cell_state = ( #[  [100 n-human 128] ,[100, n-human,128]   ] //tooye  normal-trajnet  h,c =  [n-human 128], [n-human, 128]
                              torch.zeros(batch_size,n_tracks,self.hidden_dim),
                              torch.zeros(batch_size,n_tracks,self.hidden_dim),
                                )
        # tag the start of encoding (optional)
        start_enc_tag = self.input_embedding.start_enc(observed[:,0]) #observed=[100 hist-len n-human, 2(xy)] # bayad time avval ra begirim inja ke mishe observed[:,0]
        #start_encoding_tag = [100 3 64] tooye normal-trajnet [3 64] bood     [:,:,-2]==1 elsewhere 0
        hidden_cell_state = self.tag_step(self.encoder,           hidden_cell_state,               start_enc_tag)    #  ans: h,c =[100 128] , [100 128] hameye batch ha ra gereftim vali faghat baraye robot//tooye normal-trajnet h,c[n-h 128] boodand ke batch nadasht vali hame human ha boodand
                                        #lstmCell(64,128) ,   [[100 n-h 128],[100 n-h 128]] ,       [100 n-h 64]]

        # encoder
        for j in range(observed.size(-3)-1): # range (timesteps -1)
            ##LSTM Step
            obs1 ,obs2 = observed[:,j,:,:] , observed[:,j+1,:,:]
            hidden_cell_state = self.step(self.encoder, hidden_cell_state, obs1, obs2,pooled[:,j,:]) #pooled added by saleh  pooled =[100(batch) history_len-1 32]

        return hidden_cell_state


class LSTMPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)

    def __call__(self, paths, n_predict=12, modes=1):
        self.model.eval()

        observed_path = paths[0]
        ped_id = observed_path[0].pedestrian
        ped_id_ = []
        for j in range(len(paths)):
            ped_id_.append(paths[j][0].pedestrian)
        frame_diff = observed_path[1].frame - observed_path[0].frame
        first_frame = observed_path[8].frame + frame_diff
        with torch.no_grad():
            xy = trajnettools.Reader.paths_to_xy(paths)
            xy = drop_distant(xy, r=10.0)
            xy = torch.Tensor(xy)  # .to(self.device)
            multimodal_outputs = {}
            for np in range(modes):
                _, output_scenes = self.model(xy[:9], n_predict=n_predict)
                outputs = output_scenes[-n_predict:, 0]
                output_scenes = output_scenes[-n_predict:]
                output_primary = [trajnettools.TrackRow(first_frame + i * frame_diff, ped_id, outputs[i, 0],
                                                        outputs[i, 1], 0) for i in range(len(outputs))]

                output_all = [[trajnettools.TrackRow(first_frame + i * frame_diff, ped_id_[j], output_scenes[i, j, 0],
                                                     output_scenes[i, j, 1], 0) for i in range(len(outputs))] for j in
                              range(1, output_scenes.shape[1])]

                multimodal_outputs[np] = [output_primary, output_all]
        return multimodal_outputs