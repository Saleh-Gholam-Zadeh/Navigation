

import argparse
import datetime
import logging
import time
import random
import os 
import numpy as np
import torch
import trajnettools



import sys
RL_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
path1 =os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'trajnet/base/scripts')  #necessary for below imports

print(path1)
sys.path.append(path1)#necessary to import below packages


#from .. import augmentation
import augmentation


#from .loss import PredictionLoss, L2Loss
from  loss import PredictionLoss, L2Loss


from lstm import LSTM, LSTMPredictor#, drop_distant


from pooling import Pooling, HiddenStateMLPPooling


VERSION = '0.1.0'

print(os.getcwd())
# from .. import augmentation
# from .loss import PredictionLoss, L2Loss
# from .lstm import LSTM, LSTMPredictor, drop_distant
# from .pooling import Pooling, HiddenStateMLPPooling
# from .. import __version__ as VERSION



class Trainer(object):
    def __init__(self, model=None, criterion=None, optimizer=None, lr_scheduler=None,
                 device=None, loss='L2'):
        self.model = model if model is not None else LSTM()
        if loss == 'L2':
            self.criterion = L2Loss()        
        else:
            self.criterion = PredictionLoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = (lr_scheduler
                             if lr_scheduler is not None
                             else torch.optim.lr_scheduler.StepLR(self.optimizer, 15))

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)

        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)

    def loop(self, train_scenes, val_scenes, out, epochs=35, start_epoch=0):
        return self.val(val_scenes)


    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


    def val(self, scene):  #scene =[]

        eval_start = time.time()
        #self.model.train()  # so that it does not return positions but still normals
        self.model.eval()

        #scene = drop_distant(scene) # scene in trajnet =[21 9 2]      #[9 6 2] from RL
        scene = torch.Tensor(scene).to(self.device)
        representation= self.val_batch(scene)
        return representation


    def val_batch(self, xy): # xy = [21 7 2]
        #observed = xy[:9]  # [9 7 2]
        observed = xy
       # prediction_truth = xy[9:-1].clone()  ## CLONE

        with torch.no_grad():
            #hidden_cell_state = self.model(observed, prediction_truth)
            hidden_cell_state = self.model(observed)

        return hidden_cell_state[0]    #hidden_cell_state[0]=h  , hidden_cell_state[1] =c


def ali (input ,traj_model):  #input = [9 6 2]
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', default=epochs, type=int,
    #                     help='number of epochs')
    # parser.add_argument('--lr', default=1e-3, type=float,
    #                     help='initial learning rate')
    # parser.add_argument('--type', default=type,
    #                     choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp'),
    #                     help='type of LSTM to train')
    # parser.add_argument('-o', '--output', default=None,
    #                     help='output file')
    # parser.add_argument('--disable-cuda', action='store_true',
    #                     help='disable CUDA')
    # parser.add_argument('--path', default='trajdata',
    #                     help='glob expression for data files')
    # parser.add_argument('--loss', default='L2',
    #                     help='loss function')
    #
    # pretrain = parser.add_argument_group('pretraining')
    # pretrain.add_argument('--load-state', default=os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),'trajnet/base/OUTPUT_BLOCK/trajdata/directional.pkl.state'),
    #                       help='load a pickled model state dictionary before training')
    # pretrain.add_argument('--load-full-state', default=None,
    #                       help='load a pickled full state dictionary before training')
    # pretrain.add_argument('--nonstrict-load-state', default=None,
    #                       help='load a pickled state dictionary before training')
    #
    # hyperparameters = parser.add_argument_group('hyperparameters')
    # hyperparameters.add_argument('--hidden-dim', type=int, default=128,
    #                              help='RNN hidden dimension')
    # hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
    #                              help='coordinate embedding dimension')
    # hyperparameters.add_argument('--cell_side', type=float, default=2.0,
    #                              help='cell size of real world')
    # args = parser.parse_args()
    #
    # # set model output file
    # timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    # # if args.output is None:
    # #     args.output = 'output/{}_lstm_{}.pkl'.format(args.type, timestamp)
    # if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
    #     os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    # # if args.output:
    # #     args.output = 'OUTPUT_BLOCK/{}/{}_{}.pkl'.format(args.path, args.type, args.output)
    # # else:
    # #     args.output = 'OUTPUT_BLOCK/{}/{}.pkl'.format(args.path, args.type)
    # output = 'OUTPUT_BLOCK/{}/{}.pkl'.format(args.path, args.type)
    # #
    # # configure logging
    # from pythonjsonlogger import jsonlogger
    # import socket
    # import sys
    # if args.load_full_state:
    #     file_handler = logging.FileHandler(args.output + '.log', mode='a')
    # else:
    #     file_handler = logging.FileHandler(args.output + '.log', mode='w')
    # file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    # stdout_handler = logging.StreamHandler(sys.stdout)
    # logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    # logging.info({
    #     'type': 'process',
    #     'argv': sys.argv,
    #     'args': vars(args),
    #     'version': VERSION,
    #     'hostname': socket.gethostname(),
    # })
    #
    # # refactor args for --load-state
    # # args.load_state_strict = True
    # # if args.nonstrict_load_state:
    # #     args.load_state = args.nonstrict_load_state
    # #     args.load_state_strict = False
    # # if args.load_full_state:
    # #     args.load_state = args.load_full_state   #inja addressi ke behesh dadim ra migire baraye load kardane model
    #
    # # add args.device
    # args.device = torch.device('cpu')
    # # if not args.disable_cuda and torch.cuda.is_available():
    # #     args.device = torch.device('cuda')
    #
    # # read in datasets
    # args.path = 'DATA_BLOCK/' + args.path

    # train_scenes = list(trajnettools.load_all(args.path + '/train/**/*.ndjson'))
    # val_scenes = list(trajnettools.load_all(args.path + '/val/**/*.ndjson'))

    # added by saleh
   # import pickle
    #
    # with open("debug_scenes_biwi.txt", "wb") as fp:  # Pickling
    #     pickle.dump([train_scenes, val_scenes], fp)
    #
    # sys.exit(227)

    # path_file = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'trajnet/base/scripts/lstm/debug_scenes_biwi.txt')
    #
    # with open(path_file,'rb') as f:
    #     temp = pickle.load(f)
    # train_scenes = temp[0]
    # val_scenes  = temp[1]

    # create model
    # pool = None
    # if args.type == 'hiddenstatemlp':
    #     pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim)
    # elif args.type != 'vanilla':
    #     pool = Pooling(type_=args.type, hidden_dim=args.hidden_dim, cell_side=args.cell_side)
    # model = LSTM(pool=pool,
    #              embedding_dim=args.coordinate_embedding_dim,
    #              hidden_dim=args.hidden_dim)
    # # Default Load
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # lr_scheduler = None
    start_epoch = 0

    # train
    # if args.load_state:
    #     # load pretrained model.
    #     # useful for tranfer learning
    #     with open(args.load_state, 'rb') as f:
    #         checkpoint = torch.load(f)
    #     pretrained_state_dict = checkpoint['state_dict']
    #     model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

        # if args.load_full_state:
        # # load optimizers from last training
        # # useful to continue training
        #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
        #     lr_scheduler.load_state_dict(checkpoint['scheduler'])
        #     start_epoch = checkpoint['epoch']


    # trainer

    train_scenes =None
    trainer = Trainer(traj_model, optimizer=None, lr_scheduler=None, device=None, loss=None)
    Encoded_representation=trainer.loop(train_scenes, input, './', epochs=1, start_epoch=start_epoch)  # [128]
    return Encoded_representation

def main():
    a=1
    print('hello')


if __name__ == '__main__':

    main()
