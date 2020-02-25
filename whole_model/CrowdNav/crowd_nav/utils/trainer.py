import logging
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import time

class Trainer(object):
    def __init__(self, model,traj_model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.traj_model = traj_model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None: #True
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:
                inputs,state_history,occ_history, values = data  #I added history!
                inputs = Variable(inputs)
                values = Variable(values)

                self.optimizer.zero_grad()

                human_xy = state_history[:, :, :, 9:11]  # [100 ,hist_len, n-human ,2(x,y)]
                robot_xy = state_history[:, :, 0:1, 0:2]  # [100 ,hist_len, 1 ,2(x,y)]
                joint_xy = torch.cat([robot_xy, human_xy], dim=-2)


                t0=time.clock()
                #representation_env = torch.stack([self.traj_model(joint_xy_i)[0][0] for joint_xy_i in joint_xy],dim=0)
                representation_env = self.traj_model(joint_xy,occ_history)[0].data# joint_xy=[100, hist_len, n-human+1,2(xy)]  // occ_history=[100,hist_len-1,32] //ans=[100 128]
                #inputs = torch.cat([inputs,representation_env])   #inputs=[100 2 13]
                #print("trajnet forward for "+ str(joint_xy.shape[0]) +  " batch = ", time.clock() - t0)

                t0 = time.clock()
                outputs = self.model(inputs,representation_env)
               # print(" SARL   forward for " + str(inputs.shape[0]) + " batch = ", time.clock() - t0)

                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs,state_history,occ_history, values = next(iter(self.data_loader))
            inputs = Variable(inputs)
            values = Variable(values)

            self.optimizer.zero_grad()

            human_xy = state_history[:, :, :, 9:11]  #  [100 ,hist_len, n-human ,2(x,y)]
            robot_xy = state_history[:, :, 0:1, 0:2]  # [100 ,hist_len, 1 ,2(x,y)]
            joint_xy = torch.cat([robot_xy, human_xy], dim=-2)

            t0 = time.clock()
            representation_env = self.traj_model(joint_xy,occ_history)[0].data
            #print("trajnet forward for "+ str(joint_xy.shape[0]) +  " batch = ", time.clock() - t0)

            t0 = time.clock()
            outputs = self.model(inputs,representation_env)
            #print(" SARL   forward for " + str(inputs.shape[0]) + " batch = ", time.clock() - t0)

            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss
