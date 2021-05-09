import torch
import torch.nn as nn
import torch.nn.functional as F
import dataloader
import torch.optim as optim
from abc import ABCMeta, abstractmethod
from utils import Accuracy
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utils import matplotlib_imshow
import utils
from pygnn import GNN
import os


class GNNWrapper:
    class Config:
        def __init__(self):
            self.device = None
            self.use_cuda = None
            self.dataset_path = None
            self.log_interval = None
            self.tensorboard = None
            self.task_type = None

            # hyperparams
            self.lrw = None
            self.loss_f = None
            self.epochs = None
            self.convergence_threshold = None
            self.max_iterations = None
            self.n_nodes = None
            self.state_dim = None
            self.label_dim = None
            self.output_dim = None
            self.graph_based = False
            self.activation = torch.nn.Tanh()
            self.state_transition_hidden_dims = None
            self.output_function_hidden_dims = None
            self.task_type = "semisupervised"

            self.state_net = None #
            self.dset_name = None
            self.aggregation_type = None
            # optional
            # self.loss_w = 1.
            # self.energy_weight = 0.
            # self.l2_weight = 0.

    def __init__(self, config: Config):
        self.config = config

        # to be populated
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.test_loader = None

        self.state_net = config.state_net
        if self.config.tensorboard:
            #self.writer = SummaryWriter('logs/tensorboard')
            # self.writer = SummaryWriter(comment=f'/{str(self.state_net)[0:15]}')
            self.writer = SummaryWriter(log_dir=os.path.join("logs",
                                                             f"lrw__{self.config.lrw}_state__{self.config.state_dim}",
                                                             f"{str(self.state_net).split('(')[0]}"))

        self.first_flag_writer = True

        # #plotting
        # self.arr_its_train = []
        # self.arr_acc_train = []
        # self.arr_its_test = []
        # self.arr_acc_test = []
        # #/plotting
                                                             
        # Write logs into a file:
        self.file_logs = open(f"txtlogs.txt", "a")
        self.file_logs.write(
            f" *#*#*#*#*# Logs for: Dataset:{self.config.dset_name}, learning_rate: {self.config.lrw}, state_dim:{self.config.state_dim}, aggregation function:{str(self.state_net).split('(')[0]} , aggregation type:{self.config.aggregation_type}  *#*#*#*#*#  \n")
        self.file_logs.close()

    def __call__(self, dset, state_net=None, out_net=None):
        # handle the dataset info
        self._data_loader(dset)
        self.gnn = GNN(self.config, state_net, out_net).to(self.config.device)
        self._criterion()
        self._optimizer()
        self._accuracy()

    def _data_loader(self, dset):  # handle dataset data and metadata
        self.dset = dset.to(self.config.device)
        self.config.label_dim = self.dset.node_label_dim
        self.config.n_nodes = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _optimizer(self):
        # for name, param in self.gnn.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # exit()
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=self.config.lrw)
        #self.optimizer = optim.SGD(self.gnn.parameters(), lr=self.config.lrw)

    def _criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type=self.config.task_type)
        self.ValidAccuracy = Accuracy(type=self.config.task_type)
        self.TestAccuracy = Accuracy(type=self.config.task_type)

    def train_step(self, epoch):
        self.gnn.train()
        data = self.dset
        self.optimizer.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
        # loss computation - semisupervised
        loss = self.criterion(output, data.targets)

        loss.backward()

        self.optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets)
            accuracy_train = self.TrainAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    'Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {}'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))

                self.file_logs = open(f"txtlogs.txt", "a")
                self.file_logs.write('Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {} \n'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))
                self.file_logs.close()

                # #plotting
                # self.arr_its_train.append(epoch)
                # self.arr_acc_train.append(accuracy_train)
                # #/plotting

                if self.config.tensorboard:
                    self.writer.add_scalar(
                        f'Training Accuracy_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        accuracy_train,
                        epoch)

                    self.writer.add_scalar(
                        f'Training Loss_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        loss,
                        epoch)
                    self.writer.add_scalar(
                        f'Training Iterations_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        iterations,
                        epoch)

                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
            return accuracy_train  #
        # self.TrainAccuracy.reset()
            
            

    def predict(self, edges, agg_matrix, node_labels):
        return self.gnn(edges, agg_matrix, node_labels)

    def test_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        self.TestAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output, data.targets)

            self.TestAccuracy.update(output, data.targets)
            acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Test set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_test, self.TestAccuracy.get_best()))

                self.file_logs = open(f"txtlogs.txt", "a")
                self.file_logs.write('Test set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%) \n'.format(
                        test_loss, acc_test, self.TestAccuracy.get_best()))
                self.file_logs.close()

                # #plotting
                # self.arr_its_test.append(epoch)
                # self.arr_acc_test.append(acc_test)
                # #/plotting

                if self.config.tensorboard:
                    self.writer.add_scalar(
                        f'Test Accuracy_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        acc_test,
                        epoch)
                    self.writer.add_scalar(
                        f'Test Loss_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        test_loss,
                        epoch)
                    self.writer.add_scalar(
                        f'Test Iterations_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        iterations,
                        epoch)
            
                return acc_test

    def valid_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        self.ValidAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output, data.targets)

            self.ValidAccuracy.update(output, data.targets)
            acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_valid, self.ValidAccuracy.get_best()))

                self.file_logs = open(f"txtlogs.txt", "a")
                self.file_logs.write('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%) \n'.format(
                        test_loss, acc_valid, self.ValidAccuracy.get_best()))
                self.file_logs.close()

            
                if self.config.tensorboard:
                    self.writer.add_scalar(
                        f'Valid Accuracy_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        acc_valid,
                        epoch)
                    self.writer.add_scalar(
                        f'Valid Loss_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        test_loss,
                        epoch)
                    self.writer.add_scalar(
                        f'Valid Iterations_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        iterations,
                        epoch)

                return acc_valid

class SemiSupGNNWrapper(GNNWrapper):
    class Config:
        def __init__(self):
            self.device = None
            self.use_cuda = None
            self.dataset_path = None
            self.log_interval = None
            self.tensorboard = None
            self.task_type = None

            # hyperparams
            self.lrw = None
            self.loss_f = None
            self.epochs = None
            self.convergence_threshold = None
            self.max_iterations = None
            self.n_nodes = None
            self.state_dim = None
            self.label_dim = None
            self.output_dim = None
            self.graph_based = False
            self.activation = torch.nn.Tanh()
            self.state_transition_hidden_dims = None
            self.output_function_hidden_dims = None

            self.state_net = None
            self.dset_name = None
            self.aggregation_type = None
            # optional
            # self.loss_w = 1.
            # self.energy_weight = 0.
            # self.l2_weight = 0.

    def __init__(self, config: Config):
        super().__init__(config)

    def _data_loader(self, dset):  # handle dataset data and metadata
        self.dset = dset.to(self.config.device)
        self.config.label_dim = self.dset.node_label_dim
        self.config.n_nodes = self.dset.num_nodes
        self.config.output_dim = self.dset.num_classes

    def _accuracy(self):
        self.TrainAccuracy = Accuracy(type="semisupervised")
        self.ValidAccuracy = Accuracy(type="semisupervised")
        self.TestAccuracy = Accuracy(type="semisupervised")

    def train_step(self, epoch):
        self.gnn.train()
        data = self.dset
        self.optimizer.zero_grad()
        self.TrainAccuracy.reset()
        # output computation
        output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
        # loss computation - semisupervised
        loss = self.criterion(output[data.idx_train], data.targets[data.idx_train])

        loss.backward()

        # with torch.no_grad():
        #     for name, param in self.gnn.named_parameters():
        #         if "state_transition_function" in name:
        #             #self.writer.add_histogram("gradient " + name, param.grad, epoch)
        #             param.grad = 0*  param.grad



        self.optimizer.step()

        # # updating accuracy
        # batch_acc = self.TrainAccuracy.update((output, target), batch_compute=True)
        with torch.no_grad():  # Accuracy computation
            # accuracy_train = torch.mean(
            #     (torch.argmax(output[data.idx_train], dim=-1) == data.targets[data.idx_train]).float())
            self.TrainAccuracy.update(output, data.targets, idx=data.idx_train)
            accuracy_train = self.TrainAccuracy.compute()

            if epoch % self.config.log_interval == 0:
                print(
                    'Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {}'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))

                self.file_logs = open(f"txtlogs.txt", "a")
                self.file_logs.write('Train Epoch: {} \t Mean Loss: {:.6f}\tAccuracy Full Batch: {:.6f} \t  Best Accuracy : {:.6f}  \t Iterations: {} \n'.format(
                        epoch, loss, accuracy_train, self.TrainAccuracy.get_best(), iterations))
                self.file_logs.close()

                if self.config.tensorboard:
                    self.writer.add_scalar(
                        f'Training Accuracy_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        accuracy_train,
                        epoch)
                    self.writer.add_scalar(
                        f'Training Loss_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        loss,
                        epoch)
                    self.writer.add_scalar(
                        f'Training Iterations_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        iterations,
                        epoch)
                    for name, param in self.gnn.named_parameters():
                        self.writer.add_histogram(name, param, epoch)
                        self.writer.add_histogram("gradient " + name, param.grad, epoch)
        # self.TrainAccuracy.reset()
        return accuracy_train  #

    def predict(self, edges, agg_matrix, node_labels):
        return self.gnn(edges, agg_matrix, node_labels)

    def test_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        self.TestAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output[data.idx_test], data.targets[data.idx_test])

            self.TestAccuracy.update(output, data.targets, idx=data.idx_test)
            acc_test = self.TestAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Test set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_test, self.TestAccuracy.get_best()))

                self.file_logs = open(f"txtlogs.txt", "a")
                self.file_logs.write('Test set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%) \n'.format(
                        test_loss, acc_test, self.TestAccuracy.get_best()))
                self.file_logs.close()

                if self.config.tensorboard:
                    self.writer.add_scalar(
                        f'Test Accuracy_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        acc_test,
                        epoch)
                    self.writer.add_scalar(
                        f'Test Loss_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        test_loss,
                        epoch)
                    self.writer.add_scalar(
                        f'Test Iterations_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        iterations,
                        epoch)
            return acc_test

    def valid_step(self, epoch):
        ####  TEST
        self.gnn.eval()
        data = self.dset
        self.ValidAccuracy.reset()
        with torch.no_grad():
            output, iterations = self.gnn(data.edges, data.agg_matrix, data.node_labels)
            test_loss = self.criterion(output[data.idx_valid], data.targets[data.idx_valid])

            self.ValidAccuracy.update(output, data.targets, idx=data.idx_valid)
            acc_valid = self.ValidAccuracy.compute()
            # acc_test = torch.mean(
            #     (torch.argmax(output[data.idx_test], dim=-1) == data.targets[data.idx_test]).float())

            if epoch % self.config.log_interval == 0:
                print('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%)'.format(
                    test_loss, acc_valid, self.ValidAccuracy.get_best()))

            
                self.file_logs = open(f"txtlogs.txt", "a")
                self.file_logs.write('Valid set: Average loss: {:.4f}, Accuracy:  ({:.4f}%) , Best Accuracy:  ({:.4f}%) \n'.format(
                        test_loss, acc_valid, self.ValidAccuracy.get_best()))
                self.file_logs.close()
            
                if self.config.tensorboard:
                    self.writer.add_scalar(
                        f'Valid Accuracy_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        acc_valid,
                        epoch)
                    self.writer.add_scalar(
                        f'Valid Loss_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        test_loss,
                        epoch)
                    self.writer.add_scalar(
                        f'Valid Iterations_{self.config.dset_name}/aggregation_{self.config.aggregation_type}',
                        iterations,
                        epoch)
            return acc_valid
