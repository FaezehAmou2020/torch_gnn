import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import utils
import dataloader

from gnn_wrapper import GNNWrapper
import net
from itertools import product
import time

#
# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_dev', type=int, default=0,
                        help='select specific CUDA device for training')
    parser.add_argument('--n_gpu_use', type=int, default=1,
                        help='select number of CUDA device for training')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='logging training status cadency')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='For logging the model in tensorboard')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        args.n_gpu_use = 0

    device = utils.prepare_device(n_gpu_use=args.n_gpu_use, gpu_id=args.cuda_dev)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # torch.manual_seed(args.seed)
    # # fix random seeds for reproducibility
    # SEED = 123
    # torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(SEED)

    # configugations
    cfg = GNNWrapper.Config()
    cfg.use_cuda = use_cuda
    cfg.device = device

    cfg.log_interval = args.log_interval
    cfg.tensorboard = args.tensorboard

    # cfg.batch_size = args.batch_size
    # cfg.test_batch_size = args.test_batch_size
    # cfg.momentum = args.momentum

    cfg.dataset_path = './data'
    cfg.epochs = args.epochs
    cfg.lrw = args.lr
    cfg.activation = nn.Sigmoid()
    cfg.state_transition_hidden_dims = [10, ]
    cfg.output_function_hidden_dims = [ 5]
    cfg.state_dim = 10 #
    cfg.max_iterations = 50
    cfg.convergence_threshold = 0.01
    cfg.graph_based = False
    cfg.log_interval = 10
    cfg.lrw = 0.01
    cfg.task_type = "multiclass"

    # model creation
    # model_tr = GNNWrapper(cfg)
    # model_val = GNNWrapper(cfg)
    # model_tst = GNNWrapper(cfg)

    cfg.dset_name = "sub_30_15_200"
    cfg.aggregation_type = "degreenorm"
    # dataset creation
    dset = dataloader.get_subgraph(set=cfg.dset_name, aggregation_type=cfg.aggregation_type, sparse_matrix=True)  # generate the dataset

    cfg.label_dim = dset["train"].node_label_dim

    state_nets = [
                    net.StateTransition(cfg.state_dim, cfg.label_dim,
                            mlp_hidden_dim=cfg.state_transition_hidden_dims,
                            activation_function=cfg.activation),
                    net.GINTransition(cfg.state_dim, cfg.label_dim,
                            mlp_hidden_dim=cfg.state_transition_hidden_dims,
                            activation_function=cfg.activation),
                    net.GINPreTransition(cfg.state_dim, cfg.label_dim,
                            mlp_hidden_dim=cfg.state_transition_hidden_dims,
                            activation_function=cfg.activation)
                  ]

    lrs = [0.05, 0.01, 0.001]

    hyperparameters = dict(lr=lrs, state_net=state_nets)
    hyperparameters_values = [v for v in hyperparameters.values()]

    start_0 = time.time()
    for lr, state_net in product(*hyperparameters_values):
        cfg.lrw = lr
        cfg.state_net = state_net

        print(f"learning_rate:{lr}, state_dim:{cfg.state_dim}, aggregation function:{str(state_net).split('(')[0]} ")
        # model creation
        model_tr = GNNWrapper(cfg)
        model_val = GNNWrapper(cfg)
        model_tst = GNNWrapper(cfg)
              
        # 24.3.21 STOPPER
        early_stopper = utils.EarlyStopper(cfg)


        model_tr(dset["train"], state_net=state_net)  # dataset initalization into the GNN
        model_val(dset["validation"], state_net=model_tr.gnn.state_transition_function,
                    out_net=model_tr.gnn.output_function)  # dataset initalization into the GNN
        model_tst(dset["test"], state_net=model_tr.gnn.state_transition_function,
                    out_net=model_tr.gnn.output_function)  # dataset initalization into the GNN
        # training code
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            acc_train = model_tr.train_step(epoch)
            if epoch % 10 == 0:
                acc_tst = model_tst.test_step(epoch)
                acc_val = model_val.valid_step(epoch)
                stp = early_stopper(acc_train, acc_val, acc_tst, epoch)
                
                # return -1 keeps training the model!
                if stp == -1:
                    print(f"{early_stopper.best_epoch}, \t {early_stopper.best_train}, \t, {early_stopper.best_val}, \t {early_stopper.best_test}")
                    break
                # model_tst.test_step(epoch)

        time_sample = time.time() - start
        print(f"time taken for one set: {str(time_sample)} seconds")

    time_whole = time.time() - start_0
    print(f"time taken for the whole experiment: {str(time_whole)} seconds")
    # if args.save_model:
    #     torch.save(model.gnn.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
