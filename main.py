import argparse
import os
import sys
from os.path import join as pjoin
import numpy as np
import time
import datetime
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from sklearn.metrics import average_precision_score as avg_precision, roc_auc_score, roc_curve
import pickle
import multiprocessing as mp
from chebygin import *
from graphdata import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments with Graph Neural Networks')
    # Dataset
    parser.add_argument('-D', '--dataset', type=str, default='colors-3',
                        choices=['colors-3', 'colors-4', 'colors-8', 'colors-16', 'colors-32',
                                 'triangles', 'mnist', 'mnist-75sp', 'collab', 'proteins', 'dd'],
                        help='colors-n means the colors dataset with n-dimensional features')
    parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to the dataset')
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='# of the epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training data')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--lr_decay_step', type=str, default='90', help='number of epochs after which to reduce learning rate')
    parser.add_argument('--wdecay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('-f', '--filters', type=str, default='64,64', help='number of filters in each graph layer')
    parser.add_argument('-K', '--filter_scale', type=int, default=2, help='filter scale (receptive field size), must be > 0; 1 for GCN or GIN')
    parser.add_argument('--n_hidden', type=int, default=0, help='number of hidden units inside the graph layer')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'sum'], help='neighbors aggregation inside the graph layer')
    parser.add_argument('--readout', type=str, default='sum', choices=['mean', 'sum', 'max'], help='type of global pooling over all nodes')
    parser.add_argument('--kl_weight', type=float, default=100, help='weight of the KL term in the loss')
    parser.add_argument('--pool', type=str, default=None, help='type of pooling between layers')
    # Auxiliary arguments
    parser.add_argument('--test_batch_size', type=int, default=32, help='batch size for test data')
    parser.add_argument('--log_interval', type=int, default=400, help='print interval')
    parser.add_argument('--results', type=str, default='./results',
                        help='directory to save model checkpoints and other results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='cuda/cpu')
    parser.add_argument('--seed', type=int, default=11, help='seed for shuffling nodes')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader')
    args = parser.parse_args()

    args.lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    args.filters = list(map(int, args.filters.split(',')))

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args


def count_correct(output, target, N_nodes=None, N_nodes_min=0, N_nodes_max=25):
    if output.shape[1] == 1:
        # Regression
        pred = output.round().long()
    else:
        # Classification
        pred = output.max(1, keepdim=True)[1]
    target = target.long().squeeze()
    pred = pred.squeeze()
    if N_nodes is not None:
        idx = (N_nodes >= N_nodes_min) & (N_nodes <= N_nodes_max)
        if idx.sum() > 0:
            correct = pred[idx].eq(target[idx]).sum().item()
            for lbl in torch.unique(target, sorted=True):
                idx_lbl = target[idx] == lbl
                eq = (pred[idx][idx_lbl] == target[idx][idx_lbl]).float()
                print('lbl: {}, avg acc: {:2.2f}% ({}/{})'.format(lbl, 100 * eq.mean(), int(eq.sum()),
                                                                  int(idx_lbl.float().sum())))

            eq = (pred[idx] == target[idx]).float()
            print('{} <= N_nodes <= {} (min={}, max={}), avg acc: {:2.2f}% ({}/{})'.format(N_nodes_min,
                                                                                          N_nodes_max,
                                                                                          N_nodes[idx].min(),
                                                                                          N_nodes[idx].max(),
                                                                                          100 * eq.mean(),
                                                                                                  int(eq.sum()), int(idx.sum())))
        else:
            correct = 0
            print('no graphs with nodes >= {} and <= {}'.format(N_nodes_min, N_nodes_max))
    else:
        correct = pred.eq(target).sum().item()

    return correct


def mse_loss(target, output, reduction='mean'):
    if reduction == 'mean':
        return torch.mean((target.float().squeeze() - output.float().squeeze()) ** 2)
    elif reduction == 'sum':
        return torch.sum((target.float().squeeze() - output.float().squeeze()) ** 2)
    else:
        NotImplementedError(reduction)


def normalize_data(data):
    # TODO:
    return data


def sanity_check(model, data):
    output1, other_losses1 = model(data)
    output2, other_losses2 = model(shuffle_nodes(data))
    assert torch.allclose(output1, output2, rtol=1e-03, atol=1e-05), torch.norm(output1 - output2)
    for l1, l2 in zip(other_losses1, other_losses2):
        assert torch.allclose(l1, l2), (l1, l2)
    print('model is checked for nodes shuffling')


def compute_avg(model, device, train_loader, n_batches=np.Inf):
    print('computing mean and std of input features')
    model.eval()
    x = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            x.append(data[0].view(data[0].size(0), data[0].size(1), -1).data) # B,N,F
            if batch_idx > n_batches:
                break
            del data
    x = torch.cat(x, dim=1)  # M,N,F
    print('data read', x.shape)

    mn = x.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
    sd = x.std(dim=0, keepdim=True).std(dim=1, keepdim=True)
    print('mn', mn.data.cpu().numpy())
    print('std', sd.data.cpu().numpy())
    sd[sd < 2e-2] = 1 - 1e-5
    print('corrected (non zeros) std', sd.data.cpu().numpy())
    mn = mn.to(args.device)
    sd = sd.to(args.device)
    return mn, sd

def train(model, train_loader, optimizer, epoch, device, log_interval, loss_fn):
    model.train()
    optimizer.zero_grad()
    n_samples, correct, train_loss = 0, 0, 0
    start = time.time()

    for batch_idx, data in enumerate(train_loader):
        data = data_to_device(data, device)
        data = normalize_data(data)
        if batch_idx == 0 and epoch <= 1:
            sanity_check(model.eval(), data)  # to disable the effect of dropout or other regularizers that can change behavior from batch to batch
            model.train()

        optimizer.zero_grad()
        output, other_losses = model(data)
        targets = data[3]
        loss = loss_fn(output, targets)
        for l in other_losses:
            loss += l
        loss_item = loss.item()
        train_loss += loss_item
        n_samples += len(targets)
        loss.backward()  # accumulates gradient
        optimizer.step()  # update weights

        time_iter = time.time() - start
        correct += count_correct(output.detach(), targets.detach())

        acc = 100. * correct / n_samples  # average over all examples in the dataset
        train_loss_avg  = train_loss / (batch_idx + 1)

        if (batch_idx > 0 and batch_idx % log_interval == 0) or batch_idx == len(train_loader) - 1:
            print('Train set (epoch {}): [{}/{} ({:.0f}%)]\tLoss: {:.4f} (avg: {:.4f}), other losses: {}\tAcc metric: {}/{} ({:.2f}%)\t avg sec/iter: {:.4f}'.format(
                epoch, n_samples, len(train_loader.dataset), 100. * n_samples / len(train_loader.dataset),
                loss_item, train_loss_avg, ['%.4f' % l.item() for l in other_losses],
                correct, n_samples, acc, time_iter / (batch_idx + 1)))

    print('\n')
    assert n_samples == len(train_loader.dataset), (n_samples, len(train_loader.dataset))

    return train_loss, acc


def test(model, test_loader, epoch, device, loss_fn, set_name, dataset):
    model.eval()
    n_samples, correct, test_loss = 0, 0, 0
    pred, targets, N_nodes = [], [], []
    start = time.time()

    for batch_idx, data in enumerate(test_loader):
        optimizer.zero_grad()
        data = data_to_device(data, device)
        data = normalize_data(data)
        if batch_idx == 0 and epoch <= 1:
            sanity_check(model, data)

        output, other_losses = model(data)
        loss = loss_fn(output, data[3], reduction='sum')
        for l in other_losses:
            loss += l
        test_loss += loss.item()
        pred.append(output.detach())
        targets.append(data[3].detach())
        N_nodes.append(data[4]['N_nodes'])


    pred = torch.cat(pred)
    targets = torch.cat(targets)
    N_nodes = torch.cat(N_nodes)
    if dataset.find('colors') >= 0:
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=25)
        correct += count_correct(pred[2500:5000], targets[2500:5000], N_nodes=N_nodes[2500:5000], N_nodes_min=26, N_nodes_max=200)
        correct += count_correct(pred[5000:], targets[5000:], N_nodes=N_nodes[5000:], N_nodes_min=26, N_nodes_max=200)
    elif dataset == 'triangles':
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=25)
        correct += count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=26, N_nodes_max=100)
    else:
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=1e5)

    time_iter = time.time() - start

    n_samples = len(test_loader.dataset)
    test_loss_avg = test_loss / n_samples
    acc = 100. * correct / n_samples  # average over all examples in the dataset

    print('{} set (epoch {}): Avg loss: {:.4f}, Acc metric: {}/{} ({:.2f}%)\t avg sec/iter: {:.4f}\n'.format(
        set_name.capitalize(), epoch, test_loss_avg, correct, n_samples, acc, time_iter / len(test_loader)))

    return test_loss, acc


if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)
    print('gpus: ', torch.cuda.device_count())

    args = parse_args()

    if not os.path.isdir(args.results):
        os.mkdir(args.results)

    rnd = np.random.RandomState(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset.find('colors') >= 0 or args.dataset == 'triangles':
        train_dataset = SyntheticGraphs(args.data_dir, args.dataset, 'train')
        test_dataset = SyntheticGraphs(args.data_dir, args.dataset, 'test')
        loss_fn = mse_loss
        collate_fn = collate_batch
        in_features = train_dataset.feature_dim
        out_features = 1
    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
        loss_fn = F.cross_entropy
        collate_fn = collate_batch_images
        in_features = 3
        out_features = 10
    else:
        raise NotImplementedError(args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.threads, collate_fn=collate_fn)

    model = ChebyGIN(in_features=in_features,
                     out_features=out_features,
                     filters=args.filters,
                     K=args.filter_scale,
                     n_hidden=args.n_hidden,
                     aggregation=args.aggregation,
                     dropout=args.dropout,
                     readout=args.readout,
                     pool=args.pool)
    print(model)
    # Compute the total number of trainable parameters
    print('model capacity: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    model.to(args.device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wdecay,
        betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train_loss, acc = train(model, train_loader, optimizer, epoch, args.device, args.log_interval, loss_fn)
        test_loss, acc = test(model, test_loader, epoch, args.device, loss_fn, 'test', args.dataset)

    print('done in {}'.format(datetime.datetime.now() - dt))
