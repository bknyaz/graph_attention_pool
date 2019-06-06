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
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from chebygin import *
from graphdata import *
from utils import *

sys.stdout.flush()

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
    parser.add_argument('--pool_arch', type=str, default=None, help='pooling layers architecture')
    parser.add_argument('--img_features', type=str, default='mean,coord', help='which image features to use as node features')
    # Auxiliary arguments
    parser.add_argument('--validation', action='store_true', default=False, help='run in the validation mode')
    parser.add_argument('--debug', action='store_true', default=False, help='evaluate on the test set after each epoch (only for visualization purposes)')
    parser.add_argument('--eval_attn_train', action='store_true', default=False, help='evaluate attention and save coefficients on the training set for models without learnable attention')
    parser.add_argument('--eval_attn_test', action='store_true', default=False, help='evaluate attention and save coefficients on the test set for models without learnable attention')
    parser.add_argument('--test_batch_size', type=int, default=100, help='batch size for test data')
    parser.add_argument('--alpha_ws', type=str, default=None, help='attention labels that will be used for (weak)supervision')
    parser.add_argument('--img_noise_levels', type=str, default='0.75,1.25', help='Gaussian noise standard deviations for grayscale and color image features')
    parser.add_argument('--log_interval', type=int, default=400, help='print interval')
    parser.add_argument('--results', type=str, default='./results',
                        help='directory to save model checkpoints and other results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='cuda/cpu')
    parser.add_argument('--seed', type=int, default=11, help='seed for shuffling nodes')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader')
    args = parser.parse_args()

    args.lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    args.filters = list(map(int, args.filters.split(',')))
    args.img_features = args.img_features.split(',')
    args.img_noise_levels = list(map(float, args.img_noise_levels.split(',')))

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args


def train(model, train_loader, optimizer, epoch, device, log_interval, loss_fn, feature_stats=None):
    model.train()
    optimizer.zero_grad()
    n_samples, correct, train_loss = 0, 0, 0
    alpha_pred, alpha_GT = {}, []
    start = time.time()

    # with torch.autograd.set_detect_anomaly(True):
    for batch_idx, data in enumerate(train_loader):
        data = data_to_device(data, device)
        if feature_stats is not None:
            data[0] = (data[0] - feature_stats[0]) / feature_stats[1]
        if batch_idx == 0 and epoch <= 1:
            sanity_check(model.eval(), data)  # to disable the effect of dropout or other regularizers that can change behavior from batch to batch
            model.train()
        optimizer.zero_grad()
        output, other_losses, alpha = model(data)
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

        alpha_GT.append(data[4]['node_attn'].data.cpu().numpy())
        for layer in range(len(alpha)):
            if layer not in alpha_pred:
                alpha_pred[layer] = []
            alpha_pred[layer].append(alpha[layer].data.cpu().numpy().flatten())

        acc = 100. * correct / n_samples  # average over all examples in the dataset
        train_loss_avg  = train_loss / (batch_idx + 1)

        if (batch_idx > 0 and batch_idx % log_interval == 0) or batch_idx == len(train_loader) - 1:
            print('Train set (epoch {}): [{}/{} ({:.0f}%)]\tLoss: {:.4f} (avg: {:.4f}), other losses: {}\tAcc metric: {}/{} ({:.2f}%)\t AttnAUC: {}\t avg sec/iter: {:.4f}'.format(
                epoch, n_samples, len(train_loader.dataset), 100. * n_samples / len(train_loader.dataset),
                loss_item, train_loss_avg, ['%.4f' % l.item() for l in other_losses],
                correct, n_samples, acc, ['%.2f' % a for a in attn_AUC(alpha_GT, alpha_pred)],
                time_iter / (batch_idx + 1)))

    print('\n')
    assert n_samples == len(train_loader.dataset), (n_samples, len(train_loader.dataset))

    return train_loss, acc


def test(model, test_loader, epoch, device, loss_fn, split, dataset, results_dir, seed, feature_stats=None, noises=None, img_noise_level=None, eval_attn=False, alpha_WS_name=''):
    model.eval()
    n_samples, correct, test_loss = 0, 0, 0
    pred, targets, N_nodes = [], [], []
    start = time.time()
    alpha_pred, alpha_GT = {}, []
    if eval_attn:
        alpha_pred[0] = []
        print('testing with evaluation of attention: takes longer time')

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            optimizer.zero_grad()
            data = data_to_device(data, device)
            if feature_stats is not None:
                data[0] = (data[0] - feature_stats[0]) / feature_stats[1]
            if batch_idx == 0 and epoch <= 1:
                sanity_check(model, data)

            if noises is not None:
                noise = noises[n_samples:n_samples + len(data[0])].to(device) * img_noise_level
                if len(noise.shape) == 2:
                    noise = noise.unsqueeze(2)
                data[0][:, :, :3] = data[0][:, :, :3] + noise

            output, other_losses, alpha = model(data)
            loss = loss_fn(output, data[3], reduction='sum')
            for l in other_losses:
                loss += l
            test_loss += loss.item()
            pred.append(output.detach())
            targets.append(data[3].detach())
            N_nodes.append(data[4]['N_nodes'])
            alpha_GT.append(data[4]['node_attn'].data.cpu().numpy())
            if eval_attn:
                assert len(alpha) == 0, ('invalid mode, eval_attn should be false')
                alpha_pred[0].append(attn_heatmaps(model, device, data, output.data, test_loader.batch_size, constant_mask=args.dataset=='mnist').data.cpu().numpy())
            else:
                for layer in range(len(alpha)):
                    if layer not in alpha_pred:
                        alpha_pred[layer] = []
                    alpha_pred[layer].append(alpha[layer].data.cpu().numpy())

            n_samples += len(data[0])
            if eval_attn: # and n_samples % 10 == 0:
                print('{}/{} samples processed'.format(n_samples, len(test_loader.dataset)))

            # if eval_attn and n_samples > 10:
            #     break

    assert n_samples == len(test_loader.dataset), (n_samples, len(test_loader.dataset))

    pred = torch.cat(pred)
    targets = torch.cat(targets)
    N_nodes = torch.cat(N_nodes)
    if dataset.find('colors') >= 0:
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=25)
        if pred.shape[0] > 2500:
            correct += count_correct(pred[2500:5000], targets[2500:5000], N_nodes=N_nodes[2500:5000], N_nodes_min=26, N_nodes_max=200)
            correct += count_correct(pred[5000:], targets[5000:], N_nodes=N_nodes[5000:], N_nodes_min=26, N_nodes_max=200)
    elif dataset == 'triangles':
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=25)
        if data[0].shape[1] > 25:
            correct += count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=26, N_nodes_max=100)
    else:
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=1e5)

    time_iter = time.time() - start

    test_loss_avg = test_loss / n_samples
    acc = 100. * correct / n_samples  # average over all examples in the dataset
    print('{} set (epoch {}): Avg loss: {:.4f}, Acc metric: {}/{} ({:.2f}%)\t AttnAUC: {}\t avg sec/iter: {:.4f}\n'.format(
        split.capitalize(), epoch, test_loss_avg, correct, n_samples, acc,
        ['%.2f' % a for a in attn_AUC(alpha_GT, alpha_pred)], time_iter / (batch_idx + 1)))

    if eval_attn:
        if results_dir in [None, 'None']:
            print('skip saving alpha values, invalid results dir: %s' % results_dir)
        else:
            with open(pjoin(results_dir, 'alpha_WS_%s_seed%d_%s.pkl' % (split, seed, alpha_WS_name)), 'wb') as f:
                pickle.dump(alpha_pred[0], f, protocol=2)

    return test_loss, acc


def save_checkpoint(model, scheduler, optimizer, args, epoch):
    if args.results in [None, 'None']:
        print('skip saving checkpoint, invalid results dir: %s' % args.results)
        return
    fname = '%s/checkpoint_%s_epoch%d_seed%d.pth.tar' % (args.results, args.dataset, epoch, args.seed)
    try:
        print('saving the model to %s' % fname)
        state = {
            'epoch': epoch,
            'args': args,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if os.path.isfile(fname):
            print('WARNING: file %s exists and will be overwritten' % fname)
        torch.save(state, fname)
    except Exception as e:
        print('error saving the model', e)


if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)
    print('gpus: ', torch.cuda.device_count())

    args = parse_args()

    if args.results not in [None, 'None'] and not os.path.isdir(args.results):
        os.mkdir(args.results)

    rnd = np.random.RandomState(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset.find('colors') >= 0 or args.dataset == 'triangles':
        train_dataset = SyntheticGraphs(args.data_dir, args.dataset, 'train')
        test_dataset = SyntheticGraphs(args.data_dir, args.dataset, 'val' if args.validation else 'test')
        loss_fn = mse_loss
        collate_fn = collate_batch
        in_features = train_dataset.feature_dim
        out_features = 1
    elif args.dataset in ['mnist', 'mnist-75sp']:
        use_mean_px = 'mean' in args.img_features
        use_coord = 'coord' in args.img_features
        assert use_mean_px, ('this mode is not well supported', use_mean_px)
        gt_attn_threshold = 0 if (args.pool is not None and args.pool[1] == 'gt' and args.filter_scale > 1) else 0.5
        if args.dataset == 'mnist':
            train_dataset = MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor(), attn_coef=args.alpha_ws)
        else:
            train_dataset = MNIST75sp(args.data_dir, split='train', use_mean_px=use_mean_px, use_coord=use_coord,
                                      gt_attn_threshold=gt_attn_threshold, attn_coef=args.alpha_ws)

        if args.validation:
            n_val = 5000
            if args.dataset == 'mnist':
                train_dataset.train_data = train_dataset.train_data[:-n_val]
                train_dataset.train_labels = train_dataset.train_labels[:-n_val]
                test_dataset = MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
                test_dataset.train_data = train_dataset.train_data[-n_val:]
                test_dataset.train_labels = train_dataset.train_labels[-n_val:]
            else:
                train_dataset.train_val_split(np.arange(0, train_dataset.n_samples - n_val))
                test_dataset = MNIST75sp(args.data_dir, split='train', use_mean_px=use_mean_px, use_coord=use_coord, gt_attn_threshold=gt_attn_threshold)
                test_dataset.train_val_split(np.arange(train_dataset.n_samples - n_val, train_dataset.n_samples))
        else:
            noise_file = pjoin(args.data_dir, '%s_noise.pt' % args.dataset.replace('-', '_'))
            color_noise_file = pjoin(args.data_dir, '%s_color_noise.pt' % args.dataset.replace('-', '_'))
            if args.dataset == 'mnist':
                test_dataset = MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
                noise_shape = (len(test_dataset.test_labels), 28 * 28)
            else:
                test_dataset = MNIST75sp(args.data_dir, split='test', use_mean_px=use_mean_px, use_coord=use_coord, gt_attn_threshold=gt_attn_threshold)
                noise_shape = (len(test_dataset.labels), 75)

            # Generate/load noise (save it to make reproducible)
            noises = load_save_noise(noise_file, noise_shape)
            color_noises = load_save_noise(color_noise_file, (*(noise_shape), 3))

        if args.dataset == 'mnist':
            A, coord, mask = precompute_graph_images(train_dataset.train_data.shape[1])
            collate_fn = lambda batch: collate_batch_images(batch, A, mask, use_mean_px=use_mean_px,
                                                            coord=coord if use_coord else None,
                                                            gt_attn_threshold=gt_attn_threshold)
        else:
            train_dataset.precompute_graph_images()
            test_dataset.precompute_graph_images()
            collate_fn = collate_batch

        loss_fn = F.cross_entropy

        in_features = 2
        for features in args.img_features:
            if features == 'mean':
                in_features += 1
            elif features == 'coord':
                in_features += 2
            else:
                raise NotImplementedError(features)
        in_features = np.max((in_features, 1))  # in_features=1 if neither mean nor coord are used  (dummy features will be used in this case)
        out_features = 10
    else:
        raise NotImplementedError(args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads, collate_fn=collate_fn)
    # A loader to test and evaluate attn on the training set (shouldn't be shuffled and have larger batch size multiple of 50)
    train_loader_test = DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.threads, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.threads, collate_fn=collate_fn)

    model = ChebyGIN(in_features=in_features,
                     out_features=out_features,
                     filters=args.filters,
                     K=args.filter_scale,
                     n_hidden=args.n_hidden,
                     aggregation=args.aggregation,
                     dropout=args.dropout,
                     readout=args.readout,
                     pool=args.pool,
                     pool_arch=args.pool_arch,
                     kl_weight=args.kl_weight,
                     debug=args.debug)
    print(model)
    # Compute the total number of trainable parameters
    print('model capacity: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1)

    feature_stats = None
    if args.dataset in ['mnist', 'mnist-75sp']:
        feature_stats = compute_feature_stats(model, train_loader, args.device, n_batches=1000)

    # Test function wrapper
    test_fn = lambda loader, epoch, split, noises, img_noise_level, eval_attn, alpha_WS_name: \
        test(model, loader, epoch, args.device, loss_fn, split, args.dataset, args.results, args.seed,
             feature_stats, noises=noises, img_noise_level=img_noise_level, eval_attn=eval_attn, alpha_WS_name=alpha_WS_name)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        eval_epoch = epoch <= 1 or epoch == args.epochs
        eval_attn = (epoch == args.epochs) and args.eval_attn_train

        # train_loss, acc = test_fn(train_loader_test, epoch, 'train', None, None, True, 'orig')

        train_loss, acc = train(model, train_loader, optimizer, epoch, args.device, args.log_interval, loss_fn, feature_stats)
        if eval_epoch:
            save_checkpoint(model, scheduler, optimizer, args, epoch)
            # Report Training accuracy and other metrics on the training set
            train_loss, acc = test_fn(train_loader_test, epoch, 'train', None, None, eval_attn, 'orig')

        eval_attn = (epoch == args.epochs) and args.eval_attn_test
        if args.validation:
            test_loss, acc = test_fn(test_loader, epoch, 'val', None, None, eval_attn, 'orig')

        elif eval_epoch or args.debug:
            # check for epoch == 1 just to make sure that the test function works fine for this test set before training all the way until the last epoch
            test_loss, acc = test_fn(test_loader, epoch, 'test', None, None, eval_attn, 'orig')
            if args.dataset in ['mnist', 'mnist-75sp']:
                test_loss, acc =  test_fn(test_loader, epoch, 'test', noises, args.img_noise_levels[0], eval_attn, 'noisy')
                test_loss, acc = test_fn(test_loader, epoch, 'test', color_noises, args.img_noise_levels[1], eval_attn, 'noisy-c')

    print('done in {}'.format(datetime.datetime.now() - dt))
