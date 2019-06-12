import sys
import argparse
import datetime
import platform
from torchvision import transforms
from graphdata import *
from train_test import *

sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments with Graph Neural Networks')
    # Dataset
    parser.add_argument('-D', '--dataset', type=str, default='colors-3',
                        choices=['colors-3', 'colors-4', 'colors-8', 'colors-16', 'colors-32',
                                 'triangles', 'mnist', 'mnist-75sp', 'TU'],
                        help='colors-n means the colors dataset with n-dimensional features: TU is any dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets')
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
    parser.add_argument('--n_nodes', type=int, default=25, help='maximum number of nodes in the training set for collab, proteins and dd')
    parser.add_argument('--cv_folds', type=int, default=5, help='number of folds for cross-validating hyperparameters for collab, proteins and dd')
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
                        help='directory to save model checkpoints and other results, set to None to prevent saving anything')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to load the model and optimzer states from and continue training')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='cuda/cpu')
    parser.add_argument('--seed', type=int, default=11, help='seed for shuffling nodes')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader')
    args = parser.parse_args()

    args.lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    args.filters = list(map(int, args.filters.split(',')))
    args.img_features = args.img_features.split(',')
    args.img_noise_levels = list(map(float, args.img_noise_levels.split(',')))
    args.pool = None if args.pool in [None, 'None'] else args.pool.split('_')
    args.pool_arch = None if args.pool_arch in [None, 'None'] else args.pool_arch.split('_')

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args


def load_synthetic(args):
    train_dataset = SyntheticGraphs(args.data_dir, args.dataset, 'train')
    test_dataset = SyntheticGraphs(args.data_dir, args.dataset, 'val' if args.validation else 'test')
    loss_fn = mse_loss
    collate_fn = collate_batch
    in_features = train_dataset.feature_dim
    out_features = 1
    return train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features


def load_mnist(args):
    use_mean_px = 'mean' in args.img_features
    use_coord = 'coord' in args.img_features
    assert use_mean_px, ('this mode is not well supported', use_mean_px)
    gt_attn_threshold = 0 if (args.pool is not None and args.pool[1] in ['gt'] and args.filter_scale > 1) else 0.5
    if args.dataset == 'mnist':
        train_dataset = MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor(),
                              attn_coef=args.alpha_ws)
    else:
        train_dataset = MNIST75sp(args.data_dir, split='train', use_mean_px=use_mean_px, use_coord=use_coord,
                                  gt_attn_threshold=gt_attn_threshold, attn_coef=args.alpha_ws)

    noises, color_noises = None, None
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
            test_dataset = MNIST75sp(args.data_dir, split='train', use_mean_px=use_mean_px, use_coord=use_coord,
                                     gt_attn_threshold=gt_attn_threshold)
            test_dataset.train_val_split(np.arange(train_dataset.n_samples - n_val, train_dataset.n_samples))
    else:
        noise_file = pjoin(args.data_dir, '%s_noise.pt' % args.dataset.replace('-', '_'))
        color_noise_file = pjoin(args.data_dir, '%s_color_noise.pt' % args.dataset.replace('-', '_'))
        if args.dataset == 'mnist':
            test_dataset = MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
            noise_shape = (len(test_dataset.test_labels), 28 * 28)
        else:
            test_dataset = MNIST75sp(args.data_dir, split='test', use_mean_px=use_mean_px, use_coord=use_coord,
                                     gt_attn_threshold=gt_attn_threshold)
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

    return train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features, noises, color_noises


def load_TU(args, cv_folds=5):
    loss_fn = F.cross_entropy
    collate_fn = collate_batch
    if args.pool[1] == 'gt':
        raise ValueError('ground truth attention for TU datasets is not available')
    elif args.pool in [None, 'None']:
        # Global pooling models
        datareader = DataReader(data_dir=args.data_dir, N_nodes=args.n_nodes, rnd_state=rnd, folds=0)
        train_dataset = GraphData(datareader, None, 'train_val')
        test_dataset = GraphData(datareader, None, 'test')
        in_features = train_dataset.num_features
        out_features = train_dataset.num_classes
        pool = args.pool
        kl_weight = args.kl_weight
    elif args.pool[1] in ['sup', 'unsup']:
        datareader = DataReader(data_dir=args.data_dir, N_nodes=args.n_nodes, rnd_state=rnd, folds=cv_folds)
        def set_pool(pool_thresh):
            pool = copy.deepcopy(args.pool)
            for i, s in enumerate(pool):
                try:
                    thresh = float(s)
                    pool[i] = str(pool_thresh)
                except:
                    continue
            return pool

        val_acc = []
        pool_thresh_values = np.array([1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1])
        if args.pool[1] == 'sup':
            kl_weight_values = np.array([0.1, 0.5, 1, 2, 10, 100])
        else:
            kl_weight_values = np.array([100])  # any value (ignored for unsupervised training)

        for pool_thresh in pool_thresh_values:
            for kl_weight in kl_weight_values:
                val_acc.append(
                    cross_validation(datareader, args, collate_fn, loss_fn, set_pool(pool_thresh), kl_weight, None,
                                     folds=cv_folds))
        val_acc = np.array(val_acc).reshape(len(pool_thresh_values), len(kl_weight_values))
        ind1, ind2 = np.where(val_acc == np.max(val_acc))  # np.argmax returns only first occurrence
        print(val_acc)
        print(ind1, ind2, pool_thresh_values[ind1], kl_weight_values[ind2], val_acc[ind1[0], ind2[0]])
        pool = set_pool(pool_thresh_values[ind1[0]])
        kl_weight = kl_weight_values[ind2[0]]

        train_dataset = GraphData(datareader, None, 'train_val')
        test_dataset = GraphData(datareader, None, 'test')
        in_features = train_dataset.num_features
        out_features = train_dataset.num_classes

        if args.pool[1] == 'sup':
            # Train a model with global pooling first
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads,
                                      collate_fn=collate_fn)
            train_loader_test = DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False,
                                           num_workers=args.threads, collate_fn=collate_fn)

            start_epoch, model, optimizer, scheduler = create_model_optimizer(in_features, out_features, None, kl_weight,
                                                                              args)
            for epoch in range(start_epoch, args.epochs + 1):
                scheduler.step()
                train_loss, acc = train(model, train_loader, optimizer, epoch, args, loss_fn, None)
            train_loss, train_acc, attn_WS = test(model, train_loader_test, epoch, loss_fn, 'train', args, None,
                                                  eval_attn=True)
            train_dataset = GraphData(datareader, None, 'train_val', attn_labels=attn_WS)

    return train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features, pool, kl_weight


def set_seed(seed):
    rnd = np.random.RandomState(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return rnd


if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)
    print('gpus: ', torch.cuda.device_count())
    args = parse_args()
    args.experiment_ID = '%s_%06d' % (platform.node(), dt.microsecond)
    print('experiment_ID: ', args.experiment_ID)

    if args.results not in [None, 'None'] and not os.path.isdir(args.results):
        os.mkdir(args.results)

    rnd = set_seed(args.seed)

    pool = args.pool
    kl_weight = args.kl_weight
    if args.dataset.find('colors') >= 0 or args.dataset == 'triangles':
        train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features = load_synthetic(args)
    elif args.dataset in ['mnist', 'mnist-75sp']:
        train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features, noises, color_noises = load_mnist(args)
    else:
        train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features, pool, kl_weight = load_TU(args, cv_folds=args.cv_folds)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads, collate_fn=collate_fn)
    # A loader to test and evaluate attn on the training set (shouldn't be shuffled and have larger batch size multiple of 50)
    train_loader_test = DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.threads, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.threads, collate_fn=collate_fn)

    start_epoch, model, optimizer, scheduler = create_model_optimizer(in_features, out_features, pool, kl_weight, args)

    feature_stats = None
    if args.dataset in ['mnist', 'mnist-75sp']:
        feature_stats = compute_feature_stats(model, train_loader, args.device, n_batches=1000)

    # Test function wrapper
    def test_fn(loader, epoch, split, eval_attn):
        # eval_attn = (epoch == args.epochs) and args.eval_attn_test
        test_loss, acc, _ = test(model, loader, epoch, loss_fn, split, args, feature_stats,
                               noises=None, img_noise_level=None, eval_attn=eval_attn, alpha_WS_name='orig')
        if args.dataset in ['mnist', 'mnist-75sp'] and split == 'test':
            test(model, loader, epoch, loss_fn, split, args, feature_stats,
                 noises=noises, img_noise_level=args.img_noise_levels[0], eval_attn=eval_attn, alpha_WS_name='noisy')
            test(model, loader, epoch, loss_fn, split, args, feature_stats,
                 noises=color_noises, img_noise_level=args.img_noise_levels[1], eval_attn=eval_attn, alpha_WS_name='noisy-c')
        return test_loss, acc

    if start_epoch > args.epochs:
        print('evaluating the model')
        test_fn(test_loader, start_epoch - 1, 'val' if args.validation else 'test', args.eval_attn_test)
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            eval_epoch = epoch <= 1 or epoch == args.epochs  # check for epoch == 1 just to make sure that the test function works fine for this test set before training all the way until the last epoch
            scheduler.step()
            # test_fn(train_loader_test, epoch, 'train', True)
            # test_fn(test_loader, epoch, 'test', True)
            train_loss, acc = train(model, train_loader, optimizer, epoch, args, loss_fn, feature_stats)
            if eval_epoch:
                save_checkpoint(model, scheduler, optimizer, args, epoch)
                # Report Training accuracy and other metrics on the training set
                test_fn(train_loader_test, epoch, 'train', (epoch == args.epochs) and args.eval_attn_train)

            if args.validation:
                test_fn(test_loader, epoch, 'val', (epoch == args.epochs) and args.eval_attn_test)
            elif eval_epoch or args.debug:
                test_fn(test_loader, epoch, 'test', (epoch == args.epochs) and args.eval_attn_test)

    print('done in {}'.format(datetime.datetime.now() - dt))
