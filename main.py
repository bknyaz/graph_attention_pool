import argparse
import random
import datetime
from torchvision import transforms
from graphdata import *
from train_test import *
import warnings

warnings.filterwarnings("once")

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments with Graph Neural Networks')
    # Dataset
    parser.add_argument('-D', '--dataset', type=str, default='colors-3',
                        choices=['colors-3', 'colors-4', 'colors-8', 'colors-16', 'colors-32',
                                 'triangles', 'mnist', 'mnist-75sp', 'TU'],
                        help='colors-n means the colors dataset with n-dimensional features; TU is any dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets')
    parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to the dataset')
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=None, help='# of the epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training data')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--lr_decay_step', type=str, default=None, help='number of epochs after which to reduce learning rate')
    parser.add_argument('--wdecay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('-f', '--filters', type=str, default='64,64,64', help='number of filters in each graph layer')
    parser.add_argument('-K', '--filter_scale', type=int, default=1, help='filter scale (receptive field size), must be > 0; 1 for GCN or GIN')
    parser.add_argument('--n_hidden', type=int, default=0, help='number of hidden units inside the graph layer')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'sum'], help='neighbors aggregation inside the graph layer')
    parser.add_argument('--readout', type=str, default=None, choices=['mean', 'sum', 'max'], help='type of global pooling over all nodes')
    parser.add_argument('--kl_weight', type=float, default=100, help='weight of the KL term in the loss')
    parser.add_argument('--pool', type=str, default=None, help='type of pooling between layers, None for global pooling only')
    parser.add_argument('--pool_arch', type=str, default=None, help='pooling layers architecture defining whether to use fully-connected layers or GNN and to which layer to attach (e.g.: fc_prev, gnn_prev, fc_curr, gnn_curr, fc_prev_32)')
    parser.add_argument('--init', type=str, default='normal', choices=['normal', 'uniform'], help='distribution used for initialization for the attention model')
    parser.add_argument('--scale', type=str, default='1', help='initialized weights scale for the attention model, set to None to use PyTorch default init')
    parser.add_argument('--degree_feature', action='store_true', default=False, help='use degree features (only for the Triangles dataset)')
    # TU datasets arguments
    parser.add_argument('--n_nodes', type=int, default=25, help='maximum number of nodes in the training set for collab, proteins and dd (35 for collab, 25 for proteins, 200 or 300 for dd)')
    parser.add_argument('--cv_folds', type=int, default=5, help='number of folds for cross-validating hyperparameters for collab, proteins and dd (5 or 10 shows similar results, 5 is faster)')
    parser.add_argument('--cv_threads', type=int, default=5, help='number of parallel threads for cross-validation')
    parser.add_argument('--tune_init', action='store_true', default=False, help='do not tune initialization hyperparameters')
    parser.add_argument('--ax', action='store_true', default=False, help='use AX for hyperparameter optimization (recommended)')
    parser.add_argument('--ax_trials', type=int, default=30, help='number of AX trials (hyperparameters optimization steps)')
    parser.add_argument('--cv', action='store_true', default=False, help='run in the cross-validation mode')
    parser.add_argument('--seed_data', type=int, default=111, help='random seed for data splits')
    # Image datasets arguments
    parser.add_argument('--img_features', type=str, default='mean,coord', help='image features to use as node features')
    parser.add_argument('--img_noise_levels', type=str, default=None,
                        help='Gaussian noise standard deviations for grayscale and color image features')
    # Auxiliary arguments
    parser.add_argument('--validation', action='store_true', default=False, help='run in the validation mode')
    parser.add_argument('--debug', action='store_true', default=False, help='evaluate on the test set after each epoch (only for visualization purposes)')
    parser.add_argument('--eval_attn_train', action='store_true', default=False, help='evaluate attention and save coefficients on the training set for models without learnable attention')
    parser.add_argument('--eval_attn_test', action='store_true', default=False, help='evaluate attention and save coefficients on the test set for models without learnable attention')
    parser.add_argument('--test_batch_size', type=int, default=100, help='batch size for test data')
    parser.add_argument('--alpha_ws', type=str, default=None, help='attention labels that will be used for (weak)supervision')
    parser.add_argument('--log_interval', type=int, default=400, help='print interval')
    parser.add_argument('--results', type=str, default='./results', help='directory to save model checkpoints and other results, set to None to prevent saving anything')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to load the model and optimzer states from and continue training')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='cuda/cpu')
    parser.add_argument('--seed', type=int, default=111, help='random seed for model parameters')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader')
    args = parser.parse_args()

    # Set default number of epochs and learning rate schedules and other hyperparameters
    if args.readout in [None, 'None']:
        args.readout = 'max'  # global max pooling for all datasets except for COLORS
    set_default_lr_decay_step = args.lr_decay_step in [None, 'None']
    if args.epochs in [None, 'None']:
        if args.dataset.find('mnist') >= 0:
            args.epochs = 30
            if set_default_lr_decay_step:
                args.lr_decay_step = '20,25'
        elif args.dataset == 'triangles':
            args.epochs = 100
            if set_default_lr_decay_step:
                args.lr_decay_step = '85,95'
        elif args.dataset == 'TU':
            args.epochs = 50
            if set_default_lr_decay_step:
                args.lr_decay_step = '25,35,45'
        elif args.dataset.find('color') >= 0:
            if args.readout in [None, 'None']:
                args.readout = 'sum'
            if args.pool in [None, 'None']:
                args.epochs = 100
                if set_default_lr_decay_step:
                    args.lr_decay_step = '90'
            else:
                args.epochs = 300
                if set_default_lr_decay_step:
                    args.lr_decay_step = '280'
        else:
            raise NotImplementedError(args.dataset)

    args.lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    args.filters = list(map(int, args.filters.split(',')))
    args.img_features = args.img_features.split(',')
    args.img_noise_levels = None if args.img_noise_levels in [None, 'None'] else list(map(float, args.img_noise_levels.split(',')))
    args.pool = None if args.pool in [None, 'None'] else args.pool.split('_')
    args.pool_arch = None if args.pool_arch in [None, 'None'] else args.pool_arch.split('_')
    try:
        args.scale = float(args.scale)
    except:
        args.scale = None

    args.torch = torch.__version__

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args


def load_synthetic(args):
    train_dataset = SyntheticGraphs(args.data_dir, args.dataset, 'train', degree_feature=args.degree_feature,
                                    attn_coef=args.alpha_ws)
    test_dataset = SyntheticGraphs(args.data_dir, args.dataset, 'val' if args.validation else 'test',
                                   degree_feature=args.degree_feature)
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
        color_noises = load_save_noise(color_noise_file, (noise_shape[0], noise_shape[1], 3))

    if args.dataset == 'mnist':
        A, coord, mask = precompute_graph_images(train_dataset.train_data.shape[1])
        collate_fn = lambda batch: collate_batch_images(batch, A, mask, use_mean_px=use_mean_px,
                                                        coord=coord if use_coord else None,
                                                        gt_attn_threshold=gt_attn_threshold,
                                                        replicate_features=args.img_noise_levels is not None)
    else:
        train_dataset.precompute_graph_data(replicate_features=args.img_noise_levels is not None, threads=12)
        test_dataset.precompute_graph_data(replicate_features=args.img_noise_levels is not None, threads=12)
        collate_fn = collate_batch

    loss_fn = F.cross_entropy

    in_features = 0 if args.img_noise_levels is None else 2
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
    scale, init = args.scale, args.init
    n_hidden_attn = float(args.pool_arch[2]) if (args.pool_arch is not None and len(args.pool_arch) > 2) else 0
    if args.pool is None:
        # Global pooling models
        datareader = DataReader(data_dir=args.data_dir, N_nodes=args.n_nodes, rnd_state=rnd_data, folds=0)
        train_dataset = GraphData(datareader, None, 'train_val')
        test_dataset = GraphData(datareader, None, 'test')
        in_features = train_dataset.num_features
        out_features = train_dataset.num_classes
        pool = args.pool
        kl_weight = args.kl_weight
    elif args.pool[1] == 'gt':
        raise ValueError('ground truth attention for TU datasets is not available')
    elif args.pool[1] in ['sup', 'unsup']:
        datareader = DataReader(data_dir=args.data_dir, N_nodes=args.n_nodes, rnd_state=rnd_data, folds=cv_folds)
        if args.ax:
            # Cross-validation using Ax (recommended way), Python3 must be used
            best_parameters = ax_optimize(datareader, args, collate_fn, loss_fn, None, folds=cv_folds,
                                          threads=args.cv_threads, n_trials=args.ax_trials)
            pool = args.pool
            kl_weight = best_parameters['kl_weight']
            if args.tune_init:
                scale, init = best_parameters['scale'], best_parameters['init']
            n_hidden_attn, layer = best_parameters['n_hidden_attn'], 1
            if layer == 0:
                pool = copy.deepcopy(args.pool)
                del pool[3]

            pool = set_pool(best_parameters['pool'], pool)

        else:
            if not args.cv:
                # Run with some fixed parameters without cross-validation
                pool_thresh_values = np.array([float(args.pool[-1])])
                n_hiddens = [n_hidden_attn]
                layers = [1]
            elif args.debug:
                pool_thresh_values = np.array([1e-4, 1e-1])
                n_hiddens = [n_hidden_attn]
                layers = [1]
            else:
                # Cross-validation using grid search (not recommended, since it's time consuming and not effective
                if args.data_dir.lower().find('proteins') >= 0:
                    pool_thresh_values = np.array([2e-3, 5e-3, 1e-2, 3e-2, 5e-2])
                elif args.data_dir.lower().find('dd') >= 0:
                    pool_thresh_values = np.array([1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1])
                elif args.data_dir.lower().find('collab') >= 0:
                    pool_thresh_values = np.array([1e-3, 2e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1])
                else:
                    raise NotImplementedError('this dataset is not supported currently')
                n_hiddens = np.array([0, 32])  # hidden units in the atention model
                layers = np.array([0, 1])  # layer where to attach the attention model

            if args.pool[1] == 'sup' and not args.debug and args.cv:
                kl_weight_values = np.array([0.25, 1, 2, 10])
            else:
                kl_weight_values = np.array([args.kl_weight])  # any value (ignored for unsupervised training)


            if len(pool_thresh_values) > 1 or len(kl_weight_values) > 1 or len(n_hiddens) > 1 or len(layers) > 1:
                val_acc = np.zeros((len(layers), len(n_hiddens), len(pool_thresh_values), len(kl_weight_values)))
                for i_, layer in enumerate(layers):
                    if layer == 0:
                        pool = copy.deepcopy(args.pool)
                        del pool[3]
                    else:
                        pool = args.pool
                    for j_, n_hidden_attn in enumerate(n_hiddens):
                        for k_, pool_thresh in enumerate(pool_thresh_values):
                            for m_, kl_weight in enumerate(kl_weight_values):
                                val_acc[i_, j_, k_, m_] = \
                                    cross_validation(datareader, args, collate_fn, loss_fn, set_pool(pool_thresh, pool),
                                                     kl_weight, None, n_hidden_attn=n_hidden_attn, folds=cv_folds, threads=args.cv_threads)
                ind1, ind2, ind3, ind4 = np.where(val_acc == np.max(val_acc))  # np.argmax returns only first occurrence
                print(val_acc)
                print(ind1, ind2, ind3, ind4, layers[ind1], n_hiddens[ind2], pool_thresh_values[ind3], kl_weight_values[ind4],
                      val_acc[ind1[0], ind2[0], ind3[0], ind4[0]])

                layer = layers[ind1[0]]
                if layer == 0:
                    pool = copy.deepcopy(args.pool)
                    del pool[3]
                else:
                    pool = args.pool
                n_hidden_attn = n_hiddens[ind2[0]]
                pool = set_pool(pool_thresh_values[ind3[0]], pool)
                kl_weight = kl_weight_values[ind4[0]]
            else:
                pool = args.pool
                kl_weight = args.kl_weight

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
            # Train global pooling model
            start_epoch, model, optimizer, scheduler = create_model_optimizer(in_features, out_features, None, kl_weight,
                                                                              args, scale=scale, init=init, n_hidden_attn=n_hidden_attn)
            for epoch in range(start_epoch, args.epochs + 1):
                scheduler.step()
                train_loss, acc = train(model, train_loader, optimizer, epoch, args, loss_fn, None)
            train_loss, train_acc, attn_WS = test(model, train_loader_test, epoch, loss_fn, 'train', args, None,
                                                  eval_attn=True)[:3]
            train_dataset = GraphData(datareader, None, 'train_val', attn_labels=attn_WS)
    else:
        raise NotImplementedError(args.pool)

    return train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features, pool, kl_weight, scale, init, n_hidden_attn


if __name__ == '__main__':

    # mp.set_start_method('spawn')
    dt = datetime.datetime.now()
    print('start time:', dt)
    args = parse_args()
    args.experiment_ID = '%06d' % dt.microsecond
    print('experiment_ID: ', args.experiment_ID)

    if args.cv_threads > 1 and args.dataset == 'TU':
        # this requires python3
        torch.multiprocessing.set_start_method('spawn')

    print('gpus: ', torch.cuda.device_count())

    if args.results not in [None, 'None'] and not os.path.isdir(args.results):
        os.mkdir(args.results)

    rnd, rnd_data = set_seed(args.seed, args.seed_data)

    pool = args.pool
    kl_weight = args.kl_weight
    scale = args.scale
    init = args.init
    n_hidden_attn = float(args.pool_arch[2]) if (args.pool_arch is not None and len(args.pool_arch) > 2) else 0
    if args.dataset.find('colors') >= 0 or args.dataset == 'triangles':
        train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features = load_synthetic(args)
    elif args.dataset in ['mnist', 'mnist-75sp']:
        train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features, noises, color_noises = load_mnist(args)

    else:
        train_dataset, test_dataset, loss_fn, collate_fn, in_features, out_features, pool, kl_weight, scale, init, n_hidden_attn = \
            load_TU(args, cv_folds=args.cv_folds)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads,
                              collate_fn=collate_fn)
    # A loader to test and evaluate attn on the training set (shouldn't be shuffled and have larger batch size multiple of 50)
    train_loader_test = DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.threads, collate_fn=collate_fn)
    print('test_dataset', test_dataset.split)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=args.threads, collate_fn=collate_fn)

    start_epoch, model, optimizer, scheduler = create_model_optimizer(in_features, out_features, pool, kl_weight, args,
                                                                      scale=scale, init=init, n_hidden_attn=n_hidden_attn)

    feature_stats = None
    if args.dataset in ['mnist', 'mnist-75sp']:
        feature_stats = compute_feature_stats(model, train_loader, args.device, n_batches=1000)

    # Test function wrapper
    def test_fn(loader, epoch, split, eval_attn):
        test_loss, acc, _, _ = test(model, loader, epoch, loss_fn, split, args, feature_stats,
                                        noises=None, img_noise_level=None, eval_attn=eval_attn, alpha_WS_name='orig')
        if args.dataset in ['mnist', 'mnist-75sp'] and split == 'test' and args.img_noise_levels is not None:
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
