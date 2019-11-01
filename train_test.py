import time
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from chebygin import *
from utils import *
from graphdata import *
import torch.multiprocessing as mp
import multiprocessing
try:
    import ax
    from ax.service.managed_loop import optimize
except Exception as e:
    print('AX is not available: %s' % str(e))


def set_pool(pool_thresh, args_pool):
    pool = copy.deepcopy(args_pool)
    for i, s in enumerate(pool):
        try:
            thresh = float(s)
            pool[i] = str(pool_thresh)
        except:
            continue
    return pool


def train_evaluate(datareader, args, collate_fn, loss_fn, feature_stats, parameterization, folds=10, threads=5):

    print('parameterization', parameterization)

    pool_thresh, kl_weight = parameterization['pool'], parameterization['kl_weight']
    pool = args.pool

    if args.tune_init:
        scale, init = parameterization['scale'], parameterization['init']
    else:
        scale, init = args.scale, args.init

    n_hidden_attn, layer = parameterization['n_hidden_attn'], 1
    if layer == 0:
        pool = copy.deepcopy(args.pool)
        del pool[3]

    pool = set_pool(pool_thresh, pool)

    manager = multiprocessing.Manager()
    val_acc = manager.dict()
    assert threads <= folds, (threads, folds)
    n_it = int(np.ceil(float(folds) / threads))
    for i in range(n_it):
        processes = []
        if threads <= 1:
            single_job(i * threads, datareader, args, collate_fn, loss_fn, pool, kl_weight,
                       feature_stats, val_acc, scale=scale, init=init, n_hidden_attn=n_hidden_attn)
        else:
            for fold in range(threads):
                p = mp.Process(target=single_job,
                               args=(i * threads + fold, datareader, args, collate_fn, loss_fn, pool, kl_weight,
                                     feature_stats, val_acc, scale, init, n_hidden_attn))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

    print(val_acc)
    val_acc = list(val_acc.values())
    print('average and std over {} folds: {} +- {}'.format(folds, np.mean(val_acc), np.std(val_acc)))
    metric = np.mean(val_acc) - np.std(val_acc)  # large std is considered bad
    print('metric: avg acc - std: {}'.format(metric))
    return metric


def ax_optimize(datareader, args, collate_fn, loss_fn, feature_stats, folds=10, threads=5, n_trials=30):
    parameters = [
            {"name": "pool", "type": "range", "bounds": [1e-4, 2e-2], "log_scale": False},
            {"name": "kl_weight", "type": "range", "bounds": [0.1, 10.], "log_scale": False},
            {"name": "n_hidden_attn", "type": "choice", "values": [0, 32]}  # hidden units in the attention layer (0: no hidden layer)
        ]

    if args.tune_init:
        parameters.extend([{"name": "scale", "type": "range", "bounds": [0.1, 2.], "log_scale": False},
                           {"name": "init", "type": "choice", "values": ['normal', 'uniform']}])

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=lambda parameterization: train_evaluate(datareader,
                                                                    args, collate_fn, loss_fn,
                                                                    feature_stats, parameterization, folds=folds,
                                                                    threads=threads),
        total_trials=n_trials,
        objective_name='accuracy',
    )

    print('best_parameters', best_parameters)
    print('values', values)
    return best_parameters


def train(model, train_loader, optimizer, epoch, args, loss_fn, feature_stats=None, log=True):
    model.train()
    optimizer.zero_grad()
    n_samples, correct, train_loss = 0, 0, 0
    alpha_pred, alpha_GT = {}, {}
    start = time.time()

    # with torch.autograd.set_detect_anomaly(True):
    for batch_idx, data in enumerate(train_loader):
        data = data_to_device(data, args.device)
        if feature_stats is not None:
            data[0] = (data[0] - feature_stats[0]) / feature_stats[1]
        if batch_idx == 0 and epoch <= 1:
            sanity_check(model.eval(), data)  # to disable the effect of dropout or other regularizers that can change behavior from batch to batch
            model.train()
        optimizer.zero_grad()
        mask = [data[2].view(len(data[2]), -1)]
        output, other_outputs = model(data)
        other_losses = other_outputs['reg'] if 'reg' in other_outputs else []
        alpha = other_outputs['alpha'] if 'alpha' in other_outputs else []
        mask.extend(other_outputs['mask'] if 'mask' in other_outputs else [])
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
        update_attn(data, alpha, alpha_pred, alpha_GT, mask)
        acc = 100. * correct / n_samples  # average over all examples in the dataset
        train_loss_avg  = train_loss / (batch_idx + 1)

        if log and ((batch_idx > 0 and batch_idx % args.log_interval == 0) or batch_idx == len(train_loader) - 1):
            print('Train set (epoch {}): [{}/{} ({:.0f}%)]\tLoss: {:.4f} (avg: {:.4f}), other losses: {}\tAcc metric: {}/{} ({:.2f}%)\t AttnAUC: {}\t avg sec/iter: {:.4f}'.format(
                epoch, n_samples, len(train_loader.dataset), 100. * n_samples / len(train_loader.dataset),
                loss_item, train_loss_avg, ['%.4f' % l.item() for l in other_losses],
                correct, n_samples, acc, ['%.2f' % a for a in attn_AUC(alpha_GT, alpha_pred)],
                time_iter / (batch_idx + 1)))

    assert n_samples == len(train_loader.dataset), (n_samples, len(train_loader.dataset))

    return train_loss, acc


def test(model, test_loader, epoch, loss_fn, split, args, feature_stats=None, noises=None,
         img_noise_level=None, eval_attn=False, alpha_WS_name=''):
    model.eval()
    n_samples, correct, test_loss = 0, 0, 0
    pred, targets, N_nodes = [], [], []
    start = time.time()
    alpha_pred, alpha_GT = {}, {}
    if eval_attn:
        alpha_pred[0] = []
        print('testing with evaluation of attention: takes longer time')
    if args.debug:
        debug_data = {}

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data_to_device(data, args.device)
            if feature_stats is not None:
                assert feature_stats[0].shape[2] == feature_stats[1].shape[2] == data[0].shape[2], \
                    (feature_stats[0].shape, feature_stats[1].shape, data[0].shape)
                data[0] = (data[0] - feature_stats[0]) / feature_stats[1]
            if batch_idx == 0 and epoch <= 1:
                sanity_check(model, data)

            if noises is not None:
                noise = noises[n_samples:n_samples + len(data[0])].to(args.device) * img_noise_level
                if len(noise.shape) == 2:
                    noise = noise.unsqueeze(2)
                data[0][:, :, :3] = data[0][:, :, :3] + noise

            mask = [data[2].view(len(data[2]), -1)]
            N_nodes.append(data[4]['N_nodes'].detach())
            targets.append(data[3].detach())
            output, other_outputs = model(data)
            other_losses = other_outputs['reg'] if 'reg' in other_outputs else []
            alpha = other_outputs['alpha'] if 'alpha' in other_outputs else []
            mask.extend(other_outputs['mask'] if 'mask' in other_outputs else [])
            if args.debug:
                for key in other_outputs:
                    if key.find('debug') >= 0:
                        if key not in debug_data:
                            debug_data[key] = []
                        debug_data[key].append([d.data.cpu().numpy() for d in other_outputs[key]])
            if args.torch.find('1.') == 0:
                loss = loss_fn(output, data[3], reduction='sum')
            else:
                loss = loss_fn(output, data[3], reduce=False).sum()
            for l in other_losses:
                loss += l
            test_loss += loss.item()
            pred.append(output.detach())


            update_attn(data, alpha, alpha_pred, alpha_GT, mask)
            if eval_attn:
                assert len(alpha) == 0, ('invalid mode, eval_attn should be false for this type of pooling')
                alpha_pred[0].extend(attn_heatmaps(model, args.device, data, output.data, test_loader.batch_size, constant_mask=args.dataset=='mnist'))

            n_samples += len(data[0])
            if eval_attn and (n_samples % 100 == 0 or n_samples == len(test_loader.dataset)):
                print('{}/{} samples processed'.format(n_samples, len(test_loader.dataset)))

    assert n_samples == len(test_loader.dataset), (n_samples, len(test_loader.dataset))

    pred = torch.cat(pred)
    targets = torch.cat(targets)
    N_nodes = torch.cat(N_nodes)
    if args.dataset.find('colors') >= 0:
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=25)
        if pred.shape[0] > 2500:
            correct += count_correct(pred[2500:5000], targets[2500:5000], N_nodes=N_nodes[2500:5000], N_nodes_min=26, N_nodes_max=200)
            correct += count_correct(pred[5000:], targets[5000:], N_nodes=N_nodes[5000:], N_nodes_min=26, N_nodes_max=200)
    elif args.dataset == 'triangles':
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=25)
        if pred.shape[0] > 5000:
            correct += count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=26, N_nodes_max=100)
    else:
        correct = count_correct(pred, targets, N_nodes=N_nodes, N_nodes_min=0, N_nodes_max=1e5)

    time_iter = time.time() - start

    test_loss_avg = test_loss / n_samples
    acc = 100. * correct / n_samples  # average over all examples in the dataset
    print('{} set (epoch {}): Avg loss: {:.4f}, Acc metric: {}/{} ({:.2f}%)\t AttnAUC: {}\t avg sec/iter: {:.4f}\n'.format(
        split.capitalize(), epoch, test_loss_avg, correct, n_samples, acc,
        ['%.2f' % a for a in attn_AUC(alpha_GT, alpha_pred)], time_iter / (batch_idx + 1)))

    if args.debug:
        for key in debug_data:
            for layer in range(len(debug_data[key][0])):
                print('{} (layer={}): {:.5f}'.format(key, layer, np.mean([d[layer] for d in debug_data[key]])))

    if eval_attn:
        alpha_pred = alpha_pred[0]
        if args.results in [None, 'None', ''] or alpha_WS_name == '':
            print('skip saving alpha values, invalid results dir (%s) or alpha_WS_name (%s)' % (args.results, alpha_WS_name))
        else:
            file_path = pjoin(args.results, '%s_alpha_WS_%s_seed%d_%s.pkl' % (args.dataset, split, args.seed, alpha_WS_name))
            if os.path.isfile(file_path):
                print('WARNING: file %s exists and will be overwritten' % file_path)
            with open(file_path, 'wb') as f:
                pickle.dump(alpha_pred, f, protocol=2)

    return test_loss, acc, alpha_pred, pred


def update_attn(data, alpha, alpha_pred, alpha_GT, mask):
    key = 'node_attn_eval'
    for layer in range(len(mask)):
        mask[layer] = mask[layer].data.cpu().numpy() > 0
    if key in data[4]:
        if not isinstance(data[4][key], list):
            data[4][key] = [data[4][key]]
        for layer in range(len(data[4][key])):
            if layer not in alpha_GT:
                alpha_GT[layer] = []
            # print(key, layer, len(data[4][key]), len(mask))
            alpha_GT[layer].extend(masked_alpha(data[4][key][layer].data.cpu().numpy(), mask[layer]))
    for layer in range(len(alpha)):
        if layer not in alpha_pred:
            alpha_pred[layer] = []
        alpha_pred[layer].extend(masked_alpha(alpha[layer].data.cpu().numpy(), mask[layer]))


def masked_alpha(alpha, mask):
    alpha_lst = []
    for i in range(len(alpha)):
        # print('gt', len(alpha), alpha[i].shape, mask[i].shape, alpha[i][mask[i] > 0].shape, mask[i].sum(), mask[i].min(), mask[i].max(), mask[i].dtype)
        alpha_lst.append(alpha[i][mask[i]])
    return alpha_lst


def attn_heatmaps(model, device, data, output_org, batch_size=1, constant_mask=False):
    labels = torch.argmax(output_org, dim=1)
    B, N_nodes_max, C  = data[0].shape  # N_nodes should be the same in the batch
    alpha_WS = []
    if N_nodes_max > 1000:
        print('WARNING: graph is too large (%d nodes) and not supported by this function (evaluation will be incorrect for graphs in this batch).' % N_nodes_max)
        for b in range(B):
            n = data[2][b].sum().item()
            alpha_WS.append(np.zeros((1, n)) + 1. / n)
        return alpha_WS

    if constant_mask:
        mask = torch.ones(N_nodes_max, N_nodes_max - 1).to(device)

    # Indices of nodes such that in each row one index (i.e. one node) is removed
    node_ids = torch.arange(start=0, end=N_nodes_max, device=device).view(1, -1).repeat(N_nodes_max, 1)
    node_ids[np.diag_indices(N_nodes_max, 2)] = -1
    node_ids = node_ids[node_ids >= 0].view(N_nodes_max, N_nodes_max - 1).long()

    with torch.no_grad():
        for b in range(B):
            x = torch.gather(data[0][b].unsqueeze(0).expand(N_nodes_max, -1, -1), dim=1, index=node_ids.unsqueeze(2).expand(-1, -1, C))
            if not constant_mask:
                mask = torch.gather(data[2][b].unsqueeze(0).expand(N_nodes_max, -1), dim=1, index=node_ids)
            A = torch.gather(data[1][b].unsqueeze(0).expand(N_nodes_max, -1, -1), dim=1, index=node_ids.unsqueeze(2).expand(-1, -1, N_nodes_max))
            A = torch.gather(A, dim=2, index=node_ids.unsqueeze(1).expand(-1, N_nodes_max - 1, -1))
            output = torch.zeros(N_nodes_max).to(device)
            n_chunks = int(np.ceil(N_nodes_max / float(batch_size)))
            for i in range(n_chunks):
                idx = np.arange(i * batch_size, (i + 1) * batch_size) if i < n_chunks - 1 else np.arange(i * batch_size, N_nodes_max)
                output[idx] = model([x[idx], A[idx], mask[idx], None, {}])[0][:, labels[b]].data

            alpha = torch.abs(output - output_org[b, labels[b]]).view(1, N_nodes_max) #* mask_org[b].view(1, N_nodes_max)
            if not constant_mask:
                alpha = alpha[data[2][b].view(1, N_nodes_max)]
            alpha_WS.append(normalize(alpha).data.cpu().numpy())

    return alpha_WS


def save_checkpoint(model, scheduler, optimizer, args, epoch):
    if args.results in [None, 'None']:
        print('skip saving checkpoint, invalid results dir: %s' % args.results)
        return
    file_path = '%s/checkpoint_%s_%s_epoch%d_seed%07d.pth.tar' % (args.results, args.dataset, args.experiment_ID, epoch, args.seed)
    try:
        print('saving the model to %s' % file_path)
        state = {
            'epoch': epoch,
            'args': args,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if os.path.isfile(file_path):
            print('WARNING: file %s exists and will be overwritten' % file_path)
        torch.save(state, file_path)
    except Exception as e:
        print('error saving the model', e)


def load_checkpoint(model, optimizer, scheduler, file_path):
    print('loading the model from %s' % file_path)
    state = torch.load(file_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    print('loading from epoch %d done' % state['epoch'])
    return state['epoch'] + 1  # +1 because we already finished training for this epoch


def create_model_optimizer(in_features, out_features, pool, kl_weight, args, scale=None, init=None, n_hidden_attn=None):
    set_seed(args.seed, seed_data=None)
    model = ChebyGIN(in_features=in_features,
                     out_features=out_features,
                     filters=args.filters,
                     K=args.filter_scale,
                     n_hidden=args.n_hidden,
                     aggregation=args.aggregation,
                     dropout=args.dropout,
                     readout=args.readout,
                     pool=pool,
                     pool_arch=args.pool_arch if n_hidden_attn in [None, 0] else args.pool_arch[:2] + ['%d' % n_hidden_attn],
                     large_graph=args.dataset.lower() == 'mnist',
                     kl_weight=float(kl_weight),
                     init=args.init if init is None else init,
                     scale=args.scale if scale is None else scale,
                     debug=args.debug)
    print(model)
    # Compute the total number of trainable parameters
    print('model capacity: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=0.1)
    epoch = 1
    if args.resume not in [None, 'None']:
        epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
        if epoch < args.epochs + 1:
            print('resuming training for epoch %d' % epoch)

    model.to(args.device)

    return epoch, model, optimizer, scheduler


def single_job(fold, datareader, args, collate_fn, loss_fn, pool, kl_weight, feature_stats, val_acc,
               scale=None, init=None, n_hidden_attn=None):

    set_seed(args.seed, seed_data=None)

    wsup = args.pool[1] == 'sup'
    train_loader = DataLoader(GraphData(datareader, fold, 'train'), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.threads, collate_fn=collate_fn)
    val_loader = DataLoader(GraphData(datareader, fold, 'val'), batch_size=args.test_batch_size, shuffle=False,
                            num_workers=args.threads, collate_fn=collate_fn)
    start_epoch, model, optimizer, scheduler = create_model_optimizer(train_loader.dataset.num_features,
                                                                      train_loader.dataset.num_classes,
                                                                      None if wsup else pool, kl_weight, args,
                                                                      scale=scale, init=init, n_hidden_attn=n_hidden_attn)

    for epoch in range(start_epoch, args.epochs + 1):
        scheduler.step()
        train(model, train_loader, optimizer, epoch, args, loss_fn, feature_stats, log=False)

    if wsup:
        train_loader_test = DataLoader(GraphData(datareader, fold, 'train'), batch_size=args.test_batch_size, shuffle=False,
                                       num_workers=args.threads, collate_fn=collate_fn)
        train_loss, train_acc, attn_WS = test(model, train_loader_test, epoch, loss_fn, 'train', args, feature_stats, eval_attn=True)[:3]  # test_loss, acc, alpha_pred, pred
        train_loader = DataLoader(GraphData(datareader, fold, 'train', attn_labels=attn_WS),
                                  batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.threads, collate_fn=collate_fn)
        val_loader = DataLoader(GraphData(datareader, fold, 'val'), batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.threads, collate_fn=collate_fn)
        start_epoch, model, optimizer, scheduler = create_model_optimizer(train_loader.dataset.num_features,
                                                                          train_loader.dataset.num_classes,
                                                                          pool, kl_weight, args,
                                                                          scale=scale, init=init, n_hidden_attn=n_hidden_attn)
        for epoch in range(start_epoch, args.epochs + 1):
            scheduler.step()
            train(model, train_loader, optimizer, epoch, args, loss_fn, feature_stats, log=False)

    acc = test(model, val_loader, epoch, loss_fn, 'val', args, feature_stats)[1]

    val_acc[fold] = acc


def cross_validation(datareader, args, collate_fn, loss_fn, pool, kl_weight, feature_stats, n_hidden_attn=None, folds=10, threads=5):
    print('%d-fold cross-validation' % folds)
    manager = multiprocessing.Manager()
    val_acc = manager.dict()
    assert threads <= folds, (threads, folds)
    n_it = int(np.ceil(float(folds) / threads))
    for i in range(n_it):
        processes = []
        if threads <= 1:
            single_job(i * threads, datareader, args, collate_fn, loss_fn, pool, kl_weight,
                       feature_stats, val_acc, scale=args.scale, init=args.init, n_hidden_attn=n_hidden_attn)
        else:
            for fold in range(threads):
                p = mp.Process(target=single_job, args=(i * threads + fold, datareader, args, collate_fn, loss_fn, pool, kl_weight,
                                                        feature_stats, val_acc, args.scale, args.init, n_hidden_attn))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

    print(val_acc)
    val_acc = list(val_acc.values())
    print('average and std over {} folds: {} +- {}'.format(folds, np.mean(val_acc), np.std(val_acc)))
    metric = np.mean(val_acc) - np.std(val_acc)
    print('metric: avg acc - std: {}'.format(metric))
    return metric
