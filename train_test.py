import time
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from chebygin import *
from utils import *
from graphdata import *


def train(model, train_loader, optimizer, epoch, args, loss_fn, feature_stats=None):
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

        if (batch_idx > 0 and batch_idx % args.log_interval == 0) or batch_idx == len(train_loader) - 1:
            print('Train set (epoch {}): [{}/{} ({:.0f}%)]\tLoss: {:.4f} (avg: {:.4f}), other losses: {}\tAcc metric: {}/{} ({:.2f}%)\t AttnAUC: {}\t avg sec/iter: {:.4f}'.format(
                epoch, n_samples, len(train_loader.dataset), 100. * n_samples / len(train_loader.dataset),
                loss_item, train_loss_avg, ['%.4f' % l.item() for l in other_losses],
                correct, n_samples, acc, ['%.2f' % a for a in attn_AUC(alpha_GT, alpha_pred)],
                time_iter / (batch_idx + 1)))

    print('\n')
    assert n_samples == len(train_loader.dataset), (n_samples, len(train_loader.dataset))

    return train_loss, acc


def test(model, test_loader, epoch, loss_fn, split, args, feature_stats=None, noises=None, img_noise_level=None, eval_attn=False, alpha_WS_name=''):
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
            loss = loss_fn(output, data[3], reduction='sum')
            for l in other_losses:
                loss += l
            test_loss += loss.item()
            pred.append(output.detach())


            update_attn(data, alpha, alpha_pred, alpha_GT, mask)
            if eval_attn:
                assert len(alpha) == 0, ('invalid mode, eval_attn should be false for this type of pooling')
                alpha_pred[0].extend(attn_heatmaps(model, args.device, data, output.data, test_loader.batch_size, constant_mask=args.dataset=='mnist'))

            n_samples += len(data[0])
            if eval_attn:
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

    return test_loss, acc, alpha_pred


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
    node_ids = node_ids[node_ids >= 0].view(N_nodes_max, N_nodes_max - 1)

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


def create_model_optimizer(in_features, out_features, pool, kl_weight, args):
    model = ChebyGIN(in_features=in_features,
                     out_features=out_features,
                     filters=args.filters,
                     K=args.filter_scale,
                     n_hidden=args.n_hidden,
                     aggregation=args.aggregation,
                     dropout=args.dropout,
                     readout=args.readout,
                     pool=pool,
                     pool_arch=args.pool_arch,
                     large_graph=args.dataset=='mnist',
                     kl_weight=float(kl_weight),
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


def cross_validation(datareader, args, collate_fn, loss_fn, pool, kl_weight, feature_stats, folds=10):
    print('%d-fold cross-validation' % folds)
    val_acc = []
    wsup = args.pool[1] == 'sup'
    for fold in range(folds):
        train_loader = DataLoader(GraphData(datareader, fold, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=args.threads, collate_fn=collate_fn)
        val_loader = DataLoader(GraphData(datareader, fold, 'val'), batch_size=args.test_batch_size, shuffle=False, num_workers=args.threads, collate_fn=collate_fn)
        start_epoch, model, optimizer, scheduler = create_model_optimizer(train_loader.dataset.num_features,
                                                                          train_loader.dataset.num_classes,
                                                                          None if wsup else pool, kl_weight, args)
        for epoch in range(start_epoch, args.epochs + 1):
            scheduler.step()
            train(model, train_loader, optimizer, epoch, args, loss_fn, feature_stats)

        if wsup:
            train_loader_test = DataLoader(GraphData(datareader, fold, 'train'), batch_size=args.test_batch_size, shuffle=False,
                                           num_workers=args.threads, collate_fn=collate_fn)
            train_loss, train_acc, attn_WS = test(model, train_loader_test, epoch, loss_fn, 'train', args, feature_stats, eval_attn=True)
            train_loader = DataLoader(GraphData(datareader, fold, 'train', attn_labels=attn_WS),
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.threads, collate_fn=collate_fn)
            val_loader = DataLoader(GraphData(datareader, fold, 'val'), batch_size=args.test_batch_size, shuffle=False,
                                    num_workers=args.threads, collate_fn=collate_fn)
            start_epoch, model, optimizer, scheduler = create_model_optimizer(train_loader.dataset.num_features,
                                                                              train_loader.dataset.num_classes,
                                                                              pool, kl_weight, args)
            for epoch in range(start_epoch, args.epochs + 1):
                scheduler.step()
                train(model, train_loader, optimizer, epoch, args, loss_fn, feature_stats)

        acc = test(model, val_loader, epoch, loss_fn, 'val', args, feature_stats)[1]
        val_acc.append(acc)

    print('average and std over {} folds: {} +- {}'.format(folds, np.mean(val_acc), np.std(val_acc)))

    return np.mean(val_acc)