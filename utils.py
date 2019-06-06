import numpy as np
import pandas as pd
import cv2
import os
import torch
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def attn_heatmaps(model, device, data, output_org, batch_size=1, constant_mask=False):
    labels = torch.argmax(output_org, dim=1)
    B, N_nodes, C  = data[0].shape  # N_nodes should be the same in the batch
    mask_org = data[2].float()  # B,N_nodes
    alpha_WS = []
    if constant_mask:
        mask = torch.ones(N_nodes, N_nodes - 1).to(device)

    # Indices of nodes such that in each row one index (i.e. one node) is removed
    node_ids = torch.arange(start=0, end=N_nodes, device=device).view(1, -1).repeat(N_nodes, 1)
    node_ids[np.diag_indices(N_nodes, 2)] = -1
    node_ids = node_ids[node_ids >= 0].view(N_nodes, N_nodes - 1)

    with torch.no_grad():
        for b in range(B):
            x = torch.gather(data[0][b].unsqueeze(0).expand(N_nodes, -1, -1), dim=1, index=node_ids.unsqueeze(2).expand(-1, -1, C))
            if not constant_mask:
                mask = torch.gather(data[2][b].unsqueeze(0).expand(N_nodes, -1), dim=1, index=node_ids)
            A = torch.gather(data[1][b].unsqueeze(0).expand(N_nodes, -1, -1), dim=1, index=node_ids.unsqueeze(2).expand(-1, -1, N_nodes))
            A = torch.gather(A, dim=2, index=node_ids.unsqueeze(1).expand(-1, N_nodes - 1, -1))
            output = torch.zeros(N_nodes).to(device)
            n_chunks = int(np.ceil(N_nodes / float(batch_size)))
            for i in range(n_chunks):
                idx = np.arange(i * batch_size, (i + 1) * batch_size) if i < n_chunks - 1 else np.arange(i * batch_size, N_nodes)
                output[idx] = model([x[idx], A[idx], mask[idx], None, {}])[0][:, labels[b]].data

            alpha = torch.abs(output - output_org[b, labels[b]]).view(1, N_nodes) * mask_org[b].view(1, N_nodes)
            alpha_WS.append(alpha / (alpha.sum() + 1e-7))

    alpha_WS = torch.cat(alpha_WS, dim=0)
    return alpha_WS


def load_save_noise(f, noise_shape):
    if os.path.isfile(f):
        print('loading noise from %s' % f)
        noises = torch.load(f)
    else:
        noises = torch.randn(noise_shape, dtype=torch.float)
        # np.save(f, noises.numpy())
        torch.save(noises, f)
    return noises


def list_to_torch(data):
    for i in range(len(data)):
        if isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data


def data_to_device(data, device):
    if isinstance(data, dict):
        keys = list(data.keys())
    else:
        keys = range(len(data))
    for i in keys:
        if isinstance(data[i], list) or isinstance(data[i], dict):
            data[i] = data_to_device(data[i], device)
        else:
            if isinstance(data[i], torch.Tensor):
                try:
                    data[i] = data[i].to(device)
                except:
                    print('error', i, data[i], type(data[i]))
                    raise
    return data


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


def attn_AUC(alpa_GT, alpha):
    auc = []
    alpa_GT = np.concatenate(alpa_GT).flatten() > 0
    for layer in alpha:
        auc.append(100 * roc_auc_score(y_true=alpa_GT, y_score=np.concatenate(alpha[layer]).flatten()))
    return auc


def mse_loss(target, output, reduction='mean'):
    if reduction == 'mean':
        return torch.mean((target.float().squeeze() - output.float().squeeze()) ** 2)
    elif reduction == 'sum':
        return torch.sum((target.float().squeeze() - output.float().squeeze()) ** 2)
    else:
        NotImplementedError(reduction)


def shuffle_nodes(batch):
    x, A, mask, labels, params_dict = batch
    for b in range(x.shape[0]):
        idx = np.random.permutation(x.shape[1])
        x[b] = x[b, idx]
        A[b] = A[b, :, idx][idx, :]
        mask[b] = mask[b, idx]
        if 'node_attn' in params_dict:
            params_dict['node_attn'][b] = params_dict['node_attn'][b, idx]
    return [x, A, mask, labels, params_dict]


def sanity_check(model, data):
    with torch.no_grad():
        output1, other_losses1, _ = model(data)
        output2, other_losses2, _ = model(shuffle_nodes(data))
        assert torch.allclose(output1, output2, rtol=1e-02, atol=1e-03), torch.norm(output1 - output2)
        for l1, l2 in zip(other_losses1, other_losses2):
            assert torch.allclose(l1, l2), (l1, l2)
    print('model is checked for nodes shuffling')


def compute_feature_stats(model, train_loader, device, n_batches=100):
    print('computing mean and std of input features')
    model.eval()
    x = []
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            x.append(data[0].data) # B,N,F
            if batch_idx > n_batches:
                break
    x = torch.cat(x, dim=1)  # M,N,F
    print('features shape loaded', x.shape)

    mn = x.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
    sd = x.std(dim=0, keepdim=True).std(dim=1, keepdim=True)
    print('mn', mn.data.cpu().numpy())
    print('std', sd.data.cpu().numpy())
    sd[sd < 1e-2] = 1  # to prevent dividing by a small number
    print('corrected (non zeros) std', sd.data.cpu().numpy())
    mn = mn.to(device)
    sd = sd.to(device)
    return mn, sd


def copy_data(data, idx):
    data_new = {}
    for key in data:
        if key == 'Max_degree':
            data_new[key] = data[key]
            print(key, data_new[key])
        else:
            data_new[key] = copy.deepcopy([data[key][i] for i in idx])
            if key in ['graph_labels', 'N_edges']:
                data_new[key] = np.array(data_new[key], np.int32)
            print(key, len(data_new[key]))

    return data_new


def concat_data(data):
    data_new = {}
    for key in data[0]:
        if key == 'Max_degree':
            data_new[key] = np.max(np.array([ d[key] for d in data ]))
            print(key, data_new[key])
        else:
            if key in ['graph_labels', 'N_edges']:
                data_new[key] = np.concatenate([ d[key] for d in data ])
            else:
                lst = []
                for d in data:
                    lst.extend(d[key])
                data_new[key] = lst
            print(key, len(data_new[key]))

    return data_new