import numpy as np
import os
from os.path import join as pjoin
import pickle
import copy
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import utils


def collate_batch_images(batch):
    '''
    Creates a batch of same size graphs by zero-padding node features and adjacency matrices up to
    the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
    :param batch: batch in the PyTorch Geometric format or [node_features*batch_size, A*batch_size, label*batch_size]
    :return: [node_features, A, graph_support, N_nodes, label]
    '''

    B = len(batch)
    C, H, W = batch[0][0].shape
    col, row = np.meshgrid(np.arange(W), np.arange(H))
    coord = np.stack((col, row), axis=2)  # 28,28,2
    coord = coord.reshape(-1, 2) / H
    dist = cdist(coord, coord)
    sigma = 0.1 * np.pi
    A = np.exp(- dist / sigma ** 2)
    A[np.diag_indices_from(A)] = 0
    A = torch.from_numpy(A).float().unsqueeze(0).expand(B, -1, -1)
    x = torch.stack([batch[b][0].permute(0, 2, 1).contiguous().view(C, H*W).t() for b in range(B)]).float()
    mask = torch.ones(B, H*W, dtype=torch.uint8)
    params_dict = {'N_nodes': torch.zeros(B, dtype=torch.long) + H*W,
                   'node_attn': (x > 0).view(B, H*W)}

    coord = torch.from_numpy(coord).float().unsqueeze(0).expand(B, -1, -1)
    # print(x.shape, x.min(), x.max(), coord.min(), coord.max())
    x = torch.cat((x, coord), dim=2)

    labels = torch.stack([batch[b][1] for b in range(B)])

    return [x, A, mask, labels, params_dict]


def collate_batch(batch):
    '''
    Creates a batch of same size graphs by zero-padding node features and adjacency matrices up to
    the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
    :param batch: batch in the PyTorch Geometric format or [node_features*batch_size, A*batch_size, label*batch_size]
    :return: [node_features, A, graph_support, N_nodes, label]
    '''

    B = len(batch)
    N_nodes = [batch[b][2] for b in range(B)]
    C = batch[0][0].shape[1]
    N_nodes_max = int(np.max(N_nodes))

    mask = torch.zeros(B, N_nodes_max, dtype=torch.uint8)
    A = torch.zeros(B, N_nodes_max, N_nodes_max)
    x = torch.zeros(B, N_nodes_max, C)
    has_attn = len(batch[0]) > 4 and batch[0][4] is not None
    if has_attn:
        node_attn = torch.zeros(B, N_nodes_max)

    for b in range(B):
        x[b, :N_nodes[b]] = batch[b][0]
        A[b, :N_nodes[b], :N_nodes[b]] = batch[b][1]
        mask[b][:N_nodes[b]] = 1  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1
        if has_attn:
            node_attn[b, :N_nodes[b]] = batch[b][4].squeeze()

    N_nodes = torch.from_numpy(np.array(N_nodes)).long()

    params_dict = {'N_nodes': N_nodes}
    if has_attn:
        params_dict.update({'node_attn': node_attn})

    labels = torch.from_numpy(np.array([batch[b][3] for b in range(B)])).long()
    return [x, A, mask, labels, params_dict]


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


def stats(arr):
    return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)


class SyntheticGraphs(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 dataset,
                 split):

        self.is_test = split.lower() == 'test'

        if dataset.find('colors') >= 0:
            dim = int(dataset.split('-')[1])
            data_file = 'random_graphs_colors_dim%d_%s.pkl' % (dim, split)
            is_triangles = False
            self.feature_dim = dim + 1
        if dataset.find('triangles') >= 0:
            data_file = 'random_graphs_triangles_%s.pkl' % split
            is_triangles = True
        else:
            NotImplementedError(dataset)

        with open(pjoin(data_dir, data_file), 'rb') as f:
            data = pickle.load(f)
        for key in data:
            if not isinstance(data[key], list) and not isinstance(data[key], np.ndarray):
                print(split, key, data[key])
            else:
                print(split, key, len(data[key]))

        self.Node_degrees = [np.sum(A, 1).astype(np.int32) for A in data['Adj_matrices']]

        if is_triangles:
            # use one-hot degree features as node features
            self.feature_dim = data['Max_degree'] + 1
            self.node_features = []
            for i in range(len(data['Adj_matrices'])):
                N = data['Adj_matrices'][i].shape[0]
                D_onehot = np.zeros((N, self.feature_dim ))
                D_onehot[np.arange(N), self.Node_degrees[i]] = 1
                self.node_features.append(D_onehot)
        else:
            # Add 1 feature to support new colors at test time
            self.node_features = []
            for i in range(len(data['node_features'])):
                features = data['node_features'][i]
                if features.shape[1] < self.feature_dim:
                    features = np.pad(features, ((0, 0), (0, 1)), 'constant')
                self.node_features.append(features)

        N_nodes = np.array([A.shape[0] for A in data['Adj_matrices']])
        self.Adj_matrices = data['Adj_matrices']
        self.GT_attn = data['GT_attn']
        # Normalizing ground truth attention so that it sums to 1
        for i in range(len(self.GT_attn)):
            self.GT_attn[i] = self.GT_attn[i] / (np.sum(self.GT_attn[i]) + 1e-7)
            #assert np.sum(self.GT_attn[i]) == 1, (i, np.sum(self.GT_attn[i]), self.GT_attn[i])
        self.labels = data['graph_labels'].astype(np.int32)
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)
        R = np.corrcoef(self.labels, N_nodes)[0, 1]

        degrees = []
        for i in range(len(self.Node_degrees)):
            degrees.extend(list(self.Node_degrees[i]))
        degrees = np.array(degrees, np.int32)

        print('N nodes avg/std/min/max: \t{:.2f}/{:.2f}/{:d}/{:d}'.format(*stats(N_nodes)))
        print('N edges avg/std/min/max: \t{:.2f}/{:.2f}/{:d}/{:d}'.format(*stats(data['N_edges'])))
        print('Node degree avg/std/min/max: \t{:.2f}/{:.2f}/{:d}/{:d}'.format(*stats(degrees)))
        print('Node features dim: \t\t%d' % self.feature_dim)
        print('N classes: \t\t\t%d' % self.n_classes)
        print('Correlation of labels with graph size: \t%.2f' % R)
        print('Classes: \t\t\t%s' % str(self.classes))
        for lbl in self.classes:
            idx = self.labels == lbl
            print('Class {}: \t\t\t{} samples, N_nodes: avg/std/min/max: \t{:.2f}/{:.2f}/{:d}/{:d}'.format(lbl, np.sum(idx), *stats(N_nodes[idx])))

    def __len__(self):
        return len(self.Adj_matrices)

    def __getitem__(self, index):
        data = [self.node_features[index],
                self.Adj_matrices[index],
                self.Adj_matrices[index].shape[0],
                self.labels[index],
                self.GT_attn[index]]

        data = utils.list_to_torch(data)  # convert to torch

        return data