#import matplotlib.pyplot as plt  # uncomment to plot histograms
import os
import numpy as np
import pickle
import argparse
import networkx as nx
import datetime
import copy

# Colors and Triangles datasets can be downloaded at https://gofile.io/?c=zOiltT

# In this file we provide a script to generate data for our Colors dataset:
# Running 'python data_generate.py' will generate training and test .pkl files in a local directory.
# For triangles we generate and store files in a similar way and will release the code upon acceptance.

# To load data we use the following code:
# with open('random_graphs_colors_dim3_train.pkl', 'rb') as f:
#     Adj_matrices, GT_attn, graph_labels, node_features, n_edges = pickle.load(f)
# Adj_matrices - list of adjacency matrices
# GT_attn - ground truth attention for nodes
# graph_labels - labels of graphs (number of green nodes or triangles)
# node_features - node features for each graph
# n_edges - number of edges in each graph (used for statistics)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic graph datasets')
    parser.add_argument('-D', '--dataset', type=str, default='colors', choices=['colors', 'triangles'])
    parser.add_argument('-o', '--out_dir', type=str, default='./data', help='path where to save superpixels')
    parser.add_argument('--N_train', type=int, default=500, help='number of training graphs')
    parser.add_argument('--N_test', type=int, default=2500, help='number of graphs in each test subset')
    parser.add_argument('--label_min', type=int, default=0,
                        help='smallest label value for a graph (i.e. smallest number of green nodes)')
    parser.add_argument('--label_max', type=int, default=10,
                        help='largest label value for a graph (i.e. largest number of green nodes)')
    parser.add_argument('--N_min', type=int, default=4, help='minimum number of nodes')
    parser.add_argument('--N_max', type=int, default=200, help='maximum number of nodes (default: 200 for Colors and 100 for Triangles')
    parser.add_argument('--N_max_train', type=int, default=25, help='maximum number of nodes in the training set')
    parser.add_argument('--dim', type=int, default=3, help='node feature dimensionality')
    parser.add_argument('--green_ch_index', type=int, default=1,
                        help='index of non-zero value in a one-hot node feature vector, '
                             'i.e. [0, 1, 0] in case green_channel_index=1 and dim=3')
    parser.add_argument('--seed', type=int, default=11, help='seed for shuffling nodes')
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args


def check_graph_duplicates(Adj_matrices, node_features, triangles=False, colors=False):
    n_graphs = len(Adj_matrices)
    print('check for duplicates for %d graphs' % n_graphs)
    n_duplicates = 0
    for i in range(n_graphs):
        assert Adj_matrices[i].shape[0] == node_features[i].shape[0], (
            'invalid data', i, Adj_matrices[i].shape[0], node_features[i].shape[0])
        for j in range(i + 1, n_graphs):
            if Adj_matrices[i].shape[0] == Adj_matrices[j].shape[0]:
                if np.allclose(Adj_matrices[i], Adj_matrices[j]):  # adjacency matrices are the same
                    # for Colors graphs are not considered duplicates if they have the same adjacency matrix,
                    # but different node features
                    if triangles or (colors and np.allclose(node_features[i], node_features[j])):
                        n_duplicates += 1
                        print('duplicates %d/%d' % (n_duplicates, n_graphs * (n_graphs - 1) / 2))
    if n_duplicates > 0:
        raise ValueError('%d duplicates found in the dataset' % n_duplicates)

    print('no duplicated graphs')


def get_node_features_Colors(N_nodes, N_green, dim, green_ch_index=1, new_colors=False):
    node_features = np.zeros((N_nodes, dim))

    # Generate indices for non-zero values,
    # so that the total number of nodes with features having value 1 in the green_ch_index equals N_green
    idx_not_green = rnd.randint(0, dim - 1, size=N_nodes - N_green)  # for dim=3 generate values 0,1 for non-green nodes
    idx_non_zero = np.concatenate((idx_not_green, np.zeros(N_green, np.int) + dim - 1))  # make green_ch_index=2 temporary
    idx_non_zero_cp = idx_non_zero.copy()
    idx_non_zero[idx_non_zero_cp == dim - 1] = green_ch_index  # set idx_non_zero=1 for green nodes
    idx_non_zero[idx_non_zero_cp == green_ch_index] = dim - 1  # set idx_non_zero=2 for those nodes that were green temporary
    rnd.shuffle(idx_non_zero)  # shuffle nodes
    node_features[np.arange(N_nodes), idx_non_zero] = 1

    if new_colors:
        for ind in np.where(idx_non_zero != green_ch_index)[0]:  # for non-green nodes
            node_features[ind] = rnd.randint(0, 2, size=dim)
            node_features[ind, green_ch_index] = 0  # set value at green_ch_index to 0 to avoid confusion with green nodes

    label = np.sum((np.sum(node_features, 1) == node_features[:, green_ch_index]) & (node_features[:, green_ch_index] == 1))

    gt_attn = (idx_non_zero == green_ch_index).reshape(-1, 1)
    label2 = np.sum(gt_attn)
    assert N_green == label == label2, ('invalid node features', N_green, label, label2)
    return node_features, idx_non_zero, gt_attn


def generate_graphs_Colors(N_graphs, N_min, N_max, dim, args, rnd, new_colors=False):
    Adj_matrices, node_features, GT_attn, graph_labels, N_edges = [], [], [], [], []
    n_labels = args.label_max - args.label_min + 1
    n_graphs_per_shape = int(np.ceil(N_graphs / (N_max - N_min + 1) / n_labels) * n_labels)
    for n_nodes in np.array(range(N_min, N_max + 1)):
        c = 0
        while True:
            labels = np.arange(args.label_min, n_labels)
            labels = labels[labels <= n_nodes]
            rnd.shuffle(labels)
            for lbl in labels:
                features, idx_non_zero, gt_attn = get_node_features_Colors(N_nodes=n_nodes,
                                                                           N_green=lbl,
                                                                           dim=dim,
                                                                           green_ch_index=args.green_ch_index,
                                                                           new_colors=new_colors)
                n_edges = int((rnd.rand() + 1) * n_nodes)
                A = nx.to_numpy_array(nx.gnm_random_graph(n_nodes, n_edges))
                add = True
                for k in range(len(Adj_matrices)):
                    if A.shape[0] == Adj_matrices[k].shape[0] and np.allclose(A, Adj_matrices[k]):
                        if np.allclose(node_features[k], features):
                            add = False
                            break
                if add:
                    Adj_matrices.append(A.astype(np.bool))  # binary adjacency matrix
                    graph_labels.append(lbl)
                    node_features.append(features.astype(np.bool))  # binary features
                    GT_attn.append(gt_attn)  # binary GT attention
                    N_edges.append(n_edges)
                    c += 1
                    if c >= n_graphs_per_shape:
                        break
            if c >= n_graphs_per_shape:
                break
    graph_labels = np.array(graph_labels).astype(np.int32)
    N_edges = np.array(N_edges).astype(np.int32)
    print(N_graphs, len(graph_labels))

    return {'Adj_matrices': Adj_matrices,
            'GT_attn': GT_attn,  # not normalized to sum=1
            'graph_labels': graph_labels,
            'node_features': node_features,
            'N_edges': N_edges}

def copy_data(data, idx):
    data_new = {}
    for key in data:
        data_new[key] = copy.deepcopy([data[key][i] for i in idx])
        if key in ['graph_labels', 'N_edges']:
            data_new[key] = np.array(data_new[key]).astype(np.int32)
        print(key, len(data_new[key]))

    return data_new

def concat_data(data1, data2, data3):
    data_new = {}
    for key in data1:
        if key in ['graph_labels', 'N_edges']:
            data_new[key] = np.concatenate((data1[key], data2[key], data3[key]))
        else:
            data_new[key] = data1[key] + data2[key] + data3[key]
        print(key, len(data_new[key]))

    return data_new

if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)

    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    rnd = np.random.RandomState(args.seed)

    data = generate_graphs_Colors(args.N_train + args.N_test, args.N_min, args.N_max_train, args.dim, args, rnd)
    idx = rnd.permutation(len(data['graph_labels']))
    idx_train = idx[:args.N_train]
    data_train = copy_data(data, idx_train)
    print('train orig: %d graphs' % len(idx_train))
    for i in np.unique(data_train['graph_labels']):
        print('train orig: label=%d, %d graphs' % (i, np.sum(data_train['graph_labels'] == i)))
    idx_test = idx[args.N_train: args.N_train + args.N_test]
    data_test = copy_data(data, idx_test)
    print('test orig: %d graphs' % len(idx_test))
    for i in np.unique(data_test['graph_labels']):
        print('test orig: label=%d, %d graphs' % (i, np.sum(data_test['graph_labels'] == i)))

    data_large = generate_graphs_Colors(args.N_test, args.N_max_train + 1, args.N_max, args.dim, args, rnd)
    idx_test = rnd.permutation(len(data_large['graph_labels']))[:args.N_test]
    data_test_large = copy_data(data_large, idx_test)
    print('test large: %d graphs' % len(idx_test))
    for i in np.unique(data_test_large['graph_labels']):
        print('test large: label=%d, %d graphs' % (i, np.sum(data_test_large['graph_labels'] == i)))

    data_large_c = generate_graphs_Colors(args.N_test, args.N_max_train + 1, args.N_max, args.dim + 1, args, rnd, new_colors=True)

    idx_test = rnd.permutation(len(data_large_c['graph_labels']))[:args.N_test]
    data_test_large_c = copy_data(data_large_c, idx_test)
    print('test large c: %d graphs' % len(idx_test))
    for i in np.unique(data_test_large_c['graph_labels']):
        print('test large c: label=%d, %d graphs' % (i, np.sum(data_test_large_c['graph_labels'] == i)))


    check_graph_duplicates(data_train['Adj_matrices'] + data_test['Adj_matrices'] + data_test_large['Adj_matrices'] +
                           data_test_large_c['Adj_matrices'],
                           data_train['node_features'] + data_test['node_features'] + data_test_large['node_features'] +
                           data_test_large_c['node_features'])


    with open('%s/random_graphs_colors_dim%d_train.pkl' % (args.out_dir, args.dim), 'wb') as f:
        pickle.dump(data_train, f, protocol=2)

    data_test = concat_data(data_test, data_test_large, data_test_large_c)
    with open('%s/random_graphs_colors_dim%d_test.pkl' % (args.out_dir, args.dim), 'wb') as f:
        pickle.dump(data_test, f, protocol=2)


    print('done in {}'.format(datetime.datetime.now() - dt))






