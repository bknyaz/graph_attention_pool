#import matplotlib.pyplot as plt  # uncomment to plot histograms
import os
import numpy as np
import pickle
import argparse
import networkx as nx
import datetime
import copy
import multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic graph datasets')
    parser.add_argument('-D', '--dataset', type=str, default='colors', choices=['colors', 'triangles'])
    parser.add_argument('-o', '--out_dir', type=str, default='./data', help='path where to save superpixels')
    parser.add_argument('--N_train', type=int, default=500, help='number of training graphs (500 for colors and 30000 for triangles)')
    parser.add_argument('--N_test', type=int, default=2500, help='number of graphs in each test subset (2500 for colors and 5000 for triangles)')
    parser.add_argument('--label_min', type=int, default=0,
                        help='smallest label value for a graph (i.e. smallest number of green nodes); 1 for triangles')
    parser.add_argument('--label_max', type=int, default=10,
                        help='largest label value for a graph (i.e. largest number of green nodes)')
    parser.add_argument('--N_min', type=int, default=4, help='minimum number of nodes')
    parser.add_argument('--N_max', type=int, default=200, help='maximum number of nodes (default: 200 for colors and 100 for triangles')
    parser.add_argument('--N_max_train', type=int, default=25, help='maximum number of nodes in the training set')
    parser.add_argument('--dim', type=int, default=3, help='node feature dimensionality')
    parser.add_argument('--green_ch_index', type=int, default=1,
                        help='index of non-zero value in a one-hot node feature vector, '
                             'i.e. [0, 1, 0] in case green_channel_index=1 and dim=3')
    parser.add_argument('--seed', type=int, default=11, help='seed for shuffling nodes')
    parser.add_argument('--threads', type=int, default=4, help='only for triangles')
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args


def check_graph_duplicates(Adj_matrices, node_features=None):
    n_graphs = len(Adj_matrices)
    print('check for duplicates for %d graphs' % n_graphs)
    n_duplicates = 0
    for i in range(n_graphs):
        if node_features is not None:
            assert Adj_matrices[i].shape[0] == node_features[i].shape[0], (
                'invalid data', i, Adj_matrices[i].shape[0], node_features[i].shape[0])
        for j in range(i + 1, n_graphs):
            if Adj_matrices[i].shape[0] == Adj_matrices[j].shape[0]:
                if np.allclose(Adj_matrices[i], Adj_matrices[j]):  # adjacency matrices are the same
                    # for Colors graphs are not considered duplicates if they have the same adjacency matrix,
                    # but different node features
                    if node_features is None or np.allclose(node_features[i], node_features[j]):
                        n_duplicates += 1
                        print('duplicates %d/%d' % (n_duplicates, n_graphs * (n_graphs - 1) / 2))
    if n_duplicates > 0:
        raise ValueError('%d duplicates found in the dataset' % n_duplicates)

    print('no duplicated graphs')


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


# COLORS
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
    graph_labels = np.array(graph_labels, np.int32)
    N_edges = np.array(N_edges, np.int32)
    print(N_graphs, len(graph_labels))

    return {'Adj_matrices': Adj_matrices,
            'GT_attn': GT_attn,  # not normalized to sum=1
            'graph_labels': graph_labels,
            'node_features': node_features,
            'N_edges': N_edges}


# TRIANGLES
def get_gt_atnn_triangles(args):
    G, N = args
    node_ids = []
    if G is not None:
        for clq in nx.enumerate_all_cliques(G):
            if len(clq) == 3:
                node_ids.extend(clq)
    node_ids = np.array(node_ids)
    gt_attn = np.zeros((N, 1), np.int32)
    for i in np.unique(node_ids):
        gt_attn[i] = int(np.sum(node_ids == i))
    return gt_attn  # unnormalized (do not sum to 1, i.e. use int32 for storage efficiency)


def get_graph_triangles(args):
    N_nodes, rnd = args
    N_edges = int((rnd.rand() + 1) * N_nodes)
    # N_edges = int((rnd.rand() * N_nodes + 1) * N_nodes)
    # N_edges = int((rnd.rand() * (N_nodes / 2 - 1) + 1) * N_nodes)
    #N_edges = int((rnd.rand() * (N_nodes / 2 - 1) + 1) * N_nodes)
    #print(N_nodes, N_edges)
    # assert N_edges >= N and N_edges <= N ** 2 / 2, (N_edges, N)
    # G, A = graph.random_graph(N_nodes, N_edges, seed=None)
    G = nx.dense_gnm_random_graph(N_nodes, N_edges, seed=None)
    A = nx.to_numpy_array(G)
    A_cube = A.dot(A).dot(A)
    label = int(np.trace(A_cube) / 6.)  # number of triangles
    return A.astype(np.bool), label, N_edges, G


def generate_graphs_Triangles(N_graphs, N_min, N_max, args, rnd):
    N_nodes = rnd.randint(N_min, N_max + 1, size=int(N_graphs * 10))
    print('generating %d graphs with %d-%d nodes' % (N_graphs * 10, N_min, N_max))
    # N_edges = []
    # for i in range(len(N_nodes)):
    #     if N_nodes[i]
    #     N_edges.append(int((rnd.rand() * (N_nodes[i] / 2 - 1) + 1) * N_nodes[i]))

    if args.threads > 0:
        with mp.Pool(processes=args.threads) as pool:
            data = pool.map(get_graph_triangles, [(N_nodes[i], rnd) for i in range(len(N_nodes))])
    else:
        data = [get_graph_triangles((N_nodes[i], rnd)) for i in range(len(N_nodes))]
    labels = np.array([data[i][1] for i in range(len(data))], np.int32)
    Adj_matrices, node_features, G, graph_labels, N_edges, node_degrees = [], [], [], [], [], []
    for lbl in range(args.label_min, args.label_max + 1):
        idx = np.where(labels == lbl)[0]
        c = 0
        for i in idx:
            add = True
            for k in range(len(Adj_matrices)):
                if data[i][0].shape[0] == Adj_matrices[k].shape[0] and labels[i] == graph_labels[k] and np.allclose(data[i][0], Adj_matrices[k]):
                    add = False
                    break
            if add:
                Adj_matrices.append(data[i][0])
                graph_labels.append(labels[i])
                G.append(data[i][3])
                N_edges.append(data[i][2])
                node_degrees.append(data[i][0].astype(np.int32).sum(1).max())
                c += 1
                if c >= int(N_graphs / (args.label_max - args.label_min + 1)):
                    break
        print('label={}, number of graphs={}/{}, total number of generated graphs={}'.format(lbl, c, len(idx), len(Adj_matrices)))

        assert c == int(N_graphs / (args.label_max - args.label_min + 1)), (
            'invalid data', c, int(N_graphs / (args.label_max - args.label_min + 1)))

    print('computing GT attention for %d graphs' % len(Adj_matrices))
    if args.threads > 0:
        with mp.Pool(processes=args.threads) as pool:
            GT_attn = pool.map(get_gt_atnn_triangles, [(G[i], Adj_matrices[i].shape[0]) for i in range(len(Adj_matrices))])
    else:
        GT_attn = [get_gt_atnn_triangles((G[i], Adj_matrices[i].shape[0])) for i in range(len(Adj_matrices))]

    graph_labels = np.array(graph_labels, np.int32)
    N_edges = np.array(N_edges, np.int32)

    return {'Adj_matrices': Adj_matrices,
            'GT_attn': GT_attn,  # not normalized to sum=1
            'graph_labels': graph_labels,
            'N_edges': N_edges,
            'Max_degree': np.max(node_degrees)}


if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)

    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    rnd = np.random.RandomState(args.seed)

    def print_stats(data, split_name):
        print('%s: %d graphs' % (split_name, len(data['graph_labels'])))
        for lbl in np.unique(data['graph_labels']):
            print('%s: label=%d, %d graphs' % (split_name, lbl, np.sum(data['graph_labels'] == lbl)))

    if args.dataset.lower() == 'colors':

        # Generate train and test sets
        data_test_combined, Adj_matrices, node_features = [], [], []
        for N_graphs, N_nodes_min, N_nodes_max, dim, name in zip([args.N_train + args.N_test, args.N_test, args.N_test],
                                                           [args.N_min, args.N_max_train + 1, args.N_max_train + 1],
                                                           [args.N_max_train, args.N_max, args.N_max],
                                                           [args.dim, args.dim, args.dim + 1],
                                                           ['test orig', 'test large', 'test large-c']):
            data = generate_graphs_Colors(N_graphs, N_nodes_min, N_nodes_max, dim, args, rnd, new_colors=dim==args.dim + 1)

            if name.find('orig') >= 0:
                idx = rnd.permutation(len(data['graph_labels']))
                data_train = copy_data(data, idx[:args.N_train])
                print_stats(data_train, name.replace('test', 'train'))
                node_features += data_train['node_features']
                Adj_matrices += data_train['Adj_matrices']
                data_test = copy_data(data, idx[args.N_train: args.N_train + args.N_test])
            else:
                data_test = copy_data(data, rnd.permutation(len(data['graph_labels']))[:args.N_test])

            Adj_matrices += data_test['Adj_matrices']
            node_features += data_test['node_features']
            data_test_combined.append(data_test)
            print_stats(data_test, name)

        # Check for duplicates in the combined train+test sets
        check_graph_duplicates(Adj_matrices, node_features)

        # Saving
        with open('%s/random_graphs_colors_dim%d_train.pkl' % (args.out_dir, args.dim), 'wb') as f:
            pickle.dump(data_train, f, protocol=2)

        with open('%s/random_graphs_colors_dim%d_test.pkl' % (args.out_dir, args.dim), 'wb') as f:
            pickle.dump(concat_data(data_test_combined), f, protocol=2)

    elif args.dataset.lower() == 'triangles':

        data = generate_graphs_Triangles((args.N_train + args.N_test), args.N_min, args.N_max_train, args, rnd)
        # Create balanced splits
        idx_train, idx_test = [], [] #rnd.permutation(len(data['graph_labels']))
        classes = np.unique(data['graph_labels'])
        n_classes = len(classes)
        for lbl in classes:
            idx = np.where(data['graph_labels'] == lbl)[0]
            rnd.shuffle(idx)
            n_train = int(args.N_train / n_classes)
            idx_train.append(idx[:n_train])
            idx_test.append(idx[n_train: n_train + int(args.N_test / n_classes)])
        data_train = copy_data(data, np.concatenate(idx_train))
        print_stats(data_train, 'train orig')
        data_test = copy_data(data, np.concatenate(idx_test))
        print_stats(data_test, 'test orig')

        data = generate_graphs_Triangles(args.N_test, args.N_max_train + 1, args.N_max, args, rnd)
        data_test_large = copy_data(data, rnd.permutation(len(data['graph_labels']))[:args.N_test])
        print_stats(data_test, 'test large')

        check_graph_duplicates(data_train['Adj_matrices'] + data_test['Adj_matrices'] + data_test_large['Adj_matrices'])

        # Saving
        # Max degree is max over all graphs in the training and test sets
        max_degree = np.max(np.array([d['Max_degree'] for d in (data_train, data_test, data_test_large)]))
        data_train['Max_degree'] = max_degree
        with open('%s/random_graphs_triangles_train.pkl' % args.out_dir, 'wb') as f:
            pickle.dump(data_train, f, protocol=2)

        data_test = concat_data((data_test, data_test_large))
        data_test['Max_degree'] = max_degree
        with open('%s/random_graphs_triangles_test.pkl' % args.out_dir, 'wb') as f:
            pickle.dump(data_test, f, protocol=2)

    else:
        raise NotImplementedError('unsupported dataset: ' + args.dataset)

    print('done in {}'.format(datetime.datetime.now() - dt))
