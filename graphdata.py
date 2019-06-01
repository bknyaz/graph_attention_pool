import numpy as np
import os
from os.path import join as pjoin
import pickle
import copy
import torch
import torch.utils
import torch.utils.data
import utils


def collate_batch(batch):
    '''
    Creates a batch of same size graphs by zero-padding node features and adjacency matrices up to
    the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
    :param batch: batch in the PyTorch Geometric format or [node_features*batch_size, A*batch_size, label*batch_size]
    :return: [node_features, A, graph_support, N_nodes, label]
    '''

    # [signal, int(label), W, etc, node_centrality, N_nodes]
    # data = [self.node_features[index], self.Adj_matrices[index], N_nodes, self.labels[index], self.GT_attn[index]]
    B = len(batch)
    N_nodes = [batch[b][2] for b in range(B)]
    C = batch[0][0].shape[1]
    N_nodes_max = int(np.max(N_nodes))

    mask = torch.zeros(B, N_nodes_max)
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


# Data can be downloaded from https://github.com/horacepan/gnns
# Then converted to pickle using make_pickle.py in https://github.com/horacepan/gnns
# All datasets can also be found in https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 fold_id,
                 datareader,
                 sub_fold_id=None,
                 pool=None,
                 split='train',
                 trunc_feat=False,
                 add_degree_feature=False,
                 add_degree_relation=False,
                 wl_relation=0,
                 compute_monet=False,
                 batch_size=None,
                 optimized=False,
                 hierarchy=False,
                 plot=False,
                 add_path_relation=0,
                 spatial_select=0):
        '''
        Create a dataset object that can be fed for torch.utils.data.DataLoader
        :param fold_id:
        :param data:                container returned by create_splits_from_pickle()
        :param pool:
        :param N_layers:            number of layers in the network
        :param max_dims:            dimensions for pooling (not tested)
        :param compute_centrality:  compute harmonic centrality for each node (slow), try using node degree instead
        :param rnd_state:
        :param shuffle_nodes:       set to True to make sure the model is not sensitive to the order of nodes
        :param split:
        '''
        self.fold_id = fold_id
        self.pool = pool
        self.N_layers = N_layers
        self.max_dims = max_dims
        self.compute_centrality = compute_centrality
        self.rnd_state = datareader.rnd_state
        self.datareader = datareader
        self.shuffle_nodes = shuffle_nodes
        print('node shuffling is %s' % ('on' if self.shuffle_nodes else 'off'))
        self.split = split
        self.trunc_feat = trunc_feat
        self.add_degree_feature = add_degree_feature
        self.add_degree_relation = add_degree_relation
        self.wl_relation = wl_relation
        self.compute_monet = compute_monet
        self.optimized = optimized
        self.hierarchy = hierarchy
        self.plot = plot
        self.add_path_relation = add_path_relation
        self.spatial_select = spatial_select

        self.set_fold(datareader.data, fold_id)

        if self.add_degree_feature:
            self.features_dim += (self.degree_max + 1)
            
        if self.wl_relation > 0:
            self.WL_relations = {}

        if self.add_path_relation > 0 and self.spatial_select > 0:
            assert self.add_path_relation == self.spatial_select, ('inconsistent values', self.add_path_relation, self.spatial_select)

        if self.add_path_relation > 0 or self.spatial_select > 0:
            self.cutoff = np.max((self.add_path_relation, self.spatial_select))
            print('adding path relation with cutoff %d' % self.cutoff)
            self.Path_relations = {}

        # self.features_dim *= 2

        if sub_fold_id is not None:
            raise NotImplementedError()
        #     self.idx = splits[fold_id][self.split]
        # else:
        #     self.idx = splits[fold_id][self.split][sub_fold_id]

        self.batch_size = batch_size

        # default order is not shuffled
        self.indices = np.arange(len(self.idx))  # sample indices for this epoch
        self.N_nodes_max_lst = self.get_nodes_max(self.batch_size, self.indices)

    def set_fold(self, data, fold_id):
        self.total = len(data['targets'])
        self.n_samples = self.total
        self.N_nodes_max = data['N_nodes_max']
        self.n_classes = data['n_classes']
        self.features_dim = data['features_dim']
        self.idx = data['splits'][fold_id][self.split]
         # use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        
        self.degree_max = data['degree_max']
        
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        if data['features_onehot'] is not None:
            self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
        else:
            self.features_onehot = [np.ones((data['adj_list'][i].shape[0], 1)) for i in self.idx]
            self.features_dim = 1
        if 'weak_signal_attn' in data:
            print('!!!using weakly supervised labels!!!'.upper(), self.split)
            self.w_sup_signal_attn = copy.deepcopy([data['weak_signal_attn'][i] for i in self.idx])
        else:
            self.w_sup_signal_attn = None

        shapes = np.array([A.shape[0] for A in self.adj_list])
        # plt.hist(shapes)
        #print(self.datareader.data_dir)
        # plt.savefig('%s_shapes_hist_%s.png' % (os.path.basename(self.datareader.data_dir), self.split))
        print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))
        self.indices = np.arange(len(self.idx))  # sample indices for this epoch

        print('%s: N_nodes_min: %d' % (self.split, np.min(shapes)))
        print('%s: N_nodes_max: %d'% (self.split, np.max(shapes)))


        classes = np.unique(self.labels)
        for lbl in classes:
            print('%s, Class %d: \t\t\t%d samples' % (self.split, lbl, np.sum(self.labels == lbl)))

    def get_nodes_max(self, batch_size, indices):
        N_nodes_max_lst = np.zeros(len(indices), np.int32)
        n_batches = int(np.ceil(len(indices) / float(batch_size)))
        for b in range(n_batches):
            ind = np.arange(b * batch_size,
                            np.min((len(indices), (b + 1) * batch_size)))  # indices inside the batch
            n = np.max([len(self.features_onehot[i]) for i in indices[ind]])
            for i in indices[ind]:
                N_nodes_max_lst[i] = n if self.optimized else self.N_nodes_max

        return N_nodes_max_lst

    def shuffle_loader(self):
        self.indices = self.rnd_state.permutation(len(self.idx))  # sample indices for this epoch
        self.N_nodes_max_lst = self.get_nodes_max(self.batch_size, self.indices)

    def __len__(self):
        return len(self.labels)

    # @profile
    def __getitem__(self, ind):
        index = self.indices[ind]
        # N_nodes_max = self.N_nodes_max_lst[index]
        signal = self.features_onehot[index].copy()
        label = self.labels[index]
        # print(index, self.adj_lists[index].shape[0], N_nodes_max)
        N_nodes = self.adj_list[index].shape[0]
        # print(self.w_sup_signal_attn is not None, self.split.lower())
        if self.w_sup_signal_attn is not None and self.split.lower().find('train') >= 0:
            signal_attn = self.w_sup_signal_attn[index]
            assert signal_attn is not None and len(signal_attn) == len(signal), (signal_attn.shape == signal.shape)
        else:
            signal_attn = None

        if self.add_degree_relation:
            # D_dist = (np.sum(self.adj_list[index], 1, keepdims=True) + 1e-5) ** (-0.5)
            # D_dist /= np.max(D_dist)
            # dist = cdist(D_dist, D_dist)
            # sigma = 0.01 * np.pi  # some hard coded value for now
            # W_degree = np.exp(- dist / sigma ** 2)  # np.exp(- dist)
            # W_degree[np.diag_indices_from(W_degree)] = 0
            D = np.sum(self.adj_list[index], 1, keepdims=True) ** 0.5  # N,1
            W_degree = np.stack((np.tile(D, (1, N_nodes)), np.tile(D.T, (N_nodes, 1))), 2)
            # W_degree = np.sort(W_degree)  # sort so that the first value is smaller making it nodes permutation invariant
            W_degree /= (W_degree.max() + 1e-7)
            # W_degree = utils.pad(W_degree, N_nodes_max, N_nodes_max)

        if self.add_path_relation > 0 or self.spatial_select > 0:
            if index not in self.Path_relations:
                W_path = graph.node_shortestpath_relations(self.adj_list[index], cutoff=self.cutoff)
                idx = W_path < 0  # no path
                sigma = 0.5 * np.pi  # some hard coded value for now
                W_path = np.exp(- W_path / sigma ** 2)
                W_path[idx] = 0
                W_path[np.diag_indices_from(W_path)] = 0
                self.Path_relations[index] = W_path
            else:
                W_path = self.Path_relations[index].copy()

            # W_path = utils.pad(W_path, N_nodes_max, N_nodes_max)

        # W = utils.pad(self.adj_list[index], N_nodes_max, N_nodes_max)
        W = self.adj_list[index]
        # W = np.exp(W)
        # W = W / np.sum(W, axis=1, keepdims=True)
        # W = (W + np.transpose(W, (1, 0))) / 2

        signal_levels = np.ones((len(signal), 1))  #
        # signal = utils.pad(signal, N_nodes_max)  # N,F
        # signal_levels = utils.pad(signal_levels, N_nodes_max)  # N,F

        if self.shuffle_nodes:
            signal, W, idx = graph.shuffle_graph_nodes(signal, W, to_copy=False, rnd_state=self.rnd_state)  # the objects are already copied
            signal_levels = signal_levels[idx]
            assert not self.add_path_relation

        # signal = np.concatenate((signal, np.dot(W, signal) / (np.sum(W, axis=1, keepdims=True) + 1e-5)), axis=1)

        # assert np.allclose(W, W.T), ('adjacency is not symmetric', W.shape)

        if self.hierarchy or self.add_degree_feature or self.add_degree_relation:
            D = np.sum(W, 1, keepdims=True)  # N,1  as in  [M. Simonovsky et al., CVPR'2017]
            D_onehot = np.zeros((N_nodes, self.degree_max + 1))
            D_onehot[np.arange(N_nodes), D.astype(np.int).squeeze()] = 1
            D = D ** 0.5  # N,1  as in  [M. Simonovsky et al., CVPR'2017]

        if self.hierarchy:
            # idx = np.argsort(D.squeeze())
            # W = W[:, idx]
            # W = W[idx, :]
            # signal_levels = signal_levels[idx]
            # signal = signal[idx]
            # D = D[idx, :]
            W_hier = ((D < D.squeeze()) & (D > 0))  # 1 if label is the same
            W_hier = (W_hier | W_hier.T).astype(np.float32)
            signal_levels = D
            # W_hier[np.diag_indices_from(W_hier)] = 0  # zero diag (no loop in the graph)

        if self.compute_centrality:
            node_centrality = graph.node_harmonic_centrality(W).reshape(-1, 1)
        else:
            node_centrality = np.zeros((W.shape[0], 1))

        if self.compute_monet:
            # print(coord.shape)
            # monet_coord = get_monet_coord(W, max_dims=[N_nodes_max])
            monet_coord = get_monet_coord(W, max_dims=[N_nodes])
        else:
            monet_coord = 0

        if self.pool is not None and self.pool[0] == 'graclus':

            if self.shuffle_nodes:
                raise ValueError('this case is not supported, '
                                 'because graclus_graph_pooling does not accepted padded adjacency matrices')

            if self.N_layers != 3:
                raise ValueError('this case is not supported, '
                                 'because graclus_graph_pooling requires a predefined number of coarsening layers')

            W = [ [ W ] ]
            pool_maps = []
            pmaps, graphs = graph.graclus_graph_pooling(self.adj_lists[index], levels=5)
            max_dims = [self.N] + list(self.max_dims)
            for j, pmap in enumerate(pmaps):
                pool_maps.append(utils.pad(pmap, max_dims[j + 1], max_dims[j]))
                if pmap.shape[0] > 1:
                    w = utils.pad(graphs[(j + 1) * 2].todense(), max_dims[j + 1], max_dims[j + 1])
                    w /= w.max()
                    W[0].append(w)

            for i, pmap in enumerate(pool_maps):
                pool_maps[i] = np.tile(pool_maps[i].reshape(1, pool_maps[i].shape[0], pool_maps[i].shape[1]),
                                       (i + 1, 1, 1))
        else:
            if self.wl_relation > 0:
                if index not in self.WL_relations:
                    W_wl = self.wl_similarity(W, signal, N_it=self.wl_relation)
                    self.WL_relations[index] = W_wl
                else:
                    W_wl = self.WL_relations[index].copy()

            if self.add_path_relation > 0 or self.spatial_select > 0:
                # W2 = W.dot(W)
                # dd = (np.sum(W2, axis=1) + 1e-5) ** -0.5
                # W2 = dd.reshape(-1, 1) * W2 * dd.reshape(1, -1)
                W = [ W_path[:, :, None], W[:, :, None] ]
            else:
                W = [ W[:, :, None] ]
            if self.add_degree_relation:
                W.append(W_degree)
            if self.wl_relation > 0:
                W.append(W_wl[:, :, None])
            # if self.hierarchy:
            #     W.append(W_hier)
            W = np.concatenate(W, axis=2)
            pool_maps = [ 0 ] #[ [ [ -1 ] ] * self.N_layers ]  # no pooling



        if self.plot and N_nodes >= 7 and N_nodes < 15: # debug: and self.files[index].find('train_2009_005118') >= 0:
            # print(self.files[index], sample.seg_sample)
            for i in range(W.shape[2]):
                ww = W[:, :, i]
                print(i, ww.shape, ww.min(), ww.max())
                #assert ww.min() >= 0 and ww.max() <= 1, (ww.min(), ww.max())
                plt.figure()
                plt.imshow(ww ** 0.5)  # to make small values visible
                plt.colorbar()
                fpath = 'w_%d.png' % (i)
                plt.savefig(fpath)
                plt.savefig(fpath.replace('.png', '.eps'), format='eps')
                np.save(fpath.replace('.png', '.npy'), ww)
                np.save('x_%d.png' % (i), signal)
                n = np.sum(np.isnan(ww))
                # assert ww.shape[0] == ww.shape[1] == 300, ww.shape
                assert n == 0, (i, n, ww.shape)

            raise ValueError('stop and analyze saved matrices')

        etc = pool_maps

        if self.trunc_feat > 0:
            coord = np.zeros((signal.shape[0], np.min((self.trunc_feat, signal.shape[1]))), np.float32)
            for node in range(signal.shape[0]):
                ind = np.where(signal[node] > 0)[0]
                if len(ind) == 0:
                    continue
                assert len(ind) == 1, (len(ind), ind)
                # ind = np.clip(ind, 0, self.trunc_feat - 1)
                if ind >= self.trunc_feat:
                    pass
                else:
                    coord[node][ind] = 1
        else:
            coord = signal

        if self.add_degree_feature:
            #print(D.shape, signal.shape, coord.shape)
            signal = np.concatenate((signal, D_onehot), axis=1)
            coord = np.concatenate((coord, D_onehot), axis=1)

        if self.compute_centrality:
            signal = np.concatenate((signal, node_centrality), axis=1)
            coord = np.concatenate((coord, node_centrality), axis=1)

        etc += [ coord ]
        etc += [ monet_coord ]
        # etc += [[ signal_levels ]]

        #print(W.shape)
        data = [ signal, int(label), W, etc, node_centrality, N_nodes, signal_attn ]
        data = utils.list_to_torch(data)  # convert to torch

        return data


    def wl_similarity(self, w, x, N_it=5):
        N = w.shape[0]
        x_new = x
        x_range = np.arange(1, N + 1)
        x_new_one_hot_all = []
        for it in range(N_it):
            x_colors = {}
            for xx in x_new:
                if tuple(xx) not in x_colors:
                    x_colors[tuple(xx)] = x_range[len(x_colors)]

            x2 = []
            for i in range(N):
                x2_i = [x_colors[tuple(x_new[i])]]
                for j in np.where(w[i] > 0)[0]:
                    x2_i.append(x_colors[tuple(x_new[j])])
                x2.append(sorted(x2_i))

            x_new = x2

            x_new_one_hot = np.zeros((N, len(x_colors)))
            for i in range(N):
                for j in range(len(x_colors)):
                    x_new_one_hot[i, j] = len(np.where(np.array(x_new[i]) == j + 1)[0])

            # x_new_one_hot /= np.linalg.norm(x_new_one_hot, axis=1, keepdims=True)

            x_new_one_hot_all.append(x_new_one_hot)

        x_new_one_hot_all = np.concatenate(x_new_one_hot_all, axis=1)
        A_wl = x_new_one_hot_all.dot(x_new_one_hot_all.T)
        A_wl[np.diag_indices_from(A_wl)] = 0
        A_wl /= (A_wl.max() + 1e-7)

        return A_wl


class DataReader():
    '''
    Class to read the txt files containing all data of the dataset
    '''

    def __init__(self,
                 data_dir,  # folder with txt files
                 rnd_state=None,
                 attn_labels='',
                 N_nodes=None,
                 use_cont_node_attr=False,
                 # use or not additional float valued node attributes available in some datasets
                 folds=10):

        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        if os.path.isfile('%s/data.pkl' % data_dir):
            print('loading data from %s/data.pkl' % data_dir)
            with open('%s/data.pkl' % data_dir, 'rb') as f:
                self.data = pickle.load(f)
        else:
            files = os.listdir(self.data_dir)
            data = {}
            nodes, graphs = self.read_graph_nodes_relations(
                list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
            lst = list(filter(lambda f: f.find('node_labels') >= 0, files))
            if len(lst) > 0:
                assert len(lst) == 1, (len(lst), files)
                data['features'] = self.read_node_features(lst[0],
                                                           nodes, graphs, fn=lambda s: int(s.strip()))
            else:
                data['features'] = None
            data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)
            data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                           line_parse_fn=lambda s: int(float(s.strip()))))

            if self.use_cont_node_attr:
                data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                       nodes, graphs,
                                                       fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

            features, n_edges, degrees = [], [], []
            for sample_id, adj in enumerate(data['adj_list']):
                N = len(adj)  # number of nodes
                if data['features'] is not None:
                    assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
                n = np.sum(adj)  # total sum of edges
                # assert n % 2 == 0, n
                n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
                if not np.allclose(adj, adj.T):
                    print(sample_id, 'not symmetric')
                degrees.extend(list(np.sum(adj, 1)))
                if data['features'] is not None:
                    features.append(np.array(data['features'][sample_id]))

            # Create features over graphs as one-hot vectors for each node
            features_dim = 0
            if data['features'] is not None:
                features_all = np.concatenate(features)
                features_min = features_all.min()
                features_dim = int(features_all.max() - features_min + 1)  # number of possible values

                features_onehot = []
                for i, x in enumerate(features):
                    feature_onehot = np.zeros((len(x), features_dim))
                    for node, value in enumerate(x):
                        feature_onehot[node, value - features_min] = 1
                    if self.use_cont_node_attr:
                        feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
                    features_onehot.append(feature_onehot)

                if self.use_cont_node_attr:
                    features_dim = features_onehot[0].shape[1]

            shapes = np.array([len(adj) for adj in data['adj_list']])
            labels = data['targets']  # graph class labels
            labels -= np.min(labels)  # to start from 0
            # N_nodes_max = np.max(shapes)

            classes = np.unique(labels)
            n_classes = len(classes)

            if not np.all(np.diff(classes) == 1):
                print('making labels sequential, otherwise pytorch might crash')
                labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
                for lbl in range(n_classes):
                    labels_new[labels == classes[lbl]] = lbl
                labels = labels_new
                classes = np.unique(labels)
                assert len(np.unique(labels)) == n_classes, np.unique(labels)

            print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (
            np.mean(shapes), np.std(shapes), np.min(shapes), np.max(shapes)))
            print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (
            np.mean(n_edges), np.std(n_edges), np.min(n_edges), np.max(n_edges)))
            print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (
            np.mean(degrees), np.std(degrees), np.min(degrees), np.max(degrees)))
            print('Node features dim: \t\t%d' % features_dim)
            print('N classes: \t\t\t%d' % n_classes)
            print('Classes: \t\t\t%s' % str(classes))
            for lbl in classes:
                print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

            N_graphs = len(labels)  # number of samples (graphs) in data

            if data['features'] is not None:
                for u in np.unique(features_all):
                    print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

                assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

            if data['features'] is not None:
                data['features_onehot'] = features_onehot
            else:
                data['features_onehot'] = None
            data['targets'] = labels

            data['N_nodes_max'] = np.max(shapes)  # max number of nodes
            data['features_dim'] = features_dim
            data['n_classes'] = n_classes

            with open('%s/data.pkl' % data_dir, 'wb') as f:
                pickle.dump(data, f, protocol=2)
            self.data = data

        self.w_sup_signal_attn = None
        if attn_labels is not None and os.path.isfile(attn_labels):
            with open(attn_labels, 'rb') as f:
                self.data['weak_signal_attn'] = pickle.load(f)

        # Create test sets first
        N_graphs = len(self.data['targets'])
        shapes = np.array([len(adj) for adj in self.data['adj_list']])
        if folds > 1:
            train_ids, test_ids = self.split_ids(np.arange(N_graphs), self.rnd_state, folds=folds)
        else:
            train_ids, test_ids = self.split_ids_shape(np.arange(N_graphs), shapes, self.rnd_state, N_nodes,
                                                       folds=folds)

        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train_val': train_ids[fold],
                           'test': test_ids[fold]})

        self.data['degree_max'] = int(np.max([np.sum(A, 1).max() for A in self.data['adj_list']]))
        print('degree_max', self.data['degree_max'])
        self.data['splits'] = splits



    def split_ids_shape(self, ids_all, shapes, rnd_state, N_nodes, folds=1):
        ind = np.where(shapes <= N_nodes)[0]
        print(len(shapes), len(ind), N_nodes)
        idx = rnd_state.permutation(len(ind))
        if len(idx) > 1000:
            n = 1000
        else:
            n = 500
        train_ids = [ind[idx[:n]]]
        test_ids = ind[idx[n:]]
        ind = np.where(shapes > N_nodes)[0]
        test_ids = [np.concatenate((test_ids, ind))]

        assert np.all(
            np.unique(np.concatenate((train_ids[0], test_ids[0]))) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        for fold in range(folds):
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(
                np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == len(ids_all), 'invalid splits'

        print('good')

        return train_ids, test_ids

    def split_ids(self, ids_all, rnd_state=None, folds=10):
        n = len(ids_all)
        ids = ids_all[rnd_state.permutation(n)]
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(
            np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(
                np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1

        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

        return adj_list

    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

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

        N_nodes = self.Adj_matrices[index].shape[0] #np.random.randint(np.min((4, self.N_nodes_max - 1)), self.N_nodes_max)
        data = [self.node_features[index],
                self.Adj_matrices[index],
                N_nodes,
                self.labels[index],
                self.GT_attn[index]]
        data = utils.list_to_torch(data)  # convert to torch

        return data