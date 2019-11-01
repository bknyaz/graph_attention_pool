import numpy as np
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from attention_pooling import *
from utils import *


class ChebyGINLayer(nn.Module):
    '''
    General Graph Neural Network layer that depending on arguments can be:
    1. Graph Convolution Layer (T. Kipf and M. Welling, ICLR 2017)
    2. Chebyshev Graph Convolution Layer (M. Defferrard et al., NeurIPS 2017)
    3. GIN Layer (K. Xu et al., ICLR 2019)
    4. ChebyGIN Layer (B. Knyazev et al., ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds)
    The first three types (1-3) of layers are particular cases of the fourth (4) case.
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 K,
                 n_hidden=0,
                 aggregation='mean',
                 activation=nn.ReLU(True),
                 n_relations=1):
        super(ChebyGINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_relations = n_relations
        assert K > 0, 'order is assumed to be > 0'
        self.K = K
        assert n_hidden >= 0, ('invalid n_hidden value', n_hidden)
        self.n_hidden = n_hidden
        assert aggregation in ['mean', 'sum'], ('invalid aggregation', aggregation)
        self.aggregation = aggregation
        self.activation = activation
        n_in = self.in_features * self.K * n_relations
        if self.n_hidden == 0:
            fc = [nn.Linear(n_in, self.out_features)]
        else:
            fc = [nn.Linear(n_in, n_hidden),
                  nn.ReLU(True),
                  nn.Linear(n_hidden, self.out_features)]
        if activation is not None:
            fc.append(activation)
        self.fc = nn.Sequential(*fc)
        print('ChebyGINLayer', list(self.fc.children())[0].weight.shape,
              torch.norm(list(self.fc.children())[0].weight, dim=1)[:10])

    def __repr__(self):
        return 'ChebyGINLayer(in_features={}, out_features={}, K={}, n_hidden={}, aggregation={})\nfc={}'.format(
            self.in_features,
            self.out_features,
            self.K,
            self.n_hidden,
            self.aggregation,
            str(self.fc))

    def chebyshev_basis(self, L, X, K):
        '''
        Return T_k X where T_k are the Chebyshev polynomials of order up to K.
        :param L: graph Laplacian, batch (B), nodes (N), nodes (N)
        :param X: input of size batch (B), nodes (N), features (F)
        :param K: Chebyshev polynomial order, i.e. filter size (number of hopes)
        :return: Tensor of size (B,N,K,F) as a result of multiplying T_k(L) by X for each order
        '''
        if K > 1:
            Xt = [X]
            Xt.append(torch.bmm(L, X))  # B,N,F
            for k in range(2, K):
                Xt.append(2 * torch.bmm(L, Xt[k - 1]) - Xt[k - 2])  # B,N,F
            Xt = torch.stack(Xt, 2)  # B,N,K,F
            return Xt
        else:
            # GCN
            assert K == 1, K
            return torch.bmm(L, X).unsqueeze(2)  # B,N,1,F

    def laplacian_batch(self, A, add_identity=False):
        '''
        Computes normalized Laplacian transformed so that its eigenvalues are in range [-1, 1].
        Note that sum of all eigenvalues = trace(L) = 0.
        :param A: Tensor of size (B,N,N) containing batch (B) of adjacency matrices of shape N,N
        :return: Normalized Laplacian of size (B,N,N)
        '''
        B, N = A.shape[:2]
        if add_identity:
            A = A + torch.eye(N, device=A.get_device() if A.is_cuda else 'cpu').unsqueeze(0)
        D = torch.sum(A, 1)  # nodes degree (B,N)
        D_hat = (D + 1e-5) ** (-0.5)
        L = D_hat.view(B, N, 1) * A * D_hat.view(B, 1, N)  # B,N,N
        if not add_identity:
            L = -L  # for ChebyNet to make a valid Chebyshev basis
        return D, L

    def forward(self, data):
        x, A, mask = data[:3]
        B, N, F = x.shape
        assert N == A.shape[1] == A.shape[2], ('invalid shape', N, x.shape, A.shape)

        if len(A.shape) == 3:
            A = A.unsqueeze(3)

        y_out = []
        for rel in range(A.shape[3]):
            D, L = self.laplacian_batch(A[:, :, :, rel], add_identity=self.K == 1)  # for the first layer this can be done at the preprocessing stage
            y = self.chebyshev_basis(L, x, self.K)  # B,N,K,F

            if self.aggregation == 'sum':
                # Sum features of neighbors
                if self.K == 1:
                    # GIN
                    y = y * D.view(B, N, 1, 1)
                else:
                    # ChebyGIN
                    D_GIN = torch.ones(B, N, self.K, device=x.get_device() if x.is_cuda else 'cpu')
                    D_GIN[:, :, 1:] = D.view(B, N, 1).expand(-1, -1, self.K - 1)  # keep self-loop features the same
                    y = y * D_GIN.view(B, N, self.K, 1)  # apply summation for other scales

            y_out.append(y)

        y = torch.cat(y_out, dim=2)
        y = self.fc(y.view(B, N, -1))  # B,N,F

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        y = y * mask.float()
        output = [y, A, mask]
        output.extend(data[3:] + [x])  # for python2

        return output


class GraphReadout(nn.Module):
    '''
    Global pooling layer applied after the last graph layer.
    '''
    def __init__(self,
                 pool_type):
        super(GraphReadout, self).__init__()
        self.pool_type = pool_type
        dim = 1  # pooling over nodes
        if pool_type == 'max':
            self.readout_layer = lambda x, mask: torch.max(x, dim=dim)[0]
        elif pool_type in ['avg', 'mean']:
            # sum over all nodes, then divide by the number of valid nodes in each sample of the batch
            self.readout_layer = lambda x, mask: torch.sum(x, dim=dim) / torch.sum(mask, dim=dim).float()
        elif pool_type in ['sum']:
            self.readout_layer = lambda x, mask: torch.sum(x, dim=dim)
        else:
            raise NotImplementedError(pool_type)

    def __repr__(self):
        return 'GraphReadout({})'.format(self.pool_type)

    def forward(self, data):
        x, A, mask = data[:3]
        B, N = x.shape[:2]
        x = self.readout_layer(x, mask.view(B, N, 1))
        output = [x]
        output.extend(data[1:])   # [x, *data[1:]] doesn't work in Python2
        return output


class ChebyGIN(nn.Module):
    '''
    Graph Neural Network class.
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 filters,
                 K=1,
                 n_hidden=0,
                 aggregation='mean',
                 dropout=0,
                 readout='max',
                 pool=None,  # Example: 'attn_gt_threshold_0_skip_skip'.split('_'),
                 pool_arch='fc_prev'.split('_'),
                 large_graph=False,  # > ~500 graphs
                 kl_weight=None,
                 graph_layer_fn=None,
                 init='normal',
                 scale=None,
                 debug=False):
        super(ChebyGIN, self).__init__()
        self.out_features = out_features
        assert len(filters) > 0, 'filters must be an iterable object with at least one element'
        assert K > 0, 'filter scale must be a positive integer'
        self.pool = pool
        self.pool_arch = pool_arch
        self.debug = debug
        n_prev = None

        attn_gnn = None
        if graph_layer_fn is None:
            graph_layer_fn = lambda n_in, n_out, K_, n_hidden_, activation: ChebyGINLayer(in_features=n_in,
                                                               out_features=n_out,
                                                               K=K_,
                                                               n_hidden=n_hidden_,
                                                               aggregation=aggregation,
                                                               activation=activation)
            if self.pool_arch is not None and self.pool_arch[0] == 'gnn':
                attn_gnn = lambda n_in: ChebyGIN(in_features=n_in,
                                                 out_features=0,
                                                 filters=[32, 32, 1],
                                                 K=np.min((K, 2)),
                                                 n_hidden=0,
                                                 graph_layer_fn=graph_layer_fn)

        graph_layers = []

        for layer, f in enumerate(filters + [None]):

            n_in = in_features if layer == 0 else filters[layer - 1]
            # Pooling layers
            # It's a non-standard way to put pooling before convolution, but it's important for our work
            if self.pool is not None and len(self.pool) > len(filters) + layer and self.pool[layer + 3] != 'skip':
                graph_layers.append(AttentionPooling(in_features=n_in, in_features_prev=n_prev,
                                                     pool_type=self.pool[:3] + [self.pool[layer + 3]],
                                                     pool_arch=self.pool_arch,
                                                     large_graph=large_graph,
                                                     kl_weight=kl_weight,
                                                     attn_gnn=attn_gnn,
                                                     init=init,
                                                     scale=scale,
                                                     debug=debug))

            if f is not None:
                # Graph "convolution" layers
                # no ReLU if the last layer and no fc layer after that
                graph_layers.append(graph_layer_fn(n_in, f, K, n_hidden,
                                                   None if self.out_features == 0 and layer == len(filters) - 1 else nn.ReLU(True)))
                n_prev = n_in

        if self.out_features > 0:
            # Global pooling over nodes
            graph_layers.append(GraphReadout(readout))
        self.graph_layers = nn.Sequential(*graph_layers)

        if self.out_features > 0:
            # Fully connected (classification/regression) layers
            self.fc = nn.Sequential(*(([nn.Dropout(p=dropout)] if dropout > 0 else []) + [nn.Linear(filters[-1], out_features)]))

    def forward(self, data):
        data = self.graph_layers(data)
        if self.out_features > 0:
            y = self.fc(data[0])  # B,out_features
        else:
            y = data[0]  # B,N,out_features
        return y, data[4]
