import numpy as np
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
                 activation=nn.ReLU(True)):
        super(ChebyGINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert K > 0, 'order is assumed to be > 0'
        self.K = K
        assert n_hidden >= 0, ('invalid n_hidden value', n_hidden)
        self.n_hidden = n_hidden
        assert aggregation in ['mean', 'sum'], ('invalid aggregation', aggregation)
        self.aggregation = aggregation

        n_in = self.in_features * self.K
        if self.n_hidden == 0:
            self.fc = nn.Sequential(nn.Linear(n_in, self.out_features),
                                    activation)
        else:
            self.fc = nn.Sequential(nn.Linear(n_in, n_hidden),
                                    nn.ReLU(True),
                                    nn.Linear(n_hidden, self.out_features),
                                    activation)

        # print weight norms for reproducibility analysis
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
        D_hat = (D + 1e-7) ** (-0.5)
        L = D_hat.view(B, N, 1) * A * D_hat.view(B, 1, N)  # B,N,N

        if not add_identity:
            L = -L  # for ChebyNet to make a valid Chebyshev basis
        return D, L

    def forward(self, data):
        x, A, mask = data[:3]
        B, N, F = x.shape
        x_input = x.clone()
        assert N == A.shape[1] == A.shape[2], ('invalid shape', N, x.shape, A.shape)
        # data[4]['x_prev'] = x

        D, L = self.laplacian_batch(A, add_identity=self.K == 1)  # for the first layer this can be done at the preprocessing stage
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

        y = self.fc(y.view(B, N, -1))  # B,N,F

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        y = y * mask.float()

        return [y, A, mask, *data[3:], x_input]


class GraphReadout(nn.Module):
    def __init__(self,
                 pool_type):
        super(GraphReadout, self).__init__()
        self.pool_type = pool_type
        dim = 1  # pooling over nodes
        if pool_type == 'max':
            self.readout_layer = lambda x, mask: torch.max(x, dim=dim)[0]
        elif pool_type in ['avg', 'mean']:
            # sum over all nodes, then divide by the number of valid nodes in each sample of the batch
            self.readout_layer = lambda x, mask: torch.sum(x, dim=dim) / torch.sum(mask, dim=1, keepdim=True)
        elif pool_type in ['sum']:
            self.readout_layer = lambda x, mask: torch.sum(x, dim=dim)
        else:
            raise NotImplementedError(pool_type)

    def __repr__(self):
        return 'GraphReadout({})'.format(self.pool_type)

    def forward(self, data):
        x, A, mask = data[:3]
        x = self.readout_layer(x, mask)
        return [x, *data[1:]]


class AttentionPooling(nn.Module):
    def __init__(self,
                 in_features,  # feature dimensionality in the current graph layer
                 in_features_prev,  # feature dimensionality in the previous graph layer
                 pool_type,
                 pool_arch,
                 kl_weight=None):
        super(AttentionPooling, self).__init__()
        self.pool_type = pool_type
        self.pool_arch = pool_arch
        self.kl_weight = kl_weight
        print(pool_type, pool_arch)
        if self.pool_type[1] in ['unsup', 'sup']:
            assert self.pool_arch not in [None, 'None'], self.pool_arch

            if self.pool_arch[0] == 'fc':
                n_in = in_features_prev if self.pool_arch[1] == 'prev' else in_features
                p_optimal = torch.from_numpy(np.pad(np.array([0, 1]), (0, n_in - 2), 'constant')).float().view(1, n_in)
                if len(self.pool_arch) == 2:
                    # single layer projection
                    self.proj = nn.Linear(n_in, 1, bias=False)
                    p = self.proj.weight.data.view(1, n_in)
                else:
                    # multi-layer projection
                    filters = list(map(int, self.pool_arch[2:]))
                    self.proj = []
                    for layer in range(len(filters)):
                        self.proj.append(nn.Linear(in_features=n_in if layer == 0 else filters[layer - 1],
                                                   out_features=filters[layer]))
                        if layer == 0:
                            p = self.proj[0].weight.data
                        self.proj.append(nn.ReLU(True))

                    self.proj.append(nn.Linear(filters[-1], 1))
                    self.proj = nn.Sequential(*self.proj)

                # Compute cosine similarity with the optimal vector and print values
                # ignore the last dimension, because it does not receive gradients during training
                # n_in=4 for colors-3 because some of our test subsets have 4 dimensional features
                cos_sim = self.cosine_sim(p[:, :-1], p_optimal[:, :-1])
                if p.shape[0] == 1:
                    print('p values', ['%.7f' % p_i.item() for p_i in p[0]])
                    print('cos_sim', cos_sim.item())
                else:
                    for fn in [torch.max, torch.min, torch.mean, torch.std]:
                        print('cos_sim', fn(cos_sim).item())

        elif self.pool_type[1] == 'gt':
            pass # ignore other parameters
        else:
            raise NotImplementedError(self.pool_type[1])

    def __repr__(self):
        return 'AttentionPooling({})'.format(self.pool_type)

    def cosine_sim(self, a, b):
        return torch.mm(a, b.t()) / (torch.norm(a, dim=1, keepdim=True) * torch.norm(b, dim=1, keepdim=True))

    def forward(self, data):
        KL_loss = None
        x_, A, mask_, _, params_dict = data[:5]

        mask = mask_.clone()
        x = x_.clone()

        B, N, C = x.shape
        if self.pool_type[1] in ['gt', 'sup']:
            if 'node_attn' in params_dict:
                alpha_gt = params_dict['node_attn']
            else:
                raise ValueError('ground truth node attention values node_attn required for %s' % self.pool_type)

        if self.pool_type[1] in ['unsup', 'sup']:
            attn_input = data[-1] if self.pool_arch[1] == 'prev' else x.clone()

            alpha_pre = (torch.exp(self.proj(attn_input)).view_as(mask) * mask_.float()).view(B, N)
            # alpha_pre = torch.sum(attn_input, dim=2) ** 2  #mask.float().view(B, N)

            alpha = alpha_pre / (torch.sum(alpha_pre, dim=1, keepdim=True) + 1e-7)
            if self.pool_type[1] == 'sup':
                # print(alpha.shape, alpha_gt.shape, alpha.min(), alpha.max(), alpha_gt.min(), alpha_gt.max())
                KL_loss = torch.mean((F.kl_div(torch.log(alpha + 1e-14), alpha_gt.view(B, N), reduction='none').view_as(
                    mask) * mask.float()).sum(dim=1))
                # KL_loss = self.kl_weight * torch.mean(KL_loss.sum(dim=1) / (params_dict['N_nodes'].float() + 1e-7))
        else:
            alpha = alpha_gt

        mask = mask & (alpha.view_as(mask) > 0)

        x = x * alpha.view(B, N, 1)
        if N > 700:
            x = x * params_dict['N_nodes'].view(B, 1, 1).float()

        mask = mask.view(B, N)
        drop = False
        if drop:
            N_nodes = torch.sum(mask, dim=1).long()  # B
            N_nodes_max = N_nodes.max()

            # Drop nodes
            mask, idx = torch.topk(mask, N_nodes_max, dim=1, largest=True, sorted=False)
            x = torch.gather(x, dim=1, index=idx.unsqueeze(2).expand(-1, -1, C))

            # Drop edges
            A = torch.gather(A, 1, idx.unsqueeze(2).expand(-1, -1, N))
            A = torch.gather(A, 2, idx.unsqueeze(1).expand(-1, N_nodes_max, -1))

        mask_matrix = mask.unsqueeze(2) & mask.unsqueeze(1)
        A = A * mask_matrix.float()   # or A[~mask_matrix] = 0

        # Add additional losses regularizing the model
        if 'reg' not in params_dict:
            data[4]['reg'] = []
        if KL_loss is not None:
            data[4]['reg'].append(KL_loss)

        return [x, A, mask, *data[3:]]


class ChebyGIN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters,
                 K,
                 n_hidden=0,
                 aggregation='mean',
                 dropout=0,
                 readout='max',
                 pool='attn_gt_threshold_0_skip_skip',
                 pool_arch='fc_prev',
                 kl_weight=None):
        super(ChebyGIN, self).__init__()

        self.pool = None if pool is None else pool.split('_')
        self.pool_arch = None if pool_arch is None else pool_arch.split('_')

        n_prev = None
        layers = []
        for layer, f in enumerate(filters):

            n_in = in_features if layer == 0 else filters[layer - 1]
            # Pooling layers
            if len(self.pool) > len(filters) + layer and self.pool[layer + 3] != 'skip':
                layers.append(AttentionPooling(in_features=n_in, in_features_prev=n_prev,
                                               pool_type=self.pool[:3] + [self.pool[layer + 3]],
                                               pool_arch=self.pool_arch, kl_weight=kl_weight))

            # Graph convolution layers
            layers.append(ChebyGINLayer(in_features=n_in,
                                                    out_features=f,
                                                    K=K,
                                                    n_hidden=n_hidden,
                                                    aggregation=aggregation,
                                                    activation=nn.ReLU(inplace=False)))
            n_prev = n_in

        # Global pooling over nodes
        layers.append(GraphReadout(readout))
        self.layers = nn.Sequential(*layers)

        # Fully connected (classification/regression) layers
        self.fc = nn.Sequential(*(([nn.Dropout(p=dropout)] if dropout > 0 else []) + [nn.Linear(filters[-1], out_features)]))

    def forward(self, data):
        data = self.layers(data)
        x = self.fc(data[0])  # B,out_features
        reg = data[4]['reg'] if 'reg' in data[4] else []
        return x, reg