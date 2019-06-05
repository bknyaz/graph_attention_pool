import numpy as np
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self,
                 in_features,  # feature dimensionality in the current graph layer
                 in_features_prev,  # feature dimensionality in the previous graph layer
                 pool_type,
                 pool_arch,
                 kl_weight=None,
                 drop_nodes=True):
        super(AttentionPooling, self).__init__()
        self.pool_type = pool_type
        self.pool_arch = pool_arch
        self.kl_weight = kl_weight
        self.proj = None
        self.drop_nodes = drop_nodes
        self.is_topk = self.pool_type[2].lower() == 'topk'
        if self.is_topk:
            self.ratio = float(self.pool_type[3])  # r
            assert self.ratio > 0 and self.ratio <= 1, ('invalid top-k ratio', self.ratio, self.pool_type)
        else:
            self.threshold = float(self.pool_type[3])  # \tilde{alpha}
            assert self.threshold >= 0 and self.threshold <= 1, ('invalid pooling threshold', self.threshold, self.pool_type)

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
            if not self.is_topk and self.threshold > 0:
                print('For ground truth attention threshold should be 0, but it is %f' % self.threshold)
        else:
            raise NotImplementedError(self.pool_type[1])

    def __repr__(self):
        return 'AttentionPooling(pool_type={}, pool_arch={}, topk={}, proj={})'.format(self.pool_type,
                                                                                       self.pool_arch,
                                                                                       self.is_topk,
                                                                                       self.proj)

    def cosine_sim(self, a, b):
        return torch.mm(a, b.t()) / (torch.norm(a, dim=1, keepdim=True) * torch.norm(b, dim=1, keepdim=True))

    def mask_out(self, x, mask):
        return x.view_as(mask) * mask

    def drop_nodes_edges(self, x, A, mask):
        B, N, C = x.shape
        N_nodes = torch.sum(mask, dim=1).long()  # B
        N_nodes_max = N_nodes.max()

        # Drop nodes
        mask, idx = torch.topk(mask, N_nodes_max, dim=1, largest=True, sorted=False)
        x = torch.gather(x, dim=1, index=idx.unsqueeze(2).expand(-1, -1, C))

        # Drop edges
        A = torch.gather(A, 1, idx.unsqueeze(2).expand(-1, -1, N))
        A = torch.gather(A, 2, idx.unsqueeze(1).expand(-1, N_nodes_max, -1))

        return x, A, mask

    def forward(self, data):
        KL_loss = None
        x, A, mask, _, params_dict = data[:5]

        mask_float = mask.float()

        B, N, C = x.shape
        if self.pool_type[1] in ['gt', 'sup']:
            if 'node_attn' in params_dict:
                alpha_gt = params_dict['node_attn'].view(B, N)
            else:
                raise ValueError('ground truth node attention values node_attn required for %s' % self.pool_type)

        if self.pool_type[1] in ['unsup', 'sup']:
            attn_input = data[-1] if self.pool_arch[1] == 'prev' else x.clone()

            alpha_pre = self.mask_out(torch.exp(self.proj(attn_input)), mask_float).view(B, N)
            alpha = alpha_pre / (torch.sum(alpha_pre, dim=1, keepdim=True) + 1e-7)
            if self.pool_type[1] == 'sup':
                KL_loss_per_node = self.mask_out(F.kl_div(torch.log(alpha + 1e-14), alpha_gt, reduction='none'), mask_float)  # per node loss
                KL_loss = self.kl_weight * torch.mean(KL_loss_per_node.sum(dim=1) /
                                                      (params_dict['N_nodes'].float() + 1e-7))  # mean over nodes, then mean over batches
        else:
            alpha = alpha_gt

        x = x * alpha.view(B, N, 1)
        if N > 700:
            x = x * params_dict['N_nodes'].view(B, 1, 1).float()
        mask = (mask & (alpha.view_as(mask) > self.threshold)).view(B, N)

        if self.drop_nodes:
            x, A, mask = self.drop_nodes_edges(x, A, mask)

        mask_matrix = mask.unsqueeze(2) & mask.unsqueeze(1)
        A = A * mask_matrix.float()   # or A[~mask_matrix] = 0

        # Add additional losses regularizing the model
        if 'reg' not in params_dict:
            data[4]['reg'] = []
        if KL_loss is not None:
            data[4]['reg'].append(KL_loss)

        return [x, A, mask, *data[3:]]