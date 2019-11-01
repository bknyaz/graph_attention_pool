import numpy as np
import math
import scipy
import scipy.sparse
import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from chebygin import ChebyGINLayer, ChebyGIN
from utils import *
#from torch_spline_conv import SplineBasis, SplineWeighting
#from torch_geometric.utils.repeat import repeat


class CoordLayer(nn.Module):  # computes the membership of a neighbors to the variosu bins
    """
    Fixed layer computing gaussian weights as patch operator
    """

    def __init__(self, n=25, max_rho=5):
        super(CoordLayer, self).__init__()
        sigma_rho = 1
        sigma_rho_min = 1
        self.n = n
        self.mu = Parameter((torch.rand(1, 1, 1, n) * max_rho).float())
        self.sigma = Parameter((torch.rand(1, 1, 1, n) * sigma_rho + sigma_rho_min).float())

    def forward(self, coord, idx):
        coord = coord.unsqueeze(3)  # B,N,N,1
        B, N = coord.shape[:2]

        # w = torch.zeros(B, N, N, self.n, device='cuda')
        # mask[~idx] = 1
        # print(coord.shape)
        sz = (B, N, N, self.n)
        mu = self.mu.expand(sz)
        sigma = self.sigma.expand(sz)
        coord = coord.expand(sz)
        # idx = ~torch.isnan(coord)
        w = torch.exp(-0.5 * (coord[idx] - mu[idx]) ** 2 / (1e-14 + sigma[idx] ** 2))  # B,N,N,25
        # print(torch.sum(torch.isnan(w)))
        return w


class GaussianLayerSimple(nn.Module):  # computes the membership of a neighbors to the variosu bins
    """
    Fixed layer computing gaussian weights as patch operator
    """

    def __init__(self, n_coord, n=25, max_rho=5):
        super(GaussianLayerSimple, self).__init__()
        self.n_coord = n_coord
        self.n = n
        layers = []
        for i in range(n_coord):
            layers.append(CoordLayer(n=n, max_rho=max_rho))
        self.layers = nn.ModuleList(layers)

    def forward(self, coord):
        B, N = coord.shape[:2]
        idx = ~torch.isnan(coord[:, :, :, 0])
        # assert torch.allclose(idx.float(), (~torch.isnan(coord[:, :, :, 1])).float())
        for i in range(coord.shape[3]):
            # print(i, torch.sum(c))
            if i == 0:
                w = self.layers[i](coord[:, :, :, i], idx)  # B,N,N,25
            else:
                w = w * self.layers[i](coord[:, :, :, i], idx)  # B,N,N,25

        weights = torch.zeros(B, N, N, self.n, device='cuda')
        weights[idx] = w
        return weights  # B,N,N,25


class GaussianWeightsLayer(nn.Module):  # computes the membership of a neighbors to the variosu bins
    """
    Fixed layer computing gaussian weights as patch operator
    """

    def __init__(self, n_rho=5, n_theta=5, max_rho=5, n_coord=2):
        super(GaussianWeightsLayer, self).__init__()
        # We have 25 Gaussians for all pairs of combinations rho and theta
        self.n_rho = n_rho
        self.n_theta = n_theta
        self.sigma_rho_ = 1.0
        self.sigma_rho_min_ = 1.0
        self.max_rho = max_rho

        self.sigma_theta_ = 0.75
        self.sigma_theta_min_ = 0.75
        self.n_coord = n_coord

        n = n_rho * n_theta

        # Gaussian sigma 25x1
        self.sigma_rho = (torch.rand(1, 1, 1, n) * self.sigma_rho_ + self.sigma_rho_min_).float()
        self.sigma_rho = Parameter(self.sigma_rho)  # 1,1,1,25


        if n_coord > 1:
            # initialize pseudo-coordinates u(x, y)
            coords = [torch.rand(1, 1, 1, 1, n) * max_rho, torch.rand(1, 1, 1, 1, n) * 2 * np.pi - np.pi]
            self.coords = Parameter(torch.cat(coords, dim=0).float())  # 2,1,1,1,25

            # Gaussian sigma 25x1
            self.sigma_theta = (torch.rand(1, 1, 1, n) * self.sigma_theta_ + self.sigma_theta_min_).float()
            self.sigma_theta = Parameter(self.sigma_theta)
        else:
            # initialize pseudo-coordinates u(x, y)
            coords = torch.rand(1, 1, 1, 1, n) * max_rho
            self.coords = Parameter(coords.float())  # 2,1,1,1,25

    def forward(self, inputs):

        # print(inputs.shape)
        P_rho = inputs[:, :, :, 0].unsqueeze(3)  # this can have any shape
        # print(P_rho.shape, P_theta.shape)
        if P_rho.is_cuda:
            device = P_rho.get_device()
        else:
            device = 'cpu'

        mu_rho = self.coords[0]  # 25x1
        # mu_rho   = mu_rho.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 1x1x1x25
        # mu_theta = mu_theta.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 1x1x1x25

        # P_rho   = P_rho.unsqueeze(3)  # Bx75x75x1
        # P_theta   = P_theta.unsqueeze(3)  # Bx75x75x1

        sz = P_rho.shape
        mu_rho = mu_rho.expand((sz[0], sz[1], sz[2], mu_rho.size(3)))
        sigma_rho = self.sigma_rho.expand_as(mu_rho)
        P_rho = P_rho.expand_as(mu_rho)
        idx = ~torch.isnan(P_rho)

        weights = torch.zeros(P_rho.shape, device=device)
        weights_rho = torch.exp(
            -0.5 * (P_rho[idx] - mu_rho[idx]) ** 2 / (1e-14 + sigma_rho[idx] ** 2))  # Bx75x75x25

        if inputs.shape[3] > 1:

            assert self.n_coord == 2, (self.n_coord, inputs.shape)

            mu_theta = self.coords[1]  # 25x1

            P_theta = inputs[:, :, :, 1].unsqueeze(3)
            # computation theta weights
            mu_theta = mu_rho.expand((sz[0], sz[1], sz[2], mu_theta.size(3)))
            sigma_theta = self.sigma_theta.expand_as(mu_theta)
            P_theta = P_theta.expand_as(mu_theta)

            # idx2 = ~torch.isnan(P_theta)
            # assert torch.allclose(idx.float(), idx2.float())

            # first_angle = torch.zeros(P_theta.shape, device=device)
            first_angle = torch.abs(P_theta[idx] - mu_theta[idx])  # Bx75x75x25
            # second_angle = torch.zeros(P_theta.shape, device=device)
            second_angle = torch.abs(2 * np.pi - torch.abs(P_theta[idx] - mu_theta[idx]))  # Bx75x75x25

            # weights_theta = torch.zeros(P_theta.shape, device=device)
            weights_theta = torch.exp(-0.5 * torch.min(first_angle, second_angle) ** 2 / (
                    1e-14 + sigma_theta[idx] ** 2))  # Bx75x75x25

            # computation of the final membership
            # print(weights_rho.shape, weights_theta.shape)
            weights[idx] = weights_rho * weights_theta  # Bx75x75x25
        else:
            weights[idx] = weights_rho
        # print('s1', torch.sum(torch.isnan(weights)).data.cpu().numpy(), weights.data.cpu().numpy().size)
        # s1 = torch.sum(torch.isnan(weights))
        # s2 = torch.sum(weights)
        # assert s1 == 0, '%d nan values' % s1
        # assert s2 > 0, 'sum is %f' % s2

        return weights  # BxNxNx25

    def __repr__(self):
        return 'GaussianWeightsLayer({},{},{},{},{},{})'.format(self.n_coord, self.sigma_rho_,
                                                                self.sigma_rho_min_, self.max_rho,
                                                                self.sigma_theta_, self.sigma_theta_min_)

class MoNetLayer(nn.Module):  # apply the membership to the features of the various nodes
    def __init__(self, in_features, out_features,
                 n_coord=2, angles=True,
                 n_rho=5, n_theta=5,
                 coord_transform=False, n_relations=1, activation=None, bnorm=False):

        super(MoNetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_gauss = n_rho * n_theta
        self.angles = angles
        self.coord_transform = coord_transform
        self.activation = activation
        self.bnorm = bnorm
        if self.bnorm:
            self.bn = nn.BatchNorm1d(out_features)
        if angles:
            self.gauss = GaussianWeightsLayer(n_rho=n_rho, n_theta=n_theta, n_coord=n_coord)
        else:
            self.gauss = GaussianLayerSimple(n_coord=n_coord, n=self.n_gauss)

        if self.coord_transform:
            self.fc_coord = nn.Sequential(nn.Linear(2, 2), nn.Tanh())

        self.fc = nn.Linear(in_features * self.n_gauss, out_features)

    def __repr__(self):
        return 'MoNetLayer(in_features={0}, out_features={1}, n_gauss={2}, angles={3}, coord_transform={4})\n{5}\n{6}'.format(self.in_features,
                                                                                                                              self.out_features,
                                                                                                                              self.n_gauss,
                                                                                                                              self.angles,
                                                                                                                              self.coord_transform,
                                                                                                                              str(self.gauss),
                                                                                                                              str(self.fc))

    def forward(self, data):
        x, _, mask = data[:3]
        coord = data[4]['coord']

        if self.coord_transform:
            B, N, _, C = coord.shape  # B,N,N,2
            coord = self.fc_coord(coord.view(-1, C)).view(B, N, N, -1)

        weights = self.gauss(coord)  # B,N,N,25
        # x = x.permute(0, 2, 1).contiguous()
        B, N, F = x.shape  # B,N,F
        K = weights.shape[3]  # B,N,N,25
        weights = weights.permute(0, 3, 1, 2).contiguous().view(-1, N, N)  # B,N,N,25 -> B,25,N,N -> B*25,N,N
        x = x.unsqueeze(1).expand(B, K, N, F).contiguous().view(-1, N, F)  # B*25,N,F
        x = torch.bmm(weights, x)  # B*25,N,F
        x = x.view(B, K, N, F)  # B*25,N,F -> B,25,N,F
        x = x.permute(0, 2, 1, 3).contiguous()  # B,N,25,F

        x = self.fc(x.view(B, N, -1))  # B,N,n_out

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        # print(x.shape, weights.shape, mask.shape)

        x = x * mask.float()

        if self.bnorm:
            x = self.bn(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        if self.activation is not None:
            x = self.activation(x)

        return (x, data[1], mask, data[3], data[4])


class SplineConv(nn.Module):  # apply the membership to the features of the various nodes
    def __init__(self, in_features, out_features,
                 dim=2, angles=True, kernel_size=5, degree=1,
                 is_open_spline=True, activation=None):

        super(SplineConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.angles = angles
        self.activation = activation
        self.degree = degree
        kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
        self.register_buffer('kernel_size', kernel_size)
        K = kernel_size.prod().item()
        print('K=%d' % K, (K, in_features, out_features))
        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        size = K * in_features
        bound = 1.0 / math.sqrt(size)

        self.weight = Parameter((torch.rand(K, in_features, out_features) - 0.5) * 2 * bound)
        # print(self.weight.data.min(), self.weight.data.max())
        # self.weight = Parameter((torch.rand(K * in_features, out_features) - 0.5) * 2 * bound)
        self.bias = Parameter(torch.zeros(out_features))

        self.root = Parameter((torch.rand(in_features, out_features) - 0.5) * 2 * bound)

    def __repr__(self):
        return 'SplineConv(in_features={}, out_features={}, kernel_size={}, degree={}, is_open_spline={}'.format(self.in_features,
                                                                                                          self.out_features,
                                                                                                          self.kernel_size,
                                                                                                          self.degree,
                                                                                                          self.is_open_spline)

    def bspline_basis(self, n_splines, degree, n_points=100):
        # Modified from https://github.com/mdeff/cnn_graph/blob/master/lib/models.py
        # Create knot vector and a range of samples on the curve
        assert n_splines > degree
        kv = np.array([0] * degree + list(range(n_splines - degree + 1)) +
                      [n_splines - degree] * degree, dtype='int')  # knot vector
        u = np.linspace(0, n_splines - degree, n_points)  # samples range

        def coxDeBoor(k, d):
            # Test for end conditions
            if (d == 0):
                return ((u - kv[k] >= 0) & (u - kv[k + 1] < 0)).astype(int)

            denom1 = kv[k + d] - kv[k]
            term1 = 0
            if denom1 > 0:
                term1 = ((u - kv[k]) / denom1) * coxDeBoor(k, d - 1)

            denom2 = kv[k + d + 1] - kv[k + 1]
            term2 = 0
            if denom2 > 0:
                term2 = ((-(u - kv[k + d + 1]) / denom2) * coxDeBoor(k + 1, d - 1))

            return term1 + term2

        # Compute basis for each point
        b = np.column_stack([coxDeBoor(k, degree) for k in range(n_splines)])
        b[n_points - 1][-1] = 1

        return b


    def forward(self, data):
        x, A, mask = data[:3]
        B, N, F = x.shape  # B,N,F
        coord = data[4]['coord'].clone()
        x_input = x.clone()
        # idx = ~torch.isnan(coord[:, :, :, 0]).clone()
        # print(coord.shape, coord.min(), coord.max())#, idx.shape)
        # print(torch.sum(idx.float()))#, torch.sum(~idx))
        # c1, c2 = torch.split(coord, [1, 1], dim=3)
        # coord = torch.stack((c1.view(B, N, N)[idx], c2.view(B, N, N)[idx]), dim=1)
        # print(coord.shape, torch.sum(torch.isnan(coord)))
        # coord = coord.view(-1, 2)
        # Splines = torch.from_numpy(self.bspline_basis(5, self.degree, n_points=len(coord))).float().to('cuda')
        # print(Splines.shape)  #N,5

        basis, weight_index = SplineBasis.apply(coord.view(-1, coord.shape[-1]),
                                 self._buffers['kernel_size'],
                                 self._buffers['is_open_spline'], self.degree)
        # print(len(spline))
        # idx = ~torch.isnan(coord[:, :, :, 0]).view(-1, 1).expand_as(spline[0]).byte()
        #
        # idx = (A < 1e-5).byte().view(-1, 1).expand_as(spline[0])
        # print(idx.type(), idx.shape)
        # spline = (spline[0][idx], spline[1][idx])
        # spline = (torch.clamp(spline[0], -25, 25), torch.clamp(spline[1], -25, 25))
        # for i in range(len(spline)):
        #     d = spline[i]
        #     print(i, d.shape, d.min(), d.max())#, idx.shape)

        idx = torch.isnan(basis)
        basis[idx] = 0

        mask_weight = torch.ones(weight_index.shape, dtype=torch.long, device='cuda')
        mask_weight[idx] = 0

        weight_index = weight_index * mask_weight

        # weight_index[idx] = 0
        # spline = (spline0, spline1)

        # for i in range(len(spline)):
        #     d = spline[i]
        #     print(i, d.shape, d.min(), d.max())

        # assert torch.sum(torch.isnan(basis)) == 0
        # assert torch.sum(torch.isnan(weight_index)) == 0

        # weights = spline[0].view(B, N, N, -1).permute(0, 3, 1, 2).contiguous().view(-1, N, N)  # B,N,N,25 -> B,25,N,N -> B*25,N,N
        # x = x.unsqueeze(1).expand(B, 4, N, F).contiguous().view(-1, N, F)  # B*25,N,F
        # x = torch.bmm(weights, x)  # B*25,N,F
        # x = x.view(B, 4, N, F)  # B*25,N,F -> B,25,N,F
        # x = x.permute(0, 2, 1, 3).contiguous()  # B,N,25,F
        # print(x.shape, self.weight.shape)

        # x = torch.mm(x.view(B*N, -1), self.weight.t()).view(B, N, -1)
        x_j = x.view(-1, F)
        # print(x.shape, x_j.shape)
        # x, weight, basis, weight_index
        # assert torch.sum(torch.isnan(self.weight)) == 0
        # idx = torch.isnan(self.weight)
        # self.weight[idx] = 0
        # assert torch.sum(torch.isnan(self.weight)) == 0
        # print('good')
        x = SplineWeighting.apply(x_j, self.weight, basis, weight_index)
        # print(x.shape)

        # assert torch.sum(torch.isnan(x)) == 0

        x = x.view(B, N, -1)
        # print(x.shape)
        x = x + torch.mm(x_input.view(B*N, F), self.root).view(B, N, -1)

        x = x + self.bias
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        x = x * mask.float()

        if self.activation is not None:
            x = self.activation(x)

        return (x, data[1], mask, data[3], data[4])




class MoNet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64,64,64],
                 dropout=0.2,
                 n_hidden=0,
                 splines=False,
                 n_relations=1,
                 bnorm=False,
                 n_rho=5,
                 n_theta=5,
                 K=1,
                 spring_loss=None,
                 GW=False,
                 spring_loss_weight=1.,
                 distance='vc_l2angle',  # True for images
                 coord_transform=False):  # False for images
        super(MoNet, self).__init__()

        # Graph convolution layers
        # def __init__(self, rnd_state, n_in, n_out, n_coord=2, angles=True, n_rho=5, n_theta=5, max_rho=5,
        #              coord_transform=False)
        #  + 2 * int(spring_loss)

        if splines:
            self.gconv = nn.Sequential(*([SplineConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                     out_features=f,
                                                     dim=2,
                                                     kernel_size=2, is_open_spline=False, degree=1,
                                                     activation=nn.ReLU(inplace=True)) for layer, f in
                                          enumerate(filters)]))

        else:
            self.gconv = nn.Sequential(*([MoNetLayer(in_features=in_features if layer == 0 else filters[layer - 1],
                                                     out_features=f,
                                                     n_coord=1 + int(distance.find('degree_cross') >= 0 or distance.find('angle') >= 0),
                                                     n_relations=n_relations,
                                                     n_rho=n_rho, n_theta=n_theta, angles=distance.find('vc_l2') >= 0,
                                                     coord_transform=coord_transform,
                                                     activation=nn.ReLU(inplace=True),
                                                     bnorm=bnorm) for layer, f in enumerate(filters)]))


        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            fc.append(nn.ReLU(inplace=True))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

        self.spring_loss = spring_loss
        self.distance = distance



        if spring_loss is not None and spring_loss[0] in ['unsup', 'sup']:
            # self.coord_layer = ChebyGINLayer(in_features=in_features,
            #                                  out_features=2,
            #                                  K=1,
            #                                  n_hidden=0,
            #                                  aggregation='mean',
            #                                  activation=None,
            #                                  spring_loss=spring_loss)
            graph_layer_fn = lambda n_in, n_out, K_, n_hidden_, activation, add_noise: ChebyGINLayer(in_features=n_in,
                                                                                          out_features=n_out,
                                                                                          K=K_,
                                                                                          n_hidden=n_hidden_,
                                                                                          aggregation='mean',
                                                                                          activation=activation,
                                                                                          spring_loss=spring_loss,
                                                                                          add_noise=add_noise,
                                                                                          GW=GW,
                                                                                          spring_loss_weight=spring_loss_weight,
                                                                                          distance=distance)
            self.coord_layer = ChebyGIN(in_features=in_features,
                                        out_features=0,
                                        filters=[32, 32, 2],
                                        K=K,
                                        n_hidden=0,
                                        spring_loss=None,
                                        graph_layer_fn=graph_layer_fn)

        # self.init_weights()
        # for m in self.modules():
        #     if isinstance(m, FixupBasicBlock):
        #         nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
        #             2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
        #         nn.init.constant_(m.conv2.weight, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                print('initializing layer', m.weight.data.shape)
                #nn.init.xavier_normal(m.weight.data, gain=1.)
                if m.weight.data.shape[0] <= 2:
                    m.weight.data.fill_(0)
                else:
                    nn.init.kaiming_normal(m.weight.data)#, gain=1.5)
                m.bias.data.fill_(0.1)
            # elif isinstance(m, nn.BatchNorm1d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def forward(self, data):

        def set_coord():
            if self.distance.find('vc_l2') >= 0:
                data[4]['coord'] = get_monet_coord(data[4]['coord'], data[1], angles=self.distance.find('angle') >= 0, norm=False)[0]
            elif self.distance.find('degree_max') >= 0:
                data[4]['coord'] = get_indegree_coord(data[1])[0]
            elif self.distance.find('degree_cross') >= 0:
                data[4]['coord'] = get_monet_degree(data[1])[0]
            elif self.distance.find('adj') >= 0:
                A = data[1].unsqueeze(3).clone()
                idx_nan = A == 0
                A[idx_nan] = float('nan')
                data[4]['coord'] = A
            else:
                raise NotImplementedError(self.distance)

        if self.spring_loss is not None:
            # x = data[0].clone()
            if self.spring_loss[0] in ['unsup', 'sup']:

                # data[0] = data[0][:, :, :1]
                data_coord = self.coord_layer(data)
                # print(x.shape, data_coord[0].shape)
                # if self.spring_loss[0] == 'gt':
                #     coord = data[4]['coord']
                #     data[0] = torch.cat((x, data_coord[0], coord), dim=2)  # , data[4]['coord']
                #     data[4]['coord'] = get_monet_coord(coord, data[1])[0]
                # elif self.spring_loss[0] == 'sup':
                #     coord = data_coord[4]['coord'] if len(data_coord) > 3 else data_coord[1]['coord']
                #     # print(x.shape, data_coord[0].shape, coord.shape)
                #     data[0] = torch.cat((x, data_coord[0], coord), dim=2)  # , data[4]['coord']
                # else:
                #     coord = data_coord[4]['coord'] if len(data_coord) > 3 else data_coord[1]['coord']
                #     data[4]['coord'] = get_monet_coord(coord, data[1])[0]
                #     data[0] = torch.cat((x, data_coord[0]), dim=2)

                data_coord = data_coord[4] if len(data_coord) > 3 else data_coord[1]
                data[4]['coord'] = data_coord['coord']  #get_monet_coord(data_coord['coord'], data[1])[0]
                if 'reg' in data_coord:
                    data[4]['reg'] = data_coord['reg']

            elif self.spring_loss[0] == 'gt':
                set_coord()
        else:
            set_coord()

        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x, data[4]

#python main.py -D TU --dropout 0 --results None --model monet --spring_loss gt_edges --coord_aug --threads 4
# --log_interval 50 -d /mnt/data/bknyazev/data/graph_data/NCI1 --n_nodes 0