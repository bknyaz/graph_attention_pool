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
from utils import get_monet_coord, get_monet_degree

from torch_spline_conv import SplineBasis, SplineWeighting
from torch_geometric.nn.conv import SplineConv


class SplineCNN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64,64,64],
                 dropout=0.2,
                 n_hidden=0,
                 spring_loss=None,
                 GW=False,
                 spring_loss_weight=1.,
                 angles=True,  # True for images
                 coord_transform=False):  # False for images
        super(SplineCNN, self).__init__()

        self.gconv = nn.Sequential(*([nn.Sequential(SplineConv(in_channels=(in_features + 18) if layer == 0 else filters[layer - 1],
                                                 out_channels=f,
                                                 dim=2,
                                                 kernel_size=3), nn.ReLU(True)) for layer, f in enumerate(filters)]))

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
        self.angles = angles

        if spring_loss is not None and spring_loss[0] in ['unsup', 'sup', 'gt']:
            # self.coord_layer = ChebyGINLayer(in_features=in_features,
            #                                  out_features=2,
            #                                  K=1,
            #                                  n_hidden=0,
            #                                  aggregation='mean',
            #                                  activation=None,
            #                                  spring_loss=spring_loss)
            graph_layer_fn = lambda n_in, n_out, K_, n_hidden_, activation: ChebyGINLayer(in_features=n_in,
                                                                                          out_features=n_out,
                                                                                          K=K_,
                                                                                          n_hidden=n_hidden_,
                                                                                          aggregation='mean',
                                                                                          activation=activation,
                                                                                          spring_loss=spring_loss,
                                                                                          GW=GW,
                                                                                          spring_loss_weight=spring_loss_weight,
                                                                                          degree_coord=not angles)
            self.coord_layer = ChebyGIN(in_features=in_features,
                                        out_features=0,
                                        filters=[32, 32, 16],
                                        K=5,
                                        n_hidden=0,
                                        spring_loss=None,
                                        graph_layer_fn=graph_layer_fn)

    def forward(self, data):
        if self.spring_loss is not None:
            x = data[0].clone()
            if self.spring_loss[0] in ['unsup', 'sup', 'gt']:

                # data[0] = data[0][:, :, :1]
                data_coord = self.coord_layer(data)
                # print(x.shape, data_coord[0].shape)
                if self.spring_loss[0] == 'gt':
                    coord = data[4]['coord']
                    data[0] = torch.cat((x, data_coord[0], coord), dim=2)  # , data[4]['coord']
                    data[4]['coord'] = get_monet_coord(coord, data[1])[0]
                elif self.spring_loss[0] == 'sup':
                    coord = data_coord[4]['coord'] if len(data_coord) > 3 else data_coord[1]['coord']
                    # print(x.shape, data_coord[0].shape, coord.shape)
                    data[0] = torch.cat((x, data_coord[0], coord), dim=2)  # , data[4]['coord']
                else:
                    coord = data_coord[4]['coord'] if len(data_coord) > 3 else data_coord[1]['coord']
                    data[4]['coord'] = get_monet_coord(coord, data[1])[0]
                    data[0] = torch.cat((x, data_coord[0]), dim=2)

                data_coord = data_coord[4] if len(data_coord) > 3 else data_coord[1]
                if 'reg' in data_coord:
                    data[4]['reg'] = data_coord['reg']


            elif self.spring_loss[0] == 'gt':
                if self.angles:
                    data[4]['coord'] = get_monet_coord(data[4]['coord'], data[1])[0]
                else:
                    data[4]['coord'] = get_monet_degree(data[1])[0]
        else:
            if self.angles:
                data[4]['coord'] = get_monet_coord(data[4]['coord'], data[1])[0]
            else:
                data[4]['coord'] = get_monet_degree(data[1])[0]


        # Convert to sparse format

        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x, data[4]