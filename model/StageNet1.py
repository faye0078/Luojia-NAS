import numpy as np
import luojianet_ms
from luojianet_ms import nn
from luojianet_ms import ops
import numpy
from model.cell import ReLUConvBN
from collections import OrderedDict


class SearchNet1(nn.Module):

    def __init__(self, layers, depth, connections, cell, dataset, num_classes, base_multiplier=40):
        '''
        Args:
            layers: layer × depth： one or zero, one means ture
            depth: the model scale depth
            connections: the node connections
            cell: cell type
            dataset: dataset
            base_multiplier: base scale multiplier
        '''
        super(SearchNet1, self).__init__()
        self.block_multiplier = 1
        self.base_multiplier = base_multiplier
        self.depth = depth
        self.layers = layers
        self.connections = connections
        self.node_add_num = np.zeros([len(layers), self.depth])

        half_base = int(base_multiplier // 2)
        if 'GID' in dataset:
            input_channel = 4
        else:
            input_channel = 3
        self.stem0 = nn.SequentialCell(
            nn.Conv2d(input_channel, half_base * self.block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(half_base * self.block_multiplier),
            nn.ReLU()
        )
        self.stem1 = nn.SequentialCell(
            nn.Conv2d(half_base * self.block_multiplier, half_base * self.block_multiplier, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_base * self.block_multiplier),
            nn.ReLU()
        )
        self.stem2 = nn.SequentialCell(
            nn.Conv2d(half_base * self.block_multiplier, self.base_multiplier * self.block_multiplier, 3, stride=2,padding=1),
            nn.BatchNorm2d(self.base_multiplier * self.block_multiplier),
            nn.ReLU()
        )
        self.cells = nn.CellList()

        self.cells_index = []
        multi_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        max_num_connect = 0
        num_last_features = 0
        for i in range(len(self.layers)):
            self.cells.append(nn.CellList())
            self.cells_index.append([])
            for j in range(self.depth):

                self.cells[i].append(nn.CellList())
                self.cells_index[i].append(OrderedDict())

                num_connect = 0
                for connection in self.connections:
                    if ([i, j] == connection[1]).all():
                        num_connect += 1
                        if connection[0][0] == -1:
                            self.cells[i][j].append((self.base_multiplier * multi_dict[0],
                                                         self.base_multiplier * multi_dict[connection[1][1]]))
                            self.cells_index[i][j][str(connection[0])] = int(len(self.cells[i][j]) - 1)
                        else:
                            self.cells[i][j].append(cell(self.base_multiplier * multi_dict[connection[0][1]],
                                                self.base_multiplier * multi_dict[connection[1][1]]))
                            self.cells_index[i][j][str(connection[0])] = int(len(self.cells[i][j]) - 1)
                self.node_add_num[i][j] = num_connect

                if i == len(self.layers) -1 and num_connect != 0:
                    num_last_features += self.base_multiplier * multi_dict[j]

                if num_connect > max_num_connect:
                    max_num_connect = num_connect

        self.last_conv = nn.SequentialCell(nn.Conv2d(num_last_features, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.max_num_connect = max_num_connect
        print('connections number: \n' + str(self.node_add_num))
        self.initialize_betas()


    def forward(self, x):
        features = []

        temp = self.stem0(x)
        temp = self.stem1(temp)
        pre_feature = self.stem2(temp)

        normalized_betas = ops.StandardNormal(len(self.layers), self.depth, self.max_num_connect).cuda()

        for i in range(len(self.layers)):
            for j in range(self.depth):
                num = int(self.node_add_num[i][j])
                if num == 0:
                    continue
                normalized_betas[i][j][:num] = ops.Softmax(self.betas[i][j][:num], dim=-1) * (num / self.max_num_connect)
                # if the second search progress, the denominato should be 'num'

        for i in range(len(self.layers)):
            features.append([])
            for j in range(self.depth):
                features[i].append(0)
                k = 0
                for connection in self.connections:
                    if ([i, j] == connection[1]).all():
                        if connection[0][0] == -1:
                            index = self.cells_index[i][j][str(connection[0])]
                            features[i][j] += normalized_betas[i][j][k] * self.cells[i][j][index](pre_feature)
                        else:
                            index = self.cells_index[i][j][str(connection[0])]
                            features[i][j] += normalized_betas[i][j][k] * self.cells[i][j][index](features[connection[0][0]][connection[0][1]])
                        k += 1

        last_features = [feature for feature in features[len(self.layers)-1] if feature != 0] # TODO: how to replace?
        last_features = [nn.ResizeBilinear()(feature, size=last_features[0].size()[2:], align_corners=True) for feature in last_features]
        result = ops.Concat(last_features, dim=1)
        result = self.last_conv(result)
        result = nn.ResizeBilinear()(result, size=x.size()[2:], align_corners=True)
        return result

    def initialize_betas(self):
        betas = (1e-3 * ops.StandardNormal()((len(self.layers), self.depth, self.max_num_connect)))# TODO: 含义是什么？

        self._arch_parameters = [
            betas,
        ]
        self._arch_param_names = [
            'betas',
        ]

        [self.register_parameter(name, luojianet_ms.Parameter(param)) for name, param in
         zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if
                name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if
                name not in self._arch_param_names]

