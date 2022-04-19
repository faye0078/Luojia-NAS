import luojianet_ms as luojia
from luojianet_ms import nn
import math
import numpy
import luojianet_ms.common.initializer as weight_init
from collections import OrderedDict
from model.ops import OPS, OPS_mini
from model.ops import conv3x3
class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out):
        super(ReLUConvBN, self).__init__()
        kernel_size = 3
        padding = 1
        self.scale = 1
        self.op = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.scale = C_in/C_out
        self._initialize_weights()

    def call(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                weight_init.initializer(weight_init.XavierUniform(),
                                        cell.weight.shape,
                                        cell.weight.dtype)

            elif isinstance(cell, nn.BatchNorm2d):
                if cell.weight is not None:
                    cell.weight.data.fill_(1)
                    cell.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class ConvBNReLU(nn.Module):

    def __init__(self, C_in, C_out):
        super(ConvBNReLU, self).__init__()
        kernel_size = 3
        padding = 1
        self.scale = 1
        self.op = nn.SequentialCell(
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU()
        )

        self.scale = C_in/C_out
        self._initialize_weights()

    def call(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                weight_init.initializer(weight_init.XavierUniform(),
                                        cell.weight.shape,
                                        cell.weight.dtype)

            elif isinstance(cell, nn.BatchNorm2d):
                if cell.weight is not None:
                    cell.weight.data.fill_(1)
                    cell.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))


class MixedCell(nn.Module):

    def __init__(self, C_in, C_out):
        super(MixedCell, self).__init__()
        kernel_size = 5
        padding = 2
        self.scale = 1
        self._ops = nn.CellList()
        self._ops_index = OrderedDict()
        for op_name in OPS:
            op = OPS[op_name](C_in, C_out, 1, True)
            self._ops.append(op)
            self._ops_index[op_name] = int(len(self._ops) - 1)
        self.ops_num = len(self._ops)
        self.scale = C_in/C_out
        self._initialize_weights()

    def call(self, x, cell_alphas):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return sum(w * self._ops[op](x) for w, op in zip(cell_alphas, self._ops))

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                weight_init.initializer(weight_init.XavierUniform(),
                                        cell.weight.shape,
                                        cell.weight.dtype)

            elif isinstance(cell, nn.BatchNorm2d):
                if cell.weight is not None:
                    cell.weight.data.fill_(1)
                    cell.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class MixedRetrainCell(nn.Module):

    def __init__(self, C_in, C_out, arch):
        super(MixedRetrainCell, self).__init__()
        self.scale = 1
        self._ops = nn.CellList()
        self._ops_index = OrderedDict()
        for i, op_name in enumerate(OPS):
            if arch[i] == 1:
                op = OPS[op_name](C_in, C_out, 1, True)
                self._ops.append(op)
                self._ops_index[op_name] = int(len(self._ops) - 1)
        self.ops_num = len(self._ops)
        self.scale = C_in/C_out
        self._initialize_weights()

    def call(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return sum(self._ops[op](x) for op in self._ops)

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                weight_init.initializer(weight_init.XavierUniform(),
                                        cell.weight.shape,
                                        cell.weight.dtype)

            elif isinstance(cell, nn.BatchNorm2d):
                if cell.weight is not None:
                    cell.weight.data.fill_(1)
                    cell.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))


class Fusion(nn.Module):

    def __init__(self, C_in, C_out):
        super(Fusion, self).__init__()

        self.scale = 1

        self.conv = nn.SequentialCell(
        conv3x3(C_in, C_out, 1),
        nn.BatchNorm2d(C_out, 1),
        nn.ReLU())
        self.scale = C_in / C_out
        self._initialize_weights()

    def call(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return self.conv(x)

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                weight_init.initializer(weight_init.XavierUniform(),
                                        cell.weight.shape,
                                        cell.weight.dtype)

            elif isinstance(cell, nn.BatchNorm2d):
                if cell.weight is not None:
                    cell.weight.data.fill_(1)
                    cell.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))