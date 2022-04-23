from luojianet_ms import context, set_seed
import luojianet_ms.nn as nn
import luojianet_ms as luojia
from luojianet_ms import Model, ParameterTuple, ops
from model.StageNet1 import SearchNet1
import numpy as np
from model.cell import ReLUConvBN
# context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=3)
#
# rand = ops.UniformReal(seed=2)
# tensor1 = rand((2, 12, 32, 32))
# tensor2 = rand((2, 12, 32, 32))
# loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
# a = loss(tensor1, tensor2)
# print(a)
#
# class Example(nn.Module):
#     def __init__(self):
#         super(Example, self).__init__()
#         print('看看我们的模型有哪些parameter:\t', self.parameters_dict(), end='\n')
#         self.W1_params = luojia.Parameter(ops.StandardNormal(seed=1)((2, 3)), 'w1')
#         self.W2_params = luojia.Parameter(ops.StandardNormal(seed=2)((3, 4)), 'w2')
#         print('增加W1后看看：', self.parameters_dict(), end='\n')
#
#
#     def forward(self, x):
#         return x
#
# # a = Example()
# layers = np.ones([14, 4])
# connections = np.load('/media/dell/DATA/wy/Seg_NAS/model/model_encode/GID-5/14layers_mixedcell1_3operation/first_connect_4.npy')
# net = SearchNet1(layers, 4, connections, ReLUConvBN, 'GID', 5)
# betas = net.arch_parameters()
# print(betas)
from luojianet_ms import Parameter, Tensor
from luojianet_ms.ops.composite import GradOperation
from luojianet_ms import ops as P
from luojianet_ms import dtype as mstype

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.matmul = P.MatMul()
        self.w = Parameter(Tensor(np.array([1.0, 1.0], np.float32)), name='z')
        self.b = Parameter(Tensor(np.array([1.0, 1.0], np.float32)), name='b')

    def call(self, x, y):
        x = x * self.w[0]
        x = x * self.b[0]
        y = y * self.w[1]
        y = y * self.b[1]
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Module):

    def __init__(self, net, optimizer):
        super(GradNetWrtX, self).__init__()
        self.optimizer = optimizer
        self.net = net
        self.params = ParameterTuple(list(filter(lambda x: 'b' in x.name, self.get_parameters())))
        self.grad_op = GradOperation(get_by_list=True)

    def call(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)

x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
output = GradNetWrtX(Net())(x, y)
print(output)