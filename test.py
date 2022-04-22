from luojianet_ms import context, set_seed
import luojianet_ms.nn as nn
import luojianet_ms as luojia
from luojianet_ms import Model, ParameterTuple, ops

# context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=3)
#
# rand = ops.UniformReal(seed=2)
# tensor1 = rand((2, 12, 32, 32))
# tensor2 = rand((2, 12, 32, 32))
# loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
# a = loss(tensor1, tensor2)
# print(a)

class Example(nn.Module):
    def __init__(self):
        super(Example, self).__init__()
        print('看看我们的模型有哪些parameter:\t', self.parameters_dict(), end='\n')
        self.W1_params = luojia.Parameter(ops.StandardNormal(seed=1)((2, 3)), 'w1')
        self.W2_params = luojia.Parameter(ops.StandardNormal(seed=2)((3, 4)), 'w2')
        print('增加W1后看看：', self.parameters_dict(), end='\n')


    def forward(self, x):
        return x

a = Example()