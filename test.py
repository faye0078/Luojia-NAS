from luojianet_ms import context, set_seed
import luojianet_ms.nn as nn
from luojianet_ms import Model, ParameterTuple, ops

context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=3)

rand = ops.UniformReal(seed=2)
tensor1 = rand((2, 12, 32, 32))
tensor2 = rand((2, 12, 32, 32))
loss = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
a = loss(tensor1, tensor2)
print(a)