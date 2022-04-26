import random
import numpy as np
from utils.config import obtain_retrain_args
from engine.retrainer import Trainer
from luojianet_ms import context, set_seed

# 设置所使用的GPU GRAPH_MODE
context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=1)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def main():
    args = obtain_retrain_args()
    setup_seed(args.seed)
    trainer = Trainer(args)

    print('Total Epoches:', args.epochs)
    for epoch in range(args.epochs):
        trainer.training(epoch)

if __name__ == "__main__":
    main()