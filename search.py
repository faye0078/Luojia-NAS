import random
import numpy as np
from utils.config import obtain_search_args
from engine.search_trainer import Trainer
from luojianet_ms import context, set_seed

# 设置所使用的GPU GRAPH_MODE
context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=0)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def main():
    args = obtain_search_args()
    setup_seed(args.seed)
    trainer = Trainer(args)

    print('Total Epoches:', args.epochs)
    print('Total Epoches:', args.epochs)
    trainer.training(args.epochs)

if __name__ == "__main__":
    main()