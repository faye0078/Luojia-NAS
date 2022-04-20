import os
import numpy as np
import luojianet_ms as luojia
import luojianet_ms.nn as nn
from luojianet_ms import Model
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig
from tqdm import tqdm
from utils.callback import TimeLossMonitor, SegEvalCallback
from utils.config import hrnetw48_config
from collections import OrderedDict

from dataloaders import make_data_loader

from model.StageNet1 import SearchNet1
from model.seg_hrnet import get_seg_model
# from model.StageNet2 import SearchNet2
# from model.StageNet3 import SearchNet3

from utils.copy_state_dict import copy_state_dict
from model.cell import ReLUConvBN, MixedCell


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 定义保存
        # self.saver = Saver(args)
        # self.saver.save_experiment_config()

        # 定义dataloader
        kwargs = {'run_distribute': False, 'is_train': True, 'raw': False}
        self.loader, self.image_size, self.num_classes = make_data_loader(args, args.batch_size, **kwargs)

        self.step_size = self.loader.get_dataset_size()
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')

        if self.args.search_stage == "first":
            layers = np.ones([14, 4])
            connections = np.load(self.args.model_encode_path)
            model = SearchNet1(layers, 4, connections, ReLUConvBN, self.args.dataset, self.nclass)

        elif self.args.search_stage == "second":
            layers = np.ones([14, 4])
            connections = np.load(self.args.model_encode_path)
            core_path = np.load('/media/dell/DATA/wy/Seg_NAS/model/model_encode/core_path.npy').tolist()
            model = SearchNet2(layers, 4, connections, ReLUConvBN, self.args.dataset, self.nclass,
                               core_path=core_path)

        elif self.args.search_stage == "third":
            layers = np.ones([14, 4])
            connections = np.load(self.args.model_encode_path)
            # connections = 0
            net = SearchNet3(layers, 4, connections, MixedCell, self.args.dataset, self.nclass)

        else:
            model_config = hrnetw48_config
            net = get_seg_model(model_config, self.num_classes)

        self.lr = nn.dynamic_lr.cosine_decay_lr(args.min_lr, args.lr, args.epochs * self.step_size,
                                           self.step_size, 10)
        # net.weight_parameters()
        optimizer = nn.SGD(net.trainable_params(), self.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.net, self.optimizer = net, optimizer
        self.net.set_train(True)

        # self.architect_optimizer = nn.Adam(self.net.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_weight_decay)
        # Define Evaluator
        # 加载模型
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = luojia.load_checkpoint(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.net.load_param_into_net(checkpoint['state_dict'])
            copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer']) # TODO: The details in loading optimizer
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            self.start_epoch = 0

        self.model = Model(self.net, loss_fn=self.criterion, optimizer=self.optimizer, metrics={'acc'})

    def training(self):

        time_loss_cb = TimeLossMonitor(lr_init=self.lr)
        config_ck = CheckpointConfig(save_checkpoint_steps=self.step_size * 5,
                                     keep_checkpoint_max=100)
        ckpt_cb = ModelCheckpoint(prefix="search", directory=self.args.save_checkpoint_path, config=config_ck)
        cb = [time_loss_cb, ckpt_cb]

        # val callbacks
        kwargs = {'run_distribute': False, 'is_train': False, 'raw': False}
        eval_loader, _, _ = make_data_loader(self.args, batch_size=1, **kwargs)
        eval_cb = SegEvalCallback(eval_loader, self.net, self.num_classes, start_epoch=self.args.eval_start,
                                  save_path=self.args.save_checkpoint_path, interval=1)
        cb.append(eval_cb)

        # train callbacks
        self.model.train(self.args.epochs - self.start_epoch, self.loader, callbacks=cb, sink_size=self.step_size, dataset_sink_mode=False)