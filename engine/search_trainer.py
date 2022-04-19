import os
import numpy as np
import luojianet_ms as luojia
import luojianet_ms.nn as nn
from luojianet_ms import Model
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from tqdm import tqdm
from collections import OrderedDict

from dataloaders import make_data_loader
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.evaluator import Evaluator

from model.StageNet1 import SearchNet1
# from model.StageNet2 import SearchNet2
# from model.StageNet3 import SearchNet3

from utils.copy_state_dict import copy_state_dict
from model.cell import ReLUConvBN, MixedCell


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 定义保存
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        self.opt_level = args.opt_level

        # 定义dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.nclass = make_data_loader(args, **kwargs)

        self.step_size = dataset.get_dataset_size()
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

        lr = nn.dynamic_lr.cosine_decay_lr(args.min_lr, args.lr, args.epochs * step_size,
                                           step_size, args.min_lr)
        optimizer = nn.SGD(net.weight_parameters(), lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.net, self.optimizer = net, optimizer
        self.architect_optimizer = nn.Adam(self.net.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_weight_decay)
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
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

    def training(self, epoch):

        time_cb = TimeMonitor(data_size=self.step_size)
        loss_cb = LossMonitor()
        config_ck = CheckpointConfig(save_checkpoint_steps=5,
                                     keep_checkpoint_max=100)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=self.args.save_checkpoint_path, config=config_ck)
        cb = [time_cb, loss_cb, ckpt_cb]


        self.model.train(self.args.epochs, self.dataset, callbacks=cb, sink_size=self.step_size, dataset_sink_mode=False)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r', ncols=80)
        test_loss = 0.0

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        is_best = False

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best, 'epoch{}_checkpoint.pth.tar'.format(str(epoch + 1)))

        self.saver.save_train_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU, is_best)