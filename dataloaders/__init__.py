"""Dataset generators."""
import os
import luojianet_ms.dataset.engine as de
from dataloaders.datasets.uadataset import Uadataset
from dataloaders.datasets.cityscapes import Cityscapes

def make_data_loader(args, batch_size=4, run_distribute=False, is_train=True, raw=False):
    """
    Create dataset loader.

    Args:
        data_name (str): name of dataset.
        data_path (str): dataset path.
        batchsize (int): batch size.
        run_distribute (bool): distribute training or not.
        is_train (bool): set `True` while training, otherwise `False`.
        raw (bool): if set it `False`, return mindspore dateset engine,
                    otherwise return python generator.

    Returns:
        dataset: dataset loader.
        crop_size: model input size.
        num_classes: number of classes.
        class_weights: a list of weights for each class.
    """
    if args.dataset == "uadataset":
        num_classes = 12
        crop_size = (512, 512)
        if is_train:
            datasetA = Uadataset(args.data_path,
                                 num_samples=None,
                                 num_classes=12,
                                 multi_scale=False,
                                 flip=False,
                                 ignore_label=255,
                                 base_size=512,
                                 crop_size=crop_size,
                                 downsample_rate=1,
                                 scale_factor=16,
                                 mean=[0.40781063, 0.44303973, 0.35496944],
                                 std=[0.3098623 , 0.2442191 , 0.22205387],
                                 is_train=is_train,
                                 number=1)
            datasetB = Uadataset(args.data_path,
                                 num_samples=None,
                                 num_classes=12,
                                 multi_scale=False,
                                 flip=False,
                                 ignore_label=255,
                                 base_size=512,
                                 crop_size=crop_size,
                                 downsample_rate=1,
                                 scale_factor=16,
                                 mean=[0.40781063, 0.44303973, 0.35496944],
                                 std=[0.3098623, 0.2442191, 0.22205387],
                                 is_train=is_train,
                                 number=2)


            if raw:
                return datasetA, datasetB, crop_size, num_classes
            if run_distribute:
                datasetA = de.GeneratorDataset(datasetA, column_names=["image", "label"],
                                              num_parallel_workers=8,
                                              shuffle=True,
                                              num_shards=1, shard_id=int(args.gpu_id))
                datasetB = de.GeneratorDataset(datasetB, column_names=["image", "label"],
                                               num_parallel_workers=8,
                                               shuffle=True,
                                               num_shards=1, shard_id=int(args.gpu_id))
            else:
                datasetA = de.GeneratorDataset(datasetA, column_names=["image", "label"],
                                              num_parallel_workers=8,
                                              shuffle=True)
                datasetB = de.GeneratorDataset(datasetB, column_names=["image", "label"],
                                               num_parallel_workers=8,
                                               shuffle=True)
            datasetA = datasetA.batch(batch_size, drop_remainder=True)
            datasetB = datasetB.batch(batch_size, drop_remainder=True)

            return datasetA, datasetB, crop_size, num_classes

        elif not is_train:
            dataset = Uadataset(args.data_path,
                                 num_samples=None,
                                 num_classes=12,
                                 multi_scale=False,
                                 flip=False,
                                 ignore_label=255,
                                 base_size=512,
                                 crop_size=crop_size,
                                 downsample_rate=1,
                                 scale_factor=16,
                                 mean=[0.40781063, 0.44303973, 0.35496944],
                                 std=[0.3098623 , 0.2442191 , 0.22205387],
                                 is_train=is_train,
                                 number=1)
            if raw:
                return dataset, crop_size, num_classes
            if run_distribute:
                dataset = de.GeneratorDataset(dataset, column_names=["image", "label"],
                                               num_parallel_workers=8,
                                               shuffle=True,
                                               num_shards=1, shard_id=int(args.gpu_id))
            else:
                dataset = de.GeneratorDataset(dataset, column_names=["image", "label"],
                                               num_parallel_workers=8,
                                               shuffle=True)
            dataset = dataset.batch(batch_size, drop_remainder=True)

            return dataset, crop_size, num_classes
    else:
        raise ValueError("Unsupported dataset.")
