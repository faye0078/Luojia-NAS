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
        if args.data_path is None:
            return crop_size, num_classes
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
                             is_train=is_train)

    elif args.dataset == "cityscapes":
        num_classes = 19
        if is_train:
            multi_scale = True
            flip = True
            crop_size = (512, 1024)
        else:
            multi_scale = False
            flip = False
            crop_size = (1024, 2048)
        if args.data_path is None:
            return crop_size, num_classes
        dataset = Cityscapes(args.data_path,
                             num_samples=None,
                             num_classes=19,
                             multi_scale=False,
                             flip=flip,
                             ignore_label=255,
                             base_size=2048,
                             crop_size=crop_size,
                             downsample_rate=1,
                             scale_factor=16,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225],
                             is_train=is_train)
    else:
        raise ValueError("Unsupported dataset.")
    class_weights = dataset.class_weights
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
