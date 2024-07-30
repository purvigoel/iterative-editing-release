from torch.utils.data import ConcatDataset, DataLoader, Subset
from VIBE.lib.dataset import *
import ipdb

def get_data_loaders(cfg):
    def get_2d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    def get_3d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(set='train', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    # ===== 2D keypoint datasets =====
    train_2d_dataset_names = cfg.TRAIN.DATASETS_2D
    if train_2d_dataset_names is None:
        data_3d_batch_size = cfg.TRAIN.BATCH_SIZE
        train_2d_loader = None
    else:
        train_2d_db = get_2d_datasets(train_2d_dataset_names)

        data_2d_batch_size = int(cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.DATA_2D_RATIO)
        data_3d_batch_size = cfg.TRAIN.BATCH_SIZE - data_2d_batch_size

        train_2d_loader = DataLoader(
            dataset=train_2d_db,
            batch_size=data_2d_batch_size,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
        )

    cfg.NUM_WORKERS = 0

    # ===== 3D keypoint datasets =====
    train_3d_dataset_names = cfg.TRAIN.DATASETS_3D
    train_3d_db = get_3d_datasets(train_3d_dataset_names)
    train_3d_loader = DataLoader(
        dataset=train_3d_db,
        batch_size=data_3d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    train_eval_3d_db = get_3d_datasets(train_3d_dataset_names)
    train_eval_3d_db.datasets[0].set_format = 'val'
    train_eval_3d_db_subset = Subset(train_eval_3d_db, list(range(min(500, len(train_eval_3d_db)))))
    train_eval_3d_loader = DataLoader(
        dataset=train_eval_3d_db_subset,
        batch_size=data_3d_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.NUM_WORKERS,
    )

    # ===== Evaluation dataset =====
    valid_db = eval(cfg.TRAIN.DATASET_EVAL)(set='val', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)

    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.NUM_WORKERS,
    )

    return train_2d_loader, train_3d_loader, valid_loader, train_eval_3d_loader