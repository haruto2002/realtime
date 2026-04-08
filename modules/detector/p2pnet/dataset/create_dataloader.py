import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.load_data import suggest_dataset


def collate_fn_crowd(batch):
    # re-organize the batch
    labels = []
    images = []
    for sample in batch:
        images += [img for img in sample["image"]]
        labels += [label for label in sample["label"]]
    images = torch.stack(images, dim=0)
    return {"image": images, "label": labels}


def collate_fn_test(batch):
    # re-organize the batch
    labels = []
    images = []
    for sample in batch:
        images.append(sample["image"])
        labels.append(sample["label"][0])
    images = torch.stack(images, dim=0)
    return {"image": images, "label": labels}


def suggest_data_loader(cfg, global_rank, world_size):
    train_dataset, test_dataset = suggest_dataset(cfg)

    if cfg.default.DDP:
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            rank=global_rank,
            num_replicas=world_size,
            shuffle=True,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.optimizer.batch_size.train,
            num_workers=cfg.default.num_workers,
            collate_fn=collate_fn_crowd,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler,
        )
        test_sampler = DistributedSampler(
            dataset=test_dataset,
            rank=global_rank,
            num_replicas=world_size,
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=cfg.optimizer.batch_size.test,
            num_workers=cfg.default.num_workers,
            collate_fn=collate_fn_test,
            pin_memory=True,
            drop_last=False,
            sampler=test_sampler,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn_crowd,
            num_workers=cfg.default.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.optimizer.batch_size.test,
            drop_last=False,
            collate_fn=collate_fn_test,
            num_workers=cfg.default.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )

    return train_loader, test_loader
