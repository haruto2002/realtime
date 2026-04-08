import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.create_dataloader import suggest_data_loader
from dataset.dataset_utils import copy_datasets
from network.vgg_ import get_pretrained_weight
from train_val import evaluate, train
from utils.util_config import setup_config
from utils.util_ddp import get_ddp_settings, init_ddp
from utils.util_dnn import (
    fixed_r_seed,
    suggest_loss_func,
    suggest_network,
    suggest_optimizer,
    suggest_scheduler,
)
from utils.util_saver import Saver

warnings.filterwarnings("ignore")


def main_ddp(cfg, dist_settings, nprocs):
    mp.spawn(main, args=(cfg, dist_settings), nprocs=nprocs, join=True)


def main(local_rank, cfg, dist_settings=None):
    # Fixed random seed
    fixed_r_seed(cfg)

    # init DDP
    global_rank, world_size = init_ddp(local_rank, dist_settings, cfg)

    # Show experimental settings
    if global_rank == 0:
        print(OmegaConf.to_yaml(cfg))

    # set device
    device = torch.device("cuda:%d" % local_rank)
    torch.cuda.set_device(device)

    train_loader, test_loader = suggest_data_loader(cfg, global_rank, world_size)
    model = suggest_network(cfg, device)
    if cfg.network.fix_param is not None:
        for name, p in model.named_parameters():
            if cfg.network.fix_param in name:
                p.requires_grad = False
    if cfg.default.DDP:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device, non_blocking=True)

    # cache clear
    torch.cuda.empty_cache()

    # convert model to DDP
    if cfg.default.DDP:
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # Define loss function
    loss_fn = suggest_loss_func(cfg)
    loss_fn.to(device, non_blocking=True)

    # Setup optmizer, scheduler, and gradinet-scaler
    optimizer = suggest_optimizer(cfg, model)
    scheduler = suggest_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=cfg.default.amp)

    # Setup Logger
    if global_rank == 0:
        saver = Saver(cfg)

    for epoch in range(1, cfg.default.epochs + 1):
        # Training
        train_all_loss, train_class_loss, train_location_loss, train_mae, train_mse = (
            train(
                cfg,
                model,
                loss_fn,
                train_loader,
                optimizer,
                device,
                epoch,
                scaler,
                global_rank,
            )
        )
        train_results = torch.tensor(
            [
                train_all_loss,
                train_class_loss,
                train_location_loss,
                train_mae,
                train_mse,
            ],
            dtype=torch.float32,
            device=device,
        )
        train_gather_results = [
            torch.zeros(5, dtype=torch.float32, device=device)
            for _ in range(world_size)
        ]

        # Validation

        eval_all_loss, eval_class_loss, eval_location_loss, eval_mae, eval_mse = (
            evaluate(
                cfg,
                model,
                loss_fn,
                test_loader,
                device,
                epoch,
                global_rank,
            )
        )
        val_results = torch.tensor(
            [
                eval_all_loss,
                eval_class_loss,
                eval_location_loss,
                eval_mae,
                eval_mse,
            ],
            dtype=torch.float32,
            device=device,
        )
        val_gather_results = [
            torch.zeros(5, dtype=torch.float32, device=device)
            for _ in range(world_size)
        ]

        # Synchronize GPU and gather all GPUs results
        dist.barrier()
        dist.all_gather(train_gather_results, train_results)
        dist.all_gather(val_gather_results, val_results)

        if global_rank == 0:
            # Save metric
            train_gather_results = np.array(
                [r.cpu().numpy() for r in train_gather_results]
            ).mean(axis=0)
            val_gather_results = np.array(
                [r.cpu().numpy() for r in val_gather_results]
            ).mean(axis=0)

            saver.save_results(
                epoch,
                train_gather_results,
                val_gather_results,
                optimizer.param_groups[-1]["lr"],
            )
            # Save model weights
            if epoch % 100 == 0:
                saver.save_weights(model, epoch, saver.best_val_mae)

            # Show results
            saver.show_results()
            saver.plot_log()

        # Update learning rate
        scheduler.step(epoch)

    # end DDP
    if cfg.default.DDP:
        dist.barrier()
        dist.destroy_process_group()
    saver.save_run_time()


if __name__ == "__main__":
    cfg = setup_config()
    copy_datasets(cfg)
    get_pretrained_weight()

    # DDP setting
    master, rank_offset, world_size, local_size = get_ddp_settings(cfg)
    dist_settings = [rank_offset, world_size, master]

    main_ddp(cfg, dist_settings, nprocs=local_size)
