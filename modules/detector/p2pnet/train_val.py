# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# the training routine
def train(
    cfg,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch,
    grad_scaler,
    global_rank,
):
    model.train()
    criterion.train()

    # iterate all training samples
    all_loss, loss_ce, loss_points = 0, 0, 0
    maes, mses = [], []

    if cfg.default.bar and global_rank == 0:
        bar = tqdm(total=len(data_loader), leave=False)

    iter_num = 0
    for i, batch in enumerate(data_loader):
        samples, targets = batch["image"], batch["label"]
        if epoch % 500 == 1 and i == 0 and global_rank == 0:
            save_img(samples, targets, cfg.out_dir, epoch, "train")
        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.default.amp):
            outputs = model(samples)

            # calc the losses
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
            # print(loss_dict)

        # backward
        grad_scaler.scale(losses).backward()
        grad_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.network.clip_max_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        all_loss += losses
        loss_ce += loss_dict["loss_ce"]
        loss_points += loss_dict["loss_points"]
        # calc the metrics
        mae, mse = calcurate_metric(outputs, targets)
        maes.append(float(mae))
        mses.append(float(mse))
        iter_num += 1
        if cfg.default.bar and global_rank == 0:
            bar.update(1)

    # print(all_loss)
    # print(type(all_loss))
    all_loss = all_loss.item() / iter_num
    class_loss = loss_ce.item() / iter_num
    location_loss = loss_points.item() / iter_num
    # calc MAE, MSE（厳密にはRMSE）
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    if cfg.default.bar and global_rank == 0:
        bar.close()

    return all_loss, class_loss, location_loss, mae, mse


# the inference routine
@torch.no_grad()
def evaluate(cfg, model, criterion, data_loader, device, epoch, global_rank):
    model.eval()
    all_loss = 0
    loss_ce = 0
    loss_points = 0
    maes = []
    mses = []

    if cfg.default.bar and global_rank == 0:
        bar = tqdm(total=len(data_loader), leave=False)

    iter_num = 0
    for i, batch in enumerate(data_loader):
        samples, targets = batch["image"], batch["label"]
        if epoch % 500 == 1 and i == 0 and global_rank == 0:
            save_img(samples, targets, cfg.out_dir, epoch, "validation")
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=cfg.default.amp):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
        all_loss += losses
        loss_ce += loss_dict["loss_ce"]
        loss_points += loss_dict["loss_points"]
        mae, mse = calcurate_metric(outputs, targets)
        maes.append(float(mae))
        mses.append(float(mse))
        iter_num += 1
        if cfg.default.bar and global_rank == 0:
            bar.update(1)

    all_loss = all_loss.item() / iter_num
    class_loss = loss_ce.item() / iter_num
    location_loss = loss_points.item() / iter_num

    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))
    if cfg.default.bar and global_rank == 0:
        bar.close()

    return all_loss, class_loss, location_loss, mae, mse


def calcurate_metric(outputs, targets):
    # output >> {'pred_logits':[num_batch,proposal数,2(neg,pos)], 'pred_points':[num_batch,proposal数,2(x,y)]}
    # targets >> len(targets)=num_batch, リストの要素は辞書でdict_keys(['point', 'image_id', 'labels'])
    mae = 0
    mse = 0
    for i in range(len(targets)):
        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[
            :, :, 1
        ][i]
        # outputs_scores >> [num_batch,proposal数](posだけ抜き出した)
        gt_cnt = targets[i]["point"].shape[0]
        # 0.5 is used by default
        threshold = 0.5
        predict_cnt = int((outputs_scores > threshold).sum())
        # accumulate MAE, MSE
        mae += abs(predict_cnt - gt_cnt)
        mse += (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)

    return mae / len(targets), mse / len(targets)


def save_img(samples, targets, save_dir, index, train_or_val):
    images = samples.permute(0, 2, 3, 1).detach().cpu().numpy()
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    shownumber = len(images)
    if shownumber == 1:
        fig, ax = plt.subplots()
        for i, img in enumerate(images):
            img = img * std + mean
            img = img.astype(np.int64)
            ax.imshow(img)
            x = [coor[0] for coor in targets[i]["point"]]
            y = [coor[1] for coor in targets[i]["point"]]
            ax.scatter(x, y, s=7, c="lime")
            ax.axis("off")
    else:
        showaxis = 1
        while showaxis * showaxis < shownumber:
            showaxis += 1

        fig, axs = plt.subplots(showaxis, showaxis, tight_layout=True)
        axs = axs.ravel()
        for i, img in enumerate(images):
            img = img * std + mean
            img = img.astype(np.int64)
            axs[i].imshow(img)
            x = [coor[0] for coor in targets[i]["point"]]
            y = [coor[1] for coor in targets[i]["point"]]
            axs[i].scatter(x, y, s=7, c="lime")
        for i in range(showaxis**2):
            axs[i].axis("off")

    save_path = save_dir + f"samples/{train_or_val}"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + f"/{index}.png")
    plt.clf()
    plt.close()
