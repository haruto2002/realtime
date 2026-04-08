import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from timm.scheduler import CosineLRScheduler

from p2pnet.network.backbone import Backbone_VGG
from p2pnet.network.loss_fn import SetCriterion_Crowd
from p2pnet.network.matcher import HungarianMatcher_Crowd
from p2pnet.network.p2pnet import P2PNet


def setup_device():
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    return device


def fixed_r_seed(cfg):
    random.seed(cfg.default.r_seed)
    np.random.seed(cfg.default.r_seed)
    torch.manual_seed(cfg.default.r_seed)
    torch.cuda.manual_seed(cfg.default.r_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def build_backbone(cfg):
    backbone = Backbone_VGG(cfg.network.backbone, True)
    return backbone


def build_matcher_crowd(cfg):
    matcher = HungarianMatcher_Crowd(
        cost_class=cfg.network.set_cost_class,
        cost_point=cfg.network.set_cost_point,
        matcher_cal=cfg.network.matcher_cal,
    )
    return matcher


# def suggest_network(cfg, device):
#     backbone = build_backbone(cfg)
#     model = P2PNet(backbone, cfg.network.row, cfg.network.line, device)
#     if cfg.default.finetune:
#         model.load_state_dict(torch.load(cfg.default.weight_path))

#     return model


def suggest_network(cfg, device):
    backbone = build_backbone(cfg)
    model = P2PNet(backbone, cfg.network.row, cfg.network.line, device)
    if cfg.default.finetune:
        state_dict = torch.load(
            cfg.network.init_weight, weights_only=True
        )  # torchを新しくしたので、weights_only=Trueを追加
        if cfg.network.init_weight.split("/")[-1] == "SHTechA.pth":
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
    model.compile()
    # model.compile(backend="aot_eager")
    return model


# def suggest_network_for_use(cfg, device):
#     backbone = build_backbone(cfg)
#     model = P2PNet(backbone, cfg.network.row, cfg.network.line, device)
#     return model


def suggest_loss_func(cfg):
    num_classes = 1
    weight_dict = {
        "loss_ce": cfg.network.class_loss_coef,
        "loss_points": cfg.network.point_loss_coef,
    }
    losses = ["labels", "points"]
    matcher = build_matcher_crowd(cfg)
    loss_func = SetCriterion_Crowd(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.network.eos_coef,
        losses=losses,
    )
    return loss_func


def suggest_optimizer(cfg, model):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.hp.lr,
        momentum=cfg.optimizer.hp.momentum,
        weight_decay=cfg.optimizer.hp.weight_decay,
        nesterov=True,
    )

    return optimizer


def suggest_scheduler(cfg, optimizer):
    if cfg.optimizer.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=optimizer.scheduler.step,
            gamma=cfg.optimizer.hp.lr_decay,
        )
    elif cfg.optimizer.scheduler.name == "cosine":
        if cfg.optimizer.scheduler.warmup:
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=cfg.default.epochs,
                lr_min=cfg.optimizer.hp.lr * cfg.optimizer.hp.lr_decay,
                warmup_t=cfg.optimizer.hp.lr_warmup_step,
                warmup_lr_init=cfg.optimizer.hp.lr_warmup_init,
                warmup_prefix=True,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.default.epochs,
                eta_min=cfg.optimizer.hp.lr * cfg.optimizer.hp.lr_decay,
            )
    return scheduler
