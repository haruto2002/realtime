import os
import sys
from subprocess import getoutput

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


def get_ddp_settings(cfg):
    # Check number of GPUs
    local_size = torch.cuda.device_count()

    # Debug mode
    if cfg.default.dir_name == "debug":
        print("DEBBUGING!! Runing 1 GPU")
        rank_offset = 0
        local_size = 1
        world_size = local_size
        master = str(cfg.default.master)
        os.environ["MASTER_ADDR"] = master
        os.environ["MASTER_PORT"] = str(cfg.default.port)
    else:
        if not cfg.default.multi_node:
            # DDP on single node
            rank_offset = 0
            world_size = local_size
            master = str(cfg.default.master)
            os.environ["MASTER_ADDR"] = master
            os.environ["MASTER_PORT"] = str(cfg.default.port)
        else:
            # DDP on multi node
            # get host list and define master node
            hostlist = get_hostlist(cfg)
            num_node = len(hostlist)
            cfg.tmp.n_node = num_node
            master = hostlist[0]
            os.environ["MASTER_ADDR"] = str(master)
            os.environ["MASTER_PORT"] = str(cfg.default.port)
            print(f"host list: {hostlist} master:{master} port:{cfg.default.port}")

            # get own name, node rank
            hostname = getoutput("hostname").split(".")[0]
            node_rank = hostlist.index(hostname)

            # calc gpu ids
            rank_offset = local_size * node_rank
            world_size = local_size * len(hostlist)

    return master, rank_offset, world_size, local_size


def get_hostlist(cfg):
    with open(f"{cfg.default.work_dir}/hostfiles/hostfile_{cfg.tmp.job_id}") as f:
        hostlist = f.readlines()
    return [x.rstrip("\n") for x in hostlist]


def cuDNN_settings(cfg, rank):
    if cfg.default.rapid:
        cudnn.benchmark = True
        if rank == 0:
            print("cuddn benchmark True")
    else:
        cudnn.benchmark = False
    cudnn.deterministic = True


def init_ddp(local_rank, dist_settings, cfg):
    rank_offset, world_size, master = dist_settings
    global_rank = local_rank + rank_offset  # get global rank

    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)

    # print(
    #     "DDP setup: global rank=%d, local_rank=%d, world_size=%d, master=%s, seed=%d"
    #     % (global_rank, local_rank, world_size, master, cfg.default.r_seed)
    # )
    sys.stdout.flush()

    return global_rank, world_size
