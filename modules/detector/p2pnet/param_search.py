import subprocess

import numpy as np

from utils.util_config import setup_config

if __name__ == "__main__":
    cfg = setup_config()
    np.random.seed(cfg.default.r_seed)

    trial = 1
    for i in range(cfg.sampler.iteration):
        out_dir = cfg.out_dir + f"trial_{trial:04}/"

        # Random sampling from define search space
        suggest_params = {}
        for param_info in cfg.sampler.params:
            name = param_info["name"]
            min_val = param_info["min_val"]
            max_val = param_info["max_val"]
            p = np.random.random() * (max_val - min_val) + min_val
            if name == "lr":
                suggest_params[f"{name}"] = 0.1**p
            elif name == "weight_decay":
                suggest_params[f"{name}"] = 5.0 * 0.1**p
            else:
                suggest_params[f"{name}"] = p

        subprocess.run(
            [
                "python",
                "p2pnet/main.py",
                "p2p",
                f"out_dir={out_dir}",
                f"optimizer.hp.lr={suggest_params['lr']}",
                f"optimizer.hp.momentum={suggest_params['momentum']}",
                f"optimizer.hp.weight_decay={suggest_params['weight_decay']}",
                f"optimizer.hp.lr_decay={suggest_params['lr_decay']}",
            ]
            + cfg.override_cmd
        )

        trial += 1
