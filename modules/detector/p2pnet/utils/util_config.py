import glob
import os
import sys

from omegaconf import OmegaConf


def setup_config():
    args = sys.argv

    config_file_name = args[1]
    config_file_path = f"./p2pnet/conf/{config_file_name}.yaml"
    if os.path.exists(config_file_path):
        cfg = OmegaConf.load(config_file_path)
    else:
        raise "No YAML file !!!"

    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args_list=args[2:]))

    if "out_dir" not in cfg:
        output_dir_path = (
            f"{cfg.default.dir_name}/"
            + f"{cfg.dataset.name}/"
            + f"r_seed_{cfg.default.r_seed}/"
        )
        version_num = len(sorted(glob.glob(output_dir_path + "*")))
        output_dir_path += f"version_{version_num}/"
    else:
        output_dir_path = f"{cfg.out_dir}"
    os.makedirs(output_dir_path, exist_ok=True)

    out_dir_comp = {"out_dir": output_dir_path}
    cfg = OmegaConf.merge(cfg, out_dir_comp)

    config_name_comp = {"execute_config_name": config_file_name}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    config_name_comp = {"override_cmd": args[2:]}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    with open(os.path.join(output_dir_path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    return cfg


def update_cfg(cfg):
    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)


def override_original_config(cfg):
    config_file_path = f"p2pnet/conf/{cfg.execute_config_name}.yaml"
    if os.path.exists(config_file_path):
        original_cfg = OmegaConf.load(config_file_path)
    else:
        raise "No YAML file !!!"
    return OmegaConf.merge(original_cfg, cfg)
