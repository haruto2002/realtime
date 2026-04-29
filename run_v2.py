from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf


def main(cfg_path: Path):
    cfg = OmegaConf.load(cfg_path)
    app = instantiate(cfg)
    app.run()
