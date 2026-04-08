from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf


class Detector:
    def __init__(self, cfg_path: Path):
        self.cfg = OmegaConf.load(cfg_path)
        self.detector = self.build_detector(self.cfg)

    def build_detector(self, conf_path: Path):
        cfg = OmegaConf.load(conf_path)
        detector = instantiate(cfg)
        return detector

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.detector([image])[0]

    def display_result(self, image: np.ndarray, result: np.ndarray) -> np.ndarray:
        return self.detector.display_result([image], [result])[0]
