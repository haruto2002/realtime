from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_component(conf_path: Path):
    cfg = OmegaConf.load(conf_path)
    component = instantiate(cfg)
    return component


class Detector:
    def __init__(self, cfg_path: Path):
        self.detector = build_component(cfg_path)

    def infer(self, image: np.ndarray) -> np.ndarray:
        return self.detector.infer(image)

    def infer_split(self, image: np.ndarray, center: int) -> np.ndarray:
        images = [image[:, :center], image[:, center:]]
        results = self.detector.infer_batch(images)
        merged_results = self.detector.merge_results(results[0], results[1], center)
        return merged_results

    def draw(self, image: np.ndarray, dets: np.ndarray) -> np.ndarray:
        return self.detector.display_result(image, dets)


class Tracker:
    def __init__(self, cfg_path: Path):
        self.tracker = build_component(cfg_path)

    def update(self, dets: np.ndarray) -> np.ndarray:
        return self.tracker.update(dets)

    def convert_to_tracker_inputs(self, dets):
        if dets.shape[1] == 3:  # point(x, y, score)
            return dets
        elif dets.shape[1] == 6:  # bbox(left, top, right, bottom, label, score)
            dets = [dets[:, :4], dets[:, 5:6]]
            dets = np.concatenate(dets, axis=1)
            return dets
        else:
            raise ValueError(f"Invalid detection shape: {dets.shape}")

    def draw(self, image, tracks):
        return self.tracker.draw(image, tracks)
