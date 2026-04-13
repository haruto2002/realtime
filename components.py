from pathlib import Path

import cv2
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

    def draw(self, image: np.ndarray, dets: np.ndarray) -> np.ndarray:
        return self.detector.display_result(image, dets)


class Tracker:
    def __init__(self, cfg_path: Path):
        self.tracker = build_component(cfg_path)

    def update(self, dets: np.ndarray) -> np.ndarray:
        return self.tracker.update(dets)

    def draw(self, image, tracks):
        line_thickness = 3
        for i, track in enumerate(tracks):
            x1, y1, w, h = track.tlwh
            ntbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = track.track_id
            color = self.get_color(abs(obj_id))
            cv2.rectangle(
                image,
                ntbox[0:2],
                ntbox[2:4],
                color=color,
                thickness=line_thickness,
            )
        return image

    def get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color
