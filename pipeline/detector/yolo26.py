from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(
        self,
        model_size: str,
        weight_dir: str,
        device: str,
        img_size: tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
    ):
        assert model_size in ["n", "s", "m", "l", "x"], "Invalid model size"
        self.weight_path = Path(weight_dir) / f"yolo26{model_size}.pt"
        self.device = device
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.model = self.build_detector()

    def build_detector(self):
        model = YOLO(self.weight_path)
        model.to(self.device)
        model.eval()
        return model

    def infer(self, image: np.ndarray):
        result = self.model(
            image,
            imgsz=(self.img_size[1], self.img_size[0]),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        labels = result.boxes.cls.detach().cpu().numpy()
        result = np.concatenate(
            [
                boxes,
                np.expand_dims(labels, -1),
                np.expand_dims(scores, -1),
            ],
            axis=1,
        )
        return result

    def display_result(self, image: np.ndarray, result: np.ndarray):
        for left, top, right, bottom, label, score in result:
            left, top, right, bottom = map(int, [left, top, right, bottom])
            color = self.get_color(label)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            label_name = self.model.names[int(label)]
            cv2.putText(
                image,
                label_name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3,
            )
        return image

    def get_color(self, label_id):
        color = (
            int((37 * label_id) % 255),
            int((17 * label_id) % 255),
            int((29 * label_id) % 255),
        )
        return color
