from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from modules.detector.DEIMv2.engine.core import YAMLConfig


class DEIMv2Detector:
    def __init__(
        self, cfg_path: Path, weight_path: Path, device: str, threshold: float
    ):
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.device = device
        self.model, self.img_size, self.vit_backbone = self.build_detector()
        self.threshold = threshold

    def build_detector(self):
        """Main function"""
        cfg = YAMLConfig(self.cfg_path, resume=self.weight_path)

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(self.weight_path, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        model = Model().to(self.device)
        img_size = cfg.yaml_cfg["eval_spatial_size"]
        vit_backbone = cfg.yaml_cfg.get("DINOv3STAs", False)

        return model, img_size, vit_backbone

    def infer(self, image: np.ndarray):
        # image: cv2のnumpy配列 (BGR)

        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # numpy → PIL
        im_pil = Image.fromarray(image_rgb)

        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        transforms = T.Compose(
            [
                T.Resize(self.img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if self.vit_backbone
                else T.Lambda(lambda x: x),
            ]
        )

        im_data = transforms(im_pil).unsqueeze(0).to(self.device)

        output = self.model(im_data, orig_size)
        output_labels, output_boxes, output_scores = output
        output_labels = output_labels.detach().cpu().numpy()
        output_boxes = output_boxes.detach().cpu().numpy()
        output_scores = output_scores.detach().cpu().numpy()
        result = self.post_process(
            output_labels[0],
            output_boxes[0],
            output_scores[0],
            orig_size.tolist()[0],
            self.img_size,
        )

        return result

    def post_process(self, labels, boxes, scores, image_size, resize_size):
        labels = labels[scores > self.threshold]
        boxes = boxes[scores > self.threshold]
        scores = scores[scores > self.threshold]
        # inverted_detect_boxes = self.invert_transform_boxes(
        #     image_size, resize_size, boxes
        # )
        result = np.concatenate(
            [
                boxes,
                np.expand_dims(labels, -1),
                np.expand_dims(scores, -1),
            ],
            axis=1,
        )
        return result

    def invert_transform_boxes(self, raw_size, resize_size, boxes):
        ratio_height = raw_size[0] / resize_size[0]
        ratio_width = raw_size[1] / resize_size[1]
        boxes[:, 0] *= ratio_width
        boxes[:, 1] *= ratio_height
        boxes[:, 2] *= ratio_width
        boxes[:, 3] *= ratio_height
        return boxes

    def display_result(self, image: np.ndarray, result: np.ndarray):
        for left, top, right, bottom, label, score in result:
            left, top, right, bottom = map(int, [left, top, right, bottom])
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        return image
