from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from modules.detector.p2pnet.network.backbone import Backbone_VGG
from modules.detector.p2pnet.network.p2pnet import P2PNet


class P2PNetDetector:
    def __init__(
        self,
        cfg_path,
        weight_path,
        device,
        dtype,
        threshold,
        img_size,
    ):
        assert Path(cfg_path).exists(), (
            f"Config file does not exist: {cfg_path} Current working directory: {Path.cwd()}"
        )
        assert Path(weight_path).exists(), (
            f"Weight file does not exist: {weight_path} Current working directory: {Path.cwd()}"
        )
        self.cfg_path = cfg_path
        self.weight_path = weight_path
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.detector = self.build_detector()
        self.threshold = threshold
        if img_size is not None:
            self.image_size = (int(img_size[0]), int(img_size[1]))
            self.resize_size = (
                int(img_size[0] // 128 * 128),
                int(img_size[1] // 128 * 128),
            )
            self.transforms = self.build_transforms()
        else:
            self.image_size = None

    def build_detector(self):
        cfg = OmegaConf.load(self.cfg_path)
        backbone = Backbone_VGG(cfg.network.backbone, True, False)
        model = P2PNet(backbone, cfg.network.row, cfg.network.line, self.device)
        state_dict = torch.load(
            self.weight_path, weights_only=True, map_location=self.device
        )
        model.load_state_dict(state_dict)
        model.compile()
        model.to(self.device)
        model.eval()
        return model

    def build_transforms(self):
        return A.Compose(
            [
                A.Resize(
                    self.resize_size[0],
                    self.resize_size[1],
                    interpolation=cv2.INTER_AREA,
                ),
                A.Normalize(p=1.0),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )

    def transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=img, keypoints=[])
        transformed_img = transformed["image"]
        input_img = np.array(transformed_img).astype(np.float32).transpose((2, 0, 1))
        input_img = torch.from_numpy(input_img)
        return input_img

    def preprocess(self, img):
        if self.image_size is None:
            self.image_size = img.shape[:2]
            self.resize_size = (
                int(self.image_size[0] // 128 * 128),
                int(self.image_size[1] // 128 * 128),
            )
            self.transforms = self.build_transforms()

        transformed_img = self.transform(img)
        input_img = transformed_img.unsqueeze(0).to(self.device)
        return input_img

    def infer(self, image: np.ndarray):
        input = self.preprocess(image)
        with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype):
            outputs = self.detector(input)
            outputs_scores = (
                torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1]
                .detach()
                .cpu()
                .numpy()
            )
            outputs_points = outputs["pred_points"].detach().cpu().numpy()
            result = self.post_process(
                outputs_scores[0],
                outputs_points[0],
                self.resize_size,
                self.image_size,
            )
            return result

    def post_process(
        self,
        scores,
        points,
        resize_size,
        image_size,
    ):
        points = points[scores > self.threshold]
        scores = scores[scores > self.threshold]
        inverted_detect_points = self.invert_transform_points(
            image_size, resize_size, points
        )
        result = np.concatenate(
            [inverted_detect_points, np.expand_dims(scores, -1)], axis=1
        )
        return result

    def invert_transform_points(self, raw_size, resize_size, points):
        ratio_height = raw_size[0] / resize_size[0]
        ratio_width = raw_size[1] / resize_size[1]
        points[:, 0] *= ratio_width
        points[:, 1] *= ratio_height
        return points

    def display_result(self, image: np.ndarray, result: np.ndarray):
        for x, y, score in result:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
        return image
