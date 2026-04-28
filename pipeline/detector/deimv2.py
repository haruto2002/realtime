from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from modules.detector.DEIMv2.engine.core import YAMLConfig

MAP_MODEL_SIZE_TO_FILE_NAME = {
    "Atto": "deimv2_hgnetv2_atto_coco",
    "Femto": "deimv2_hgnetv2_femto_coco",
    "Pico": "deimv2_hgnetv2_pico_coco",
    "N": "deimv2_hgnetv2_n_coco",
    "S": "deimv2_dinov3_s_coco",
    "M": "deimv2_dinov3_m_coco",
    "L": "deimv2_dinov3_l_coco",
    "X": "deimv2_dinov3_x_coco",
}

LABEL_MAP_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush",
}


class DEIMv2Detector:
    def __init__(
        self,
        model_size: str,
        cfg_dir: str,
        weight_dir: str,
        device: str,
        threshold: float,
    ):
        self.cfg_path = Path(cfg_dir) / f"{MAP_MODEL_SIZE_TO_FILE_NAME[model_size]}.yml"
        self.weight_path = (
            Path(weight_dir) / f"{MAP_MODEL_SIZE_TO_FILE_NAME[model_size]}.pth"
        )
        self.device = device
        self.model, self.img_size, self.vit_backbone = self.build_detector()
        self.transforms = self.build_transforms()
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

    def build_transforms(self):
        return T.Compose(
            [
                T.Resize(self.img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if self.vit_backbone
                else T.Lambda(lambda x: x),
            ]
        )

    def infer(self, image: np.ndarray):
        # image: cv2のnumpy配列 (BGR)

        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # numpy → PIL
        im_pil = Image.fromarray(image_rgb)

        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)

        im_data = self.transforms(im_pil).unsqueeze(0).to(self.device)

        output = self.model(im_data, orig_size)
        output_labels, output_boxes, output_scores = output
        output_labels = output_labels.detach().cpu().numpy()
        output_boxes = output_boxes.detach().cpu().numpy()
        output_scores = output_scores.detach().cpu().numpy()
        result = self.post_process(output_labels[0], output_boxes[0], output_scores[0])

        return result

    def infer_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """複数枚を1回の forward で推論する。各要素は cv2 の BGR (H,W,3)。"""
        if len(images) == 0:
            return []

        tensors: list[torch.Tensor] = []
        orig_rows: list[list[int]] = []
        for image in images:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(image_rgb)
            w, h = im_pil.size
            orig_rows.append([w, h])
            tensors.append(self.transforms(im_pil))

        im_data = torch.stack(tensors, dim=0).to(self.device)
        orig_size = torch.tensor(orig_rows, device=self.device)

        output = self.model(im_data, orig_size)
        output_labels, output_boxes, output_scores = output
        output_labels = output_labels.detach().cpu().numpy()
        output_boxes = output_boxes.detach().cpu().numpy()
        output_scores = output_scores.detach().cpu().numpy()

        results: list[np.ndarray] = []
        for i in range(len(images)):
            results.append(
                self.post_process(output_labels[i], output_boxes[i], output_scores[i])
            )
        return results

    def post_process(self, labels, boxes, scores):
        #  memo: deimv2の出力は元の画像サイズの座標系に戻されている
        labels = labels[scores > self.threshold]
        boxes = boxes[scores > self.threshold]
        scores = scores[scores > self.threshold]
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
            label_name = LABEL_MAP_ID_TO_NAME[int(label + 1)]
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
