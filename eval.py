import glob
import os

import cv2
import numpy as np
from mean_average_precision import MetricBuilder
from tqdm import tqdm

from processor.components import Detector
from pathlib import Path


def main():
    model="yolo26"
    if model == "yolo26":
        config = f"img_conf/detector/yolo26/yolo26.yaml"
    elif model == "deimv2":
        config = f"img_conf/detector/deimv2/deimv2.yaml"
    else:
        raise ValueError(f"Invalid model: {model}")
    devide_size = None
    detector = Detector(Path(config), devide_size)
    print("Detector Set")

    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=True, num_classes=1
    )
    print("Metric Builder Set")

    dataset_dir = "samples/img"
    img_dir = os.path.join(dataset_dir, "images")
    gt_dir = os.path.join(dataset_dir, "labels")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    assert len(img_files) == len(gt_files)

    save_dir = f"vis/{model}"
    os.makedirs(save_dir, exist_ok=True)
    for img_file, gt_file in tqdm(zip(img_files, gt_files), total=len(img_files)):
        name = img_file.split("/")[-1].split(".")[0]
        image, gt = load_dataset(img_file, gt_file)
        # gt -> [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        # pred -> [xmin, ymin, xmax, ymax, class_id, confidence]
        pred = detector.infer(image)  # [xmin, ymin, xmax, ymax, label, score]
        pred = filter_by_label(pred, 0)
        vis = display(image, gt, pred, threshold=0.1)
        cv2.imwrite(f"{save_dir}/{name}.jpg", vis)
        metric_fn.add(pred, gt)

    print(
        f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0.0, 1.01, 0.01), mpolicy='soft')['mAP']}"
    )


def load_dataset(
    path2img: str,
    path2gt: str,
):
    anno = np.loadtxt(path2gt, delimiter=" ", dtype=np.float32)  # [_, cx, cy, w, h]
    if anno.shape[0] == 0:
        anno = anno.reshape(-1, 5)
    ltbr = convert_xywh_to_tlbr(anno)

    gt = np.concatenate(
        [ltbr, np.zeros((anno.shape[0], 3))], axis=1
    )  # [xmin, ymin, xmax, ymax, 1, 1, 1]
    image = cv2.imread(path2img)
    height, width = image.shape[:2]
    gt[:, :4] *= [width, height, width, height]
    return image, gt


def convert_xywh_to_tlbr(anno: np.ndarray):
    ltbr = np.zeros((anno.shape[0], 4))
    ltbr[:, 0] = anno[:, 1] - anno[:, 3] / 2
    ltbr[:, 1] = anno[:, 2] - anno[:, 4] / 2
    ltbr[:, 2] = anno[:, 1] + anno[:, 3] / 2
    ltbr[:, 3] = anno[:, 2] + anno[:, 4] / 2
    return ltbr

def filter_by_label(pred: np.ndarray, label: int):
    return pred[pred[:, -2] == label]


def display(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, threshold: float):
    if gt is not None:
        for xmin, ymin, xmax, ymax, class_id, difficult, crowd in gt:
            cv2.rectangle(
                image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1
            )
    if pred is not None:
        for xmin, ymin, xmax, ymax, label, score in pred:
            if score < threshold:
                continue
            cv2.rectangle(
                image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1
            )
    return image


if __name__ == "__main__":
    main()
