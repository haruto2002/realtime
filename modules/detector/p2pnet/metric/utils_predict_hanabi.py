import cv2
import numpy as np
import os
import torch
import albumentations as A
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import copy
import pandas as pd


def load_from_path(path2img, path2gt):
    img = cv2.imread(path2img)
    gt_points = np.loadtxt(path2gt)
    return img, gt_points


def predict(img_id, raw_img, raw_gt_points, model, device):
    raw_img, gt_points, trans_data = load_data(raw_img, raw_gt_points)

    # load input img and gt
    # raw_img, raw_gt_points, trans_data = load_data(path2img, path2gt)
    transformed_img = transform(raw_img, trans_data)
    sample = np.array(transformed_img).astype(np.float32).transpose((2, 0, 1))
    sample = torch.from_numpy(sample).clone()
    sample = sample.unsqueeze(0)
    sample = sample.to(device)

    # run inference
    outputs = model(sample)
    outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
    outputs_points = outputs["pred_points"][0]
    threshold = 0.5
    detect_scores = outputs_scores[outputs_scores > threshold].detach().cpu().numpy()
    detect_points = outputs_points[outputs_scores > threshold].detach().cpu().numpy()

    detect_points = reconstruction(transformed_img, detect_points, trans_data)

    if not len(detect_scores) == len(detect_points):
        raise ValueError("The number of detect_score and detect_points are not same!")

    detections = [
        {
            "img_id": img_id,
            "confidence": detect_scores[i],
            "coordinate": detect_points[i],
        }
        for i in range(len(detect_scores))
    ]

    gts = [
        {
            "img_id": img_id,
            "coordinate": gt_points[i],
            "match": False,
        }
        for i in range(len(gt_points))
    ]

    return detections, gts


def predict_slide(
    img_id, raw_img, raw_gt_points, model, device, h_size=720, w_size=1280
):
    # h_size = 2160
    # w_size = 3840
    # h_size = 720
    # w_size = 1280
    img_raw_list = separate(raw_img, h_size, w_size)

    # total_cnt = 0
    h_num = len(img_raw_list)
    w_num = len(img_raw_list[0])
    # total_img = [[[] for _ in range(w_num)] for _ in range(h_num)]
    full_detections = []
    for i in range(h_num):
        for j in range(w_num):
            pathch_img = img_raw_list[i][j]
            detections, _ = predict(img_id, pathch_img, [], model, device)
            for k in range(len(detections)):
                detections[k]["coordinate"] = [
                    detections[k]["coordinate"][0] + w_size * j,
                    detections[k]["coordinate"][1] + h_size * i,
                ]
            full_detections += detections
    # load gt in img
    height, width, _ = raw_img.shape
    gt_points = load_gt_in_img(raw_gt_points, width, height)
    full_gts = [
        {
            "img_id": img_id,
            "coordinate": gt_points[i],
            "match": False,
        }
        for i in range(len(gt_points))
    ]
    return full_detections, full_gts


def load_gt_in_img(gt_points, img_width, img_height):
    gt_points = np.array(gt_points)
    if len(gt_points) > 0:
        if gt_points.ndim == 1:
            gt_points = np.array([gt_points])
        gt_points = gt_points[0 <= gt_points[:, 0]]
        gt_points = gt_points[gt_points[:, 0] < img_width]
        gt_points = gt_points[0 <= gt_points[:, 1]]
        gt_points = gt_points[gt_points[:, 1] < img_height]
    return gt_points


def load_data(raw_img, raw_gt_points):
    # def load_data(path2img, path2gt):
    # load img
    # raw_img = cv2.imread(path2img)
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    # load gt
    height, width, _ = img.shape
    # raw_gt_points = np.loadtxt(path2gt)
    gt_points = load_gt_in_img(raw_gt_points, width, height)
    # Apply resize
    new_height = height // 128 * 128
    new_width = width // 128 * 128
    # new_height = height // 16 * 16
    # new_width = width // 16 * 16

    trans_data = height, width, new_height, new_width

    return img, gt_points, trans_data


def transform(img, trans_data):
    height, width, new_height, new_width = trans_data
    # transform
    transform = A.Compose(
        [
            A.Resize(new_height, new_width, interpolation=cv2.INTER_AREA),
            A.Normalize(p=1.0),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    transformed = transform(image=img, keypoints=[])
    transformed_img = transformed["image"]

    return transformed_img


def reconstruction(img, points, trans_data):
    height, width, new_height, new_width = trans_data
    if len(points) > 0:
        if points.ndim == 1:
            points = np.array([points])
        points[points[:, 0] < 0, 0] = 0
        points[new_width <= points[:, 0], 0] = new_width - 0.001
        points[points[:, 1] < 0, 1] = 0
        points[new_height <= points[:, 1], 1] = new_height - 0.001
    back_transform = A.Compose(
        [A.Resize(height, width, interpolation=cv2.INTER_AREA)],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    back_transformed = back_transform(image=img, keypoints=points)
    back_transformed_points = np.array(back_transformed["keypoints"])

    return back_transformed_points


def separate(img, h_size, w_size):
    height, width, _ = img.shape
    patch_list = []
    h_num = height // h_size
    w_num = width // w_size
    patch_list = [[[] for _ in range(w_num)] for _ in range(h_num)]
    for i in range(h_num):
        for j in range(w_num):
            patch = separate_method(
                img, h_size * i, h_size * (i + 1), w_size * j, w_size * (j + 1)
            )
            patch_list[i][j] = patch

    return patch_list


def separate_method(img, y_low, y_up, x_low, x_up):
    separated_img = img[y_low:y_up, x_low:x_up]
    return separated_img
