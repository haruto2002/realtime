import cv2
import numpy as np
import torch
import albumentations as A


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


def predict_points(img, model, device):
    height, width, _ = img.shape
    new_height = height // 128 * 128
    new_width = width // 128 * 128
    trans_data = height, width, new_height, new_width

    transformed_img = transform(img, trans_data)
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

    return detect_points, detect_scores


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


def predict_slide(raw_img, model, device, h_size=720, w_size=1280):
    img_raw_list = separate(raw_img, h_size, w_size)
    h_num = len(img_raw_list)
    w_num = len(img_raw_list[0])
    # full_detections = []
    # full_scores = []
    full_detections = np.array([]).reshape(0, 2)
    full_scores = np.array([])
    for i in range(h_num):
        for j in range(w_num):
            pathch_img = img_raw_list[i][j]
            detections, scores = predict_points(pathch_img, model, device)
            for k in range(len(detections)):
                detections[k, 0] += w_size * j
                detections[k, 1] += h_size * i
            # full_detections += list(detections)
            # full_scores += list(scores)
            if detections.shape[0] != 0:
                full_detections = np.concatenate([full_detections, detections])
                full_scores = np.concatenate([full_scores, scores])
    return full_detections, full_scores
