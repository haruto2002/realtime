import glob
import cv2
import albumentations as A
import numpy as np
import os
from multiprocessing import Pool

def parallel_process(args):
    path2img, pad, padded_size, separate_size = args
    img = cv2.imread(path2img)
    if pad:
        img = padding_img(img, padded_size)
    patch_list, info_list = separate(path2img, img, separate_size[0], separate_size[1])
    return patch_list, info_list

def set_data(img_dir, img_size=(4320, 7680), separate_size=(1152, 1280), padded_size=(4608, 7680)):
    # img_size = (4320, 7680)
    # separate_size = (720, 1280)
    path2img_list = glob.glob(f"{img_dir}/*.png")
    # path2img_list = path2img_list[:1]  # debug
    assert cv2.imread(path2img_list[0]).shape[:2] == img_size, "img size is wrong"

    if img_size == padded_size:
        pad = False
    else:
        pad = True

    # マルチプロセス処理の準備
    args_list = [(path2img, pad, padded_size, separate_size) for path2img in path2img_list]
    pool_size = os.cpu_count()
    
    # マルチプロセス処理の実行
    with Pool(processes=pool_size) as pool:
        results = pool.map(parallel_process, args_list)

    # 結果の集約
    img_list = []
    info_list = []
    for patch_list, img_info_list in results:
        img_list.extend(patch_list)
        info_list.extend(img_info_list)

    height, width = separate_size
    new_height = height // 128 * 128
    new_width = width // 128 * 128
    if height == new_height and width == new_width:
        trans_data = None
    else:
        trans_data = height, width, new_height, new_width

    return img_list, info_list, trans_data

def padding_img(img, padded_size):
    height, width = img.shape[:2]
    padded_img = np.zeros((padded_size[0], padded_size[1], 3), dtype=img.dtype)
    padded_img[:height, :width,:] = img
    return padded_img

def separate(path2img, img, h_size, w_size):
    height, width, _ = img.shape
    patch_list = []
    info_list = []
    h_num = height // h_size
    w_num = width // w_size
    assert height % h_size == 0 or width % w_size == 0, "can't separate"
    patch_list = [[] for _ in range(w_num * h_num)]
    num = 0
    for i in range(h_num):
        for j in range(w_num):
            patch = separate_method(
                img, h_size * i, h_size * (i + 1), w_size * j, w_size * (j + 1)
            )
            patch_list[num] = patch
            num += 1
            info_list.append(
                {
                    "img_name": path2img.split("/")[-1][:-4],
                    "num_parts": (h_num, w_num),
                    "pos": (i, j),
                }
            )
    return patch_list, info_list


def separate_method(img, y_low, y_up, x_low, x_up):
    separated_img = img[y_low:y_up, x_low:x_up]
    return separated_img


def reconstruction(img, points, trans_data):
    if trans_data is None:
        return points
    else:
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
