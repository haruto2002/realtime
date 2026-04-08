import numpy as np
import os
import glob
from tqdm import tqdm
import yaml
from multiprocessing import Pool
import argparse

import warnings

warnings.filterwarnings("ignore")


def create_full_detection(source_dir, save_dir):
    full_img_size, patch_size = load_info(source_dir)
    patch_size = patch_size[::-1]
    frame_list = sorted([d for d in glob.glob(f"{source_dir}/*") if os.path.isdir(d)])
    pool_list = []
    for frame_path in tqdm(frame_list):
        frame_name = frame_path.split("/")[-1]
        patch_list = glob.glob(f"{frame_path}/data/*.txt")
        frame_save_path = f"{save_dir}/{frame_name}.txt"
        pool_list.append((patch_list, patch_size, frame_save_path))

    poool_size = os.cpu_count()
    with Pool(poool_size) as p:
        list(
            tqdm(
                p.imap_unordered(parallel_merge_patch, pool_list), total=len(pool_list)
            )
        )


def load_info(source_dir):
    with open(f"{source_dir}/input_info.yaml", "r") as f:
        info = yaml.safe_load(f)
    full_img_size = info["img_size"]
    img_size = info["separate_size"]
    return full_img_size, img_size


def parallel_merge_patch(pool_list):
    patch_list, patch_size, frame_save_path = pool_list
    merge_patch(patch_list, patch_size, frame_save_path)


def merge_patch(patch_list, patch_size, frame_save_path):
    full_det = []
    for patch_path in patch_list:
        patch_pos = (
            int(os.path.basename(patch_path).split(".")[0].split("_")[1]),
            int(os.path.basename(patch_path).split(".")[0].split("_")[0]),
        )
        det = np.loadtxt(patch_path, delimiter=",")
        if len(det) == 0:
            continue
        if det.ndim == 1:
            det = det.reshape(1, det.shape[0])
        det[:, 0] = det[:, 0] + patch_pos[0] * patch_size[0]
        det[:, 1] = det[:, 1] + patch_pos[1] * patch_size[1]
        full_det.extend(det)
    full_det = np.array(full_det)
    np.savetxt(frame_save_path, full_det)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="demo/patch_detection")
    parser.add_argument("--save_dir", type=str, default="demo/full_detection")
    return parser.parse_args()


def main():
    args = get_args()
    source_dir = args.source_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    create_full_detection(source_dir, save_dir)


if __name__ == "__main__":
    main()
