import glob

import numpy as np

import torch
from torch.utils.data import Dataset

import albumentations as A
import cv2
from dataset.dataset_utils import suggest_dataset_root_dir


class DatasetLoader(Dataset):
    def __init__(self, cfg, data_root, phase):
        self.cfg = cfg
        self.root_dir = f"{data_root}" + f"{phase}/"
        self.phase = phase
        self.load_img_anno_path()

        # if self.cfg.dataset.cache:
        #     self.manager = Manager()
        #     self.cache_img = self.manager.dict()
        #     self.cache_label = self.manager.dict()
        #     print(f"Load {self.phase} image !!!")
        #     print(self.img_list)
        #     for i in range(self.nSamples):
        #         print(self.img_list[i], self.img_map[i])
        #         img, gts = self.load_data(self.img_list[i], self.img_map[i])
        #         self.cache_img[i] = img
        #         self.cache_label[i] = gts

    def __len__(self):
        return self.nSamples

    def load_img_anno_path(self):
        dir_list = glob.glob(self.root_dir + "*")
        self.img_list = []
        self.img_map = {}

        for dir in dir_list:
            img_path = glob.glob(f"{dir}/*.jpg")
            anno_path = glob.glob(f"{dir}/*.txt")
            self.img_list.append(img_path[0])
            self.img_map[img_path[0]] = anno_path[0]
        # number of samples
        self.nSamples = len(self.img_list)

    def load_data(self, img_path, gt_path):
        # Load the images
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load ground truth points
        h, w, _ = img.shape
        points = np.loadtxt(gt_path)
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        else:
            points = points[points[:, 0] >= 0]
            points = points[points[:, 0] <= w]
            points = points[points[:, 1] >= 0]
            points = points[points[:, 1] <= h]
        return img, np.array(points)

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        # if self.cfg.dataset.cache:
        #     img = self.cache_img[index].copy()
        #     points = self.cache_label[index].copy()
        # else:
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        img, points = self.load_data(img_path, gt_path)

        if self.phase == "train":
            return self.apply_train_aug(img, points, img_path)

        else:
            return self.apply_test_aug(img, points, img_path)

    def apply_train_aug(self, img, points, img_path):
        transform_random_scale_list = []
        if self.cfg.dataset.name in ["UCF-QNRF", "NWPU-Crowd"]:
            # Set max side length
            if self.cfg.dataset.name == "UCF-QNRF":
                max_size = 1408
            elif self.cfg.dataset.name == "NWPU-Crowd":
                max_size = 1920

            # Add Letterbox
            if max_size < max(img.shape[:3]):
                transform_random_scale_list += [A.LongestMaxSize(max_size=max_size, p=1)]
        # Add Random scale
        transform_random_scale_list += [
            A.RandomScale(scale_limit=0.3, p=1),
            A.PadIfNeeded(
                min_height=self.cfg.dataset.crop_size,
                min_width=self.cfg.dataset.crop_size,
                value=(0, 0, 0),
                border_mode=cv2.BORDER_CONSTANT,
            ),
        ]
        # Setup random scale data aug
        transform_random_scale = A.Compose(
            transform_random_scale_list,
            keypoint_params=A.KeypointParams(format="xy"),
        )
        # Setup random crop
        if self.cfg.dataset.color_jitter and self.cfg.dataset.cut_out:
            transform_random_crop = A.Compose(
                [
                    A.RandomCrop(self.cfg.dataset.crop_size, self.cfg.dataset.crop_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.Affine(p=0.5, scale=(0.9, 1.1), rotate=(-10, 10)),
                    A.Cutout(
                        p=0.5,
                        num_holes=1,
                        max_w_size=self.cfg.dataset.crop_size // 4,
                        max_h_size=self.cfg.dataset.crop_size // 4,
                    ),
                    A.ColorJitter(
                        brightness=(self.cfg.dataset.brightness),
                        contrast=self.cfg.dataset.contrast,
                        saturation=self.cfg.dataset.saturation,
                        hue=self.cfg.dataset.hue,
                        p=1.0,
                    ),
                    A.Normalize(p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        elif not self.cfg.dataset.color_jitter and self.cfg.dataset.cut_out:
            transform_random_crop = A.Compose(
                [
                    A.RandomCrop(self.cfg.dataset.crop_size, self.cfg.dataset.crop_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.Affine(p=0.5, scale=(0.9, 1.1), rotate=(-10, 10)),
                    A.Cutout(
                        p=0.5,
                        num_holes=1,
                        max_w_size=self.cfg.dataset.crop_size // 4,
                        max_h_size=self.cfg.dataset.crop_size // 4,
                    ),
                    A.Normalize(p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        elif self.cfg.dataset.color_jitter and not self.cfg.dataset.cut_out:
            transform_random_crop = A.Compose(
                [
                    A.RandomCrop(self.cfg.dataset.crop_size, self.cfg.dataset.crop_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.Affine(p=0.5, scale=(0.9, 1.1), rotate=(-10, 10)),
                    A.ColorJitter(
                        brightness=(self.cfg.dataset.brightness) / 255,
                        contrast=self.cfg.dataset.contrast,
                        saturation=self.cfg.dataset.saturation / 255,
                        hue=self.cfg.dataset.hue / 180,
                        p=1.0,
                    ),
                    A.Normalize(p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        else:
            transform_random_crop = A.Compose(
                [
                    A.RandomCrop(self.cfg.dataset.crop_size, self.cfg.dataset.crop_size, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.Affine(p=0.5, scale=(0.9, 1.1), rotate=(-10, 10)),
                    A.Normalize(p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        output_transform_random_scale = transform_random_scale(image=img, keypoints=points)
        crop_freq = self.cfg.dataset.crop_freq
        patch_images = []
        target = [{} for _ in range(crop_freq)]
        for i in range(crop_freq):
            num_trial = 0
            while True:
                output_transform_random_crop = transform_random_crop(
                    image=output_transform_random_scale["image"],
                    keypoints=output_transform_random_scale["keypoints"],
                )
                num_trial += 1
                if len(output_transform_random_crop["keypoints"]) > 0 or num_trial > 300:
                    break
            patch_point = output_transform_random_crop["keypoints"]
            if patch_point == []:
                patch_point = torch.tensor([], dtype=torch.float32).resize_(0, 2)
            else:
                patch_point = torch.tensor(patch_point, dtype=torch.float32)

            patch_img = output_transform_random_crop["image"]
            patch_images.append(patch_img)

            target[i]["point"] = patch_point
            image_id = int(img_path.split("/")[-2][6:])
            # image_id = int(img_path.split("/")[-1].split(".")[0].split("_")[-1])
            image_id = torch.tensor([image_id]).long()
            target[i]["image_id"] = image_id
            target[i]["labels"] = torch.ones([patch_point.shape[0]]).long()

        patch_images = np.array(patch_images).astype(np.float32).transpose((0, 3, 1, 2))
        img = torch.from_numpy(patch_images).clone()
        return {"image": img, "label": target}

    def apply_test_aug(self, img, points, img_path):
        # Apply letter box
        if self.cfg.dataset.name in ["UCF-QNRF", "NWPU-Crowd"]:
            if self.cfg.dataset.name == "UCF-QNRF":
                max_size = 1408
            elif self.cfg.dataset.name == "NWPU-Crowd":
                max_size = 1920
            if max_size < max(img.shape[:3]):
                trainform_letterbox = A.Compose(
                    A.LongestMaxSize(max_size=max_size, p=1.0),
                    keypoint_params=A.KeypointParams(format="xy"),
                )
                transfromed_letter_box = trainform_letterbox(image=img, keypoints=points)
                img = transfromed_letter_box["image"]
                points = transfromed_letter_box["keypoints"]

        # Apply resize
        height, width, _ = img.shape
        new_height = height // 128 * 128
        new_width = width // 128 * 128
        transform = A.Compose(
            [
                A.Resize(new_height, new_width, interpolation=cv2.INTER_AREA),
                A.Normalize(p=1.0),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )
        transformed = transform(image=img, keypoints=points)
        img = transformed["image"]
        point = transformed["keypoints"]

        if len(point) == 0:
            point = torch.tensor([], dtype=torch.float32).resize_(0, 2)
        else:
            point = torch.tensor(point, dtype=torch.float32)

        point = [point]  # batchsize=1扱いにするため
        target = [{}]
        target[0]["point"] = point[0]
        image_id = int(img_path.split("/")[-2][6:])
        image_id = torch.tensor([image_id]).long()
        target[0]["image_id"] = image_id
        target[0]["labels"] = torch.ones([point[0].shape[0]]).long()

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).clone()

        return {"image": img, "label": target}


def suggest_dataset(cfg):
    root_dir = suggest_dataset_root_dir(cfg)

    train_dataset = DatasetLoader(cfg, root_dir, "train")
    test_dataset = DatasetLoader(cfg, root_dir, "test")
    return train_dataset, test_dataset
