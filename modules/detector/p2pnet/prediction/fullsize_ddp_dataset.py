import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import albumentations as A
import cv2
import numpy as np
import os
import glob


class FullSizeDataset(Dataset):
    """
    Dataset that returns images at original size

    Args:
        img_dir: 画像ディレクトリ
    """

    def __init__(self, img_dir):
        self.img_dir = img_dir

        # 画像パスのリストを取得
        self.path2img_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if len(self.path2img_list) == 0:
            self.path2img_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            if len(self.path2img_list) == 0:
                raise FileNotFoundError(img_dir)

    def __len__(self):
        return len(self.path2img_list)

    def __getitem__(self, index):
        """画像をそのまま返す"""
        # 画像の読み込み
        img = cv2.imread(self.path2img_list[index])

        # 前処理（正規化）
        img, original_height, original_width, transformed_height, transformed_width = (
            self._transform(img)
        )

        # 画像名を取得
        img_name = os.path.splitext(os.path.basename(self.path2img_list[index]))[0]

        return {
            "image": img,
            "img_name": img_name,
            "original_height": original_height,
            "original_width": original_width,
            "transformed_height": transformed_height,
            "transformed_width": transformed_width,
        }

    def _transform(self, img):
        """画像の前処理（正規化）"""
        original_height, original_width, _ = img.shape
        transformed_height = original_height // 128 * 128
        transformed_width = original_width // 128 * 128
        transform = A.Compose(
            [
                A.Resize(
                    transformed_height, transformed_width, interpolation=cv2.INTER_AREA
                ),
                A.Normalize(p=1.0),
            ]
        )

        transformed = transform(image=img)
        transformed_img = transformed["image"]
        transformed_img = (
            np.array(transformed_img).astype(np.float32).transpose((2, 0, 1))
        )
        return (
            torch.from_numpy(transformed_img).clone(),
            original_height,
            original_width,
            transformed_height,
            transformed_width,
        )


def collate_fn(batch):
    """バッチ処理用の関数"""
    images = []
    img_names = []
    original_heights = []
    original_widths = []
    transformed_heights = []
    transformed_widths = []
    # バッチ内の各画像を収集
    for item in batch:
        images.append(item["image"])
        img_names.append(item["img_name"])
        original_heights.append(item["original_height"])
        original_widths.append(item["original_width"])
        transformed_heights.append(item["transformed_height"])
        transformed_widths.append(item["transformed_width"])
    # Stack images into tensor
    images_tensor = torch.stack(images, dim=0)

    return {
        "image": images_tensor,
        "img_name": img_names,
        "original_height": original_heights,
        "original_width": original_widths,
        "transformed_height": transformed_heights,
        "transformed_width": transformed_widths,
    }


def create_fullsize_dataloader(cfg, img_dir, global_rank, world_size, **dataset_kwargs):
    """Create data loader"""
    dataset = FullSizeDataset(img_dir, **dataset_kwargs)
    ddp_sampler = DistributedSampler(
        dataset=dataset,
        rank=global_rank,
        num_replicas=world_size,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.optimizer.batch_size.test,
        num_workers=cfg.default.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        sampler=ddp_sampler,
    )
    return dataloader
