import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np
import os
import glob
from multiprocessing import Pool


class DangerDataset(Dataset):
    """
    Dataset for danger prediction

    Args:
        img_dir: 画像ディレクトリ
        img_size: 元画像サイズ (height, width)
        separate_size: 分割サイズ (height, width)
        padded_size: パディング後のサイズ (height, width)
    """

    def __init__(
        self,
        img_dir,
        img_size=(4320, 7680),
        separate_size=(1152, 1280),
        padded_size=(4608, 7680),
    ):
        self.img_dir = img_dir
        self.img_size = img_size
        self.separate_size = separate_size
        self.padded_size = padded_size

        # 画像パスのリストを取得
        self.path2img_list = glob.glob(os.path.join(img_dir, "*.png"))
        self.path2img_list.sort()

        # 画像サイズの確認
        assert (
            cv2.imread(self.path2img_list[0]).shape[:2] == img_size
        ), "img size is wrong"

        # パディングの必要性確認
        self.pad = img_size != padded_size

        # 分割情報の計算
        self.h_num = padded_size[0] // separate_size[0]
        self.w_num = padded_size[1] // separate_size[1]
        self.total_patches = len(self.path2img_list) * self.h_num * self.w_num

        # Check if resize is needed
        height, width = separate_size
        new_height = height // 128 * 128
        new_width = width // 128 * 128
        self.trans_data = (
            None
            if (height == new_height and width == new_width)
            else (height, width, new_height, new_width)
        )

    def __len__(self):
        return self.total_patches

    def __getitem__(self, index):
        # Identify original image and patch position from index
        img_idx = index // (self.h_num * self.w_num)
        patch_idx = index % (self.h_num * self.w_num)
        h_idx = patch_idx // self.w_num
        w_idx = patch_idx % self.w_num

        # 画像の読み込み
        img = cv2.imread(self.path2img_list[img_idx])

        # パディング処理
        if self.pad:
            img = self._padding_img(img)

        # パッチの切り出し
        h_size, w_size = self.separate_size
        patch = img[
            h_size * h_idx : h_size * (h_idx + 1), w_size * w_idx : w_size * (w_idx + 1)
        ]

        # Preprocessing (normalization, etc.)
        patch = self._transform(patch)

        # 情報の作成
        info = {
            "img_name": os.path.splitext(os.path.basename(self.path2img_list[img_idx]))[
                0
            ],
            "num_parts": (self.h_num, self.w_num),
            "pos": (h_idx, w_idx),
        }

        return {"image": patch, "info": info}

    def _padding_img(self, img):
        """画像のパディング処理"""
        height, width = img.shape[:2]
        padded_img = np.zeros(
            (self.padded_size[0], self.padded_size[1], 3), dtype=img.dtype
        )
        padded_img[:height, :width, :] = img
        return padded_img

    def _transform(self, img):
        """画像の前処理（リサイズと正規化）"""
        if self.trans_data is not None:
            height, width, new_height, new_width = self.trans_data
            transform = A.Compose(
                [
                    A.Resize(new_height, new_width, interpolation=cv2.INTER_AREA),
                    A.Normalize(p=1.0),
                ]
            )
        else:
            transform = A.Compose(
                [
                    A.Normalize(p=1.0),
                ]
            )

        transformed = transform(image=img)
        transformed_img = transformed["image"]
        transformed_img = (
            np.array(transformed_img).astype(np.float32).transpose((2, 0, 1))
        )
        return torch.from_numpy(transformed_img).clone()


def collate_fn(batch):
    """バッチ処理用の関数"""
    images = []
    info = []
    for sample in batch:
        images.append(sample["image"])
        info.append(sample["info"])
    images = torch.stack(images, dim=0)
    return {"image": images, "info": info}


def create_dataloader(cfg, img_dir, **dataset_kwargs):
    """Create data loader"""
    dataset = DangerDataset(img_dir, **dataset_kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.optimizer.batch_size.test,
        shuffle=False,
        num_workers=cfg.default.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader
