import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import albumentations as A
import cv2
import numpy as np
import os
import glob


class PatchDataset(Dataset):
    """
    Dataset that generates patches by splitting images at once

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
        self.path2img_list = sorted(glob.glob(os.path.join(img_dir, "*.png")))

        # 画像サイズの確認
        assert (
            cv2.imread(self.path2img_list[0]).shape[:2] == img_size
        ), "img size is wrong"

        # パディングの必要性確認
        self.pad = img_size != padded_size

        # 分割情報の計算
        self.h_num = padded_size[0] // separate_size[0]
        self.w_num = padded_size[1] // separate_size[1]

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
        return len(self.path2img_list)

    def __getitem__(self, index):
        """Generate all patches from one image"""
        # 画像の読み込み
        img = cv2.imread(self.path2img_list[index])

        # パディング処理
        if self.pad:
            img = self._padding_img(img)

        # 画像を分割してパッチリストを作成
        patches = []
        patch_infos = []
        h_size, w_size = self.separate_size
        img_name = os.path.splitext(os.path.basename(self.path2img_list[index]))[0]

        for h_idx in range(self.h_num):
            for w_idx in range(self.w_num):
                # パッチの切り出し
                patch = img[
                    h_size * h_idx : h_size * (h_idx + 1),
                    w_size * w_idx : w_size * (w_idx + 1),
                ]

                # Preprocessing (normalization, etc.)
                patch = self._transform(patch)
                patches.append(patch)

                # パッチ情報の作成
                patch_infos.append(
                    {
                        "img_name": img_name,
                        "num_parts": (self.h_num, self.w_num),
                        "pos": (h_idx, w_idx),
                    }
                )

        return {
            "patches": patches,
            "patch_infos": patch_infos,
            "total_patches": self.h_num * self.w_num,
        }

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
    all_patches = []
    all_infos = []

    # Collect patches from each image in batch
    for item in batch:
        all_patches.extend(item["patches"])
        all_infos.extend(item["patch_infos"])

    # Stack patches into tensor
    patches_tensor = torch.stack(all_patches, dim=0)

    return {"image": patches_tensor, "info": all_infos}


def create_patch_dataloader(cfg, img_dir, global_rank, world_size, **dataset_kwargs):
    """Create data loader"""
    dataset = PatchDataset(img_dir, **dataset_kwargs)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=cfg.optimizer.batch_size.test,
    #     shuffle=False,
    #     num_workers=cfg.default.num_workers,
    #     collate_fn=collate_fn,
    #     pin_memory=True,
    #     persistent_workers=True
    # )
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
