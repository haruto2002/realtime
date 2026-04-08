import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
import cv2
from torch.utils.data import Dataset
import numpy as np

class DatasetLoader(Dataset):
    """
    img_list: list of image paths
    img_info: list of image info(info.keys() >> "image_id","num_parts"(num_x,num_y),"pos"(pos_x,pos_y))
    trans_data: transform data >> (height, width, new_height, new_width)
    """

    def __init__(self, img_list, info_list, trans_data):
        self.trans_data = trans_data
        self.img_list = img_list
        self.info_list = info_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        img = self.img_list[index]
        info = self.info_list[index]
        img = self.test_transform(img)
        return {"image": img, "info": info}

    def test_transform(self, img):
        if self.trans_data is not None:
            height, width, new_height, new_width = self.trans_data
            transform = A.Compose(
                [
                    A.Resize(new_height, new_width, interpolation=cv2.INTER_AREA),
                    A.Normalize(p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        else:
            transform = A.Compose(
                [
                    A.Normalize(p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        transformed = transform(image=img, keypoints=[])
        transformed_img = transformed["image"]

        transformed_img = np.array(transformed_img).astype(np.float32).transpose((2, 0, 1))
        transformed_img = torch.from_numpy(transformed_img).clone()

        return transformed_img


def collate_fn_test(batch):
    # re-organize the batch
    images = []
    info = []
    for sample in batch:
        images.append(sample["image"])
        info.append(sample["info"])
    images = torch.stack(images, dim=0)
    return {"image": images, "info": info}


def suggest_data_loader(cfg, global_rank, world_size, img_list, info_list, trans_data):

    testset = DatasetLoader(img_list, info_list, trans_data)

    if cfg.default.DDP:
        test_sampler = DistributedSampler(
            dataset=testset,
            rank=global_rank,
            num_replicas=world_size,
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=cfg.optimizer.batch_size.test,
            num_workers=cfg.default.num_workers,
            collate_fn=collate_fn_test,
            pin_memory=True,
            drop_last=False,
            sampler=test_sampler,
        )
    else:
        test_loader = DataLoader(
            testset,
            batch_size=cfg.optimizer.batch_size.test,
            drop_last=False,
            collate_fn=collate_fn_test,
            num_workers=cfg.default.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )

    return test_loader


