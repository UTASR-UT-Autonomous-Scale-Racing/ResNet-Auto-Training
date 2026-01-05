"""
Reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import os

import cv2
import numpy as np
from PIL import Image
import torch
import transforms as my_transforms
from torchvision.io import read_image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import v2 as T
from typing import Dict, List, Optional, Tuple, Union


DATA_ROOT = os.path.join("data", "dataset")
DATA_IMAGES = os.path.join(DATA_ROOT, "images")
DATA_TARGETS = os.path.join(DATA_ROOT, "targets")
NUM_CLASSES = 2
OUTPUT_ROOT = "checkpoints"

base_transforms = [
    [],  # Includes no transforms for inference and training
    [my_transforms.RandomHorizontalFlip(0.5)],
]

# Generate all permutations of the base transforms
available_transforms = (
    base_transforms  # + [list(p) for p in permutations(base_transforms) if p]
)


class MultiObjectMaskDataset(torch.utils.data.Dataset):
    imgs: List[str]
    img_dir: str
    masks: Optional[List[str]]
    target_dir: Optional[str]
    inference: bool
    transforms: list
    dilate_mask: bool  # to dilate, or not to dilate: that is the question
    dilate_kernel_dimension: int
    dilate_iterations: int
    _original_len: int
    len: int

    def __init__(
        self,
        imgs,
        image_dir,
        target_dir=None,
        masks=None,
        inference=False,
        train_transforms=False,
        dilate_mask=True,
        dilate_kernel_dimension=5,
        dilate_iterations=1,
    ) -> None:
        self.inference = inference
        if not inference:
            assert target_dir is not None, "Training mode requires target_dir"
            assert masks is not None, "Training mode requires masks"

        # imgs, masks must be aligned
        self.img_dir = image_dir
        self.target_dir = target_dir
        self.imgs = imgs
        self.masks = masks
        self.dilate_mask = dilate_mask
        self.dilate_kernel_dimension = dilate_kernel_dimension
        self.dilate_iterations = dilate_iterations
        self._original_len = len(self.imgs)

        if train_transforms:
            self.transforms = [
                get_transform(train_transforms, i)
                for i in range(len(available_transforms))
            ]
            self.len = self._original_len * (len(self.transforms))
        else:
            self.transforms = [get_transform(train_transforms)]
            self.len = self._original_len

    def __getitem__(
        self, idx
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict]]:
        transform_type = idx // self._original_len
        img_idx = idx % self._original_len

        # load images and masks
        img_path = os.path.join(self.img_dir, self.imgs[img_idx])
        img = Image.open(img_path).convert("RGB")

        if self.inference:
            img = self.transforms[transform_type](img)
            return img, {}

        assert self.target_dir is not None, "Training mode requires target_dir"
        assert self.masks is not None, "Training mode requires masks"

        mask_path = os.path.join(self.target_dir, self.masks[img_idx])
        mask = read_image(mask_path)[0].numpy().astype(np.uint8)

        # map 255 to 1
        mask[mask == 255] = 1

        # dilation
        if self.dilate_mask:
            assert self.dilate_kernel_dimension is not None
            kernel = np.ones(
                (self.dilate_kernel_dimension, self.dilate_kernel_dimension),
            )
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_iterations)

        semantic_mask = torch.from_numpy(mask).long()

        # transforms
        img, semantic_mask = self.transforms[transform_type](img, semantic_mask)

        return img, semantic_mask

    def __len__(self) -> int:
        return self.len


def get_segmentation_model(num_classes) -> torch.nn.Module:
    """Build the Deeplabv3 model with MobileNetV3-Large backbone"""
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
    return model


def get_transform(train, transform_type=0) -> T.Compose:
    transforms = []
    if train:
        transforms.extend(available_transforms[transform_type])
    transforms.extend(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )  # equivalent to ToTensor()
    return T.Compose(transforms)


def get_device() -> torch.device:
    """Returns the most appropriate device for torch

    Note: mps is dubious don't use or face hair loss
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    # return torch.device("mps")
    else:
        return torch.device("cpu")
