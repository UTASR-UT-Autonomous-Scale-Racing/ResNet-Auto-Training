"""
Improvised from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import os

from PIL import Image
import torch
import transforms as my_transforms
from torchvision.io import read_image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import v2 as T


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
    def __init__(
        self,
        imgs,
        image_dir,
        target_dir=None,
        masks=None,
        inference=False,
        train_transforms=False,
    ):
        self.inference = inference
        assert (
            inference is False
            and target_dir is not None
            and masks is not None
            or inference is True
        ), "Training mode requires target_dir and masks"

        # imgs, masks must be aligned
        self.img_dir = image_dir
        self.target_dir = target_dir
        self.imgs = imgs
        self.masks = masks
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

    def __getitem__(self, idx):
        transform_type = idx // self._original_len
        img_idx = idx % self._original_len

        # load images and masks from disk
        img_path = os.path.join(self.img_dir, self.imgs[img_idx])
        img = Image.open(img_path).convert("RGB")

        if self.inference:
            img = self.transforms[transform_type](img)
            return img, {}

        mask_path = os.path.join(self.target_dir, self.masks[img_idx])
        # mask = Image.open(mask_path).convert("L")
        semantic_mask = read_image(mask_path)[0].long()
        # target = {"mask": mask}

        # transforms
        img, semantic_mask = self.transforms[transform_type](img, semantic_mask)

        return img, semantic_mask

    def __len__(self):
        return self.len


def get_segmentation_model(num_classes):
    """Build the Deeplabv3 model with MobileNetV3-Large backbone"""
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
    return model


def get_transform(train, transform_type=0):
    transforms = []
    if train:
        transforms.extend(available_transforms[transform_type])
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_device():
    """Returns the most appropriate device for torch

    Note: mps is dubious don't use or face hair loss
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    # return torch.device("mps")
    else:
        return torch.device("cpu")
