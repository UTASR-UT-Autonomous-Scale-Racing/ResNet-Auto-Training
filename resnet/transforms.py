import random
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as T
from typing import Tuple


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class RandomCrop:
    """Random crop that applies the same crop to image & mask."""

    def __init__(self, size: Tuple[int, int]):
        self.crop = T.RandomCrop(size)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        i, j, h, w = self.crop.get_params(image, self.crop.size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return image, mask


class ToTensor:
    """Convert image and mask to torch.Tensor."""

    def __call__(self, image, mask):
        # If already a tensor, keep as is
        if not isinstance(image, torch.Tensor):
            image = TF.pil_to_tensor(image)
        if not isinstance(mask, torch.Tensor):
            mask = TF.pil_to_tensor(mask)

        # Ensure image is float and mask is long (for class labels)
        image = image.float() / 255.0
        mask = mask.long()
        return image, mask


class Normalize:
    """Normalize image only (mask stays unchanged)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # Only normalize image
        image = TF.normalize(image, self.mean, self.std)
        return image, mask


class Compose:
    """Compose transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_transform(train: bool, crop_size: Tuple[int, int] = (360, 640)):
    """
    Returns a transform pipeline for semantic segmentation.

    Args:
        train (bool): Whether this is training transform (includes augmentation).
        crop_size (Tuple[int,int]): Crop size (height, width) for RandomCrop.
    """
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.append(RandomCrop(crop_size))

    transforms.append(ToTensor())

    # If using pretrained backbone, use ImageNet normalization
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transforms)
