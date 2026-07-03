import random
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as T
from typing import Tuple
import cv2 as cv
from numpy.typing import NDArray


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


class PILToTensor:
    """Convert a PIL image to a uint8 tensor (mask unchanged)."""

    def __call__(self, image, mask):
        if not isinstance(image, torch.Tensor):
            image = TF.pil_to_tensor(image)
        return image, mask


class ToTensor:
    """Scale the image to float [0, 1] and cast the mask to long."""

    def __call__(self, image, mask):
        return image.float() / 255.0, mask.long()


class Normalize:
    """Normalize image only (mask stays unchanged)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # Only normalize image
        image = TF.normalize(image, self.mean, self.std)
        return image, mask


def clean_image(nparray: NDArray, clahe) -> NDArray:
    """CLAHE + edge sharpening on a uint8 (H, W, C) RGB array.

    Adapted from https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
    https://docs.opencv.org/4.x/d5/dc4/tutorial_adding_images.html
    https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    """
    nparray = cv.cvtColor(nparray, cv.COLOR_RGB2Lab)
    nparray[:, :, 0] = clahe.apply(nparray[:, :, 0])
    nparray = cv.cvtColor(nparray, cv.COLOR_Lab2RGB)

    nparray = cv.bilateralFilter(nparray, 5, 75, 75)

    edges = cv.convertScaleAbs(cv.Laplacian(nparray, cv.CV_16S, ksize=3))
    nparray = cv.addWeighted(nparray, 1, edges, 0.35, 0.0)

    blurred = cv.GaussianBlur(nparray, (5, 5), 0, 0)
    return cv.addWeighted(nparray, 3, blurred, -2, 0.0)


class CleanTransform:
    """Apply clean_image to a uint8 (C, H, W) tensor and return one."""

    def __init__(self, clipLimit=1.5, tileGridSize=(15, 15)) -> None:
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.clahe = cv.createCLAHE(clipLimit, tileGridSize)

    def __getstate__(self):
        return {"clipLimit": self.clipLimit, "tileGridSize": self.tileGridSize}

    def __setstate__(self, state):
        self.__init__(state["clipLimit"], state["tileGridSize"])

    def __call__(self, image: torch.Tensor, mask):
        nparray = image.permute(1, 2, 0).cpu().numpy()
        nparray = clean_image(nparray, self.clahe)
        image = torch.from_numpy(nparray).permute(2, 0, 1)
        return image, mask


class Compose:
    """Compose transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_transform(train: bool, crop_size: Tuple[int, int] = (360, 640), clean=True):
    """
    Returns a transform pipeline for semantic segmentation.

    Args:
        train (bool): Whether this is training transform (includes augmentation).
        crop_size (Tuple[int,int]): Crop size (height, width) for RandomCrop.
    """
    transforms = [PILToTensor()]
    if clean:
        transforms.append(CleanTransform())
    if train:
        transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.append(RandomCrop(crop_size))

    transforms.append(ToTensor())

    # If using pretrained backbone, use ImageNet normalization
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transforms)
