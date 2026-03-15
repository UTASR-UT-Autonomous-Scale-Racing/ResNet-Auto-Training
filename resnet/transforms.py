import random
import torch
from torch import nn, Tensor
import torchvision
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
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



#OPENCV VERSION ONLY WORKS ON CPU. IF EFFICIENCY ISSUES ARISE, I WILL REWORK IT USING KORNIA.
class CleanTransform(nn.Module):
    """Adapted from https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
    https://docs.opencv.org/4.x/d5/dc4/tutorial_adding_images.html
    https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    """
    def __init__(self, clipLimit = 1.5, tileGridSize=(15, 15)) -> None:
        super().__init__()
        self.CLAHE = cv.createCLAHE(clipLimit, tileGridSize)

    def forward(self, img: torchvision.tv_tensors.Image, mask):
        #permute dimensions from standard tv_tensor (C,H,W) to (H,W,C) format.
        img = img.permute(1,2,0)

        #turn into numpyarray for cv, formatted (H,W,C)
        nparray = img.detach().cpu().numpy()
        nparray = cv.cvtColor(nparray, cv.COLOR_RGB2Lab)

        #apply histogram equalization
        nparray[:,:,0] = self.CLAHE.apply(nparray[:,:,0])
        nparray = cv.cvtColor(nparray, cv.COLOR_Lab2RGB)

        #use bilateralFilter to highlight edges
        nparray = cv.bilateralFilter(nparray, 5, 75, 75)

        #apply Laplacian function to get 3d array of edges
        ddepth = cv.CV_16S
        kernel_size = 3
        edges = cv.Laplacian(nparray, ddepth, ksize=kernel_size)
        edges = cv.convertScaleAbs(edges)
        lpWeight = 0.35
        nparray = cv.addWeighted(nparray, 1, edges, lpWeight, 0.0)

        #unsharp mask operation
        det = cv.GaussianBlur(nparray, (5, 5),0,0)
        nparray = cv.addWeighted(nparray, 3, det, -2, 0.0)
        img = torch.from_numpy(nparray)
        img = img.permute(2,0,1)
        img = torchvision.tv_tensors.Image(img)
        return img, mask

#For inference only, avoids converting between tensor and numpy array
def CleanTransformInference(nparray: NDArray, CLAHE):
    nparray = cv.cvtColor(nparray, cv.COLOR_RGB2Lab)
    nparray[:,:,0] = CLAHE.apply(nparray[:,:,0])
    nparray = cv.cvtColor(nparray, cv.COLOR_Lab2RGB)
    nparray = cv.bilateralFilter(nparray, 5, 75, 75)
    ddepth = cv.CV_16S
    kernel_size = 3
    edges = cv.Laplacian(nparray, ddepth, ksize=kernel_size)
    edges = cv.convertScaleAbs(edges)
    lpWeight = 0.35
    nparray = cv.addWeighted(nparray, 1, edges, lpWeight, 0.0)
    det = cv.GaussianBlur(nparray, (5, 5),0,0)
    nparray = cv.addWeighted(nparray, 3, det, -2, 0.0)
    return nparray


class Compose:
    """Compose transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def get_transform(train: bool, crop_size: Tuple[int, int] = (360, 640), clean = True):
    """
    Returns a transform pipeline for semantic segmentation.

    Args:
        train (bool): Whether this is training transform (includes augmentation).
        crop_size (Tuple[int,int]): Crop size (height, width) for RandomCrop.
    """
    transforms = []
    if clean == True:
        transforms.append(CleanTransform())
    if train:
        transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.append(RandomCrop(crop_size))

    transforms.append(ToTensor())

    # If using pretrained backbone, use ImageNet normalization
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transforms)
