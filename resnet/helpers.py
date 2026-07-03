import cv2
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm
import torch
import transforms as my_transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from typing import Dict, List, Optional, Tuple, Union


NUM_CLASSES = 2
DATA_ROOT = os.path.join("data", "dataset")
DATA_IMAGES = os.path.join(DATA_ROOT, "images")
DATA_TARGETS = os.path.join(DATA_ROOT, "targets")
OUTPUT_ROOT = "checkpoints"
DEVICE = "cuda"


class MultiObjectMaskDataset(torch.utils.data.Dataset):
    imgs: List[str]
    img_dir: str
    target_dir: Optional[str]
    masks: Optional[List[str]]
    inference: bool
    dilate_mask: bool  # to dilate, or not to dilate: that is the question
    dilate_kernel_dimension: int
    dilate_iterations: int

    def __init__(
        self,
        imgs,
        image_dir,
        target_dir=None,
        masks=None,
        inference=False,
        dilate_mask=True,
        dilate_kernel_dimension=5,
        dilate_iterations=1,
        train_transforms=False,
    ) -> None:
        self.imgs = imgs
        self.img_dir = image_dir
        self.target_dir = target_dir
        self.masks = masks
        self.inference = inference
        self.dilate_mask = dilate_mask
        self.dilate_kernel_dimension = dilate_kernel_dimension
        self.dilate_iterations = dilate_iterations
        self.transform = my_transforms.get_transform(
            train_transforms, clean=not inference
        )

    def __getitem__(
        self, idx
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict]]:
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.inference:
            img, _ = self.transform(img, torch.zeros((1, 1), dtype=torch.long))
            return img, {}

        assert self.target_dir is not None, "Training mode requires target_dir"
        assert self.masks is not None, "Training mode requires masks"

        mask_path = os.path.join(self.target_dir, self.masks[idx])
        mask_image = np.array(Image.open(mask_path).convert("L"), np.uint8)

        # binary mask
        mask = (mask_image > 127).astype(np.uint8)

        # dilation
        if self.dilate_mask:
            kernel = np.ones(
                (self.dilate_kernel_dimension, self.dilate_kernel_dimension), np.uint8
            )
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_iterations)

        semantic_mask = torch.from_numpy(mask).long()

        # transforms
        img, semantic_mask = self.transform(img, semantic_mask)

        return img, semantic_mask

    def __len__(self) -> int:
        return len(self.imgs)


def cull_similar_frames(
    imgs: List[str],
    masks: List[str],
    image_dir: str,
    threshold: float = 0.9,
    window: int = 30,
) -> Tuple[List[str], List[str]]:
    kept_imgs: List[str] = []
    kept_masks: List[str] = []
    win_imgs: List[str] = []
    win_masks: List[str] = []
    win_grey: List[np.ndarray] = []

    def is_novel(grey: np.ndarray) -> bool:
        return all(
            structural_similarity(prev, grey, data_range=255) <= threshold
            for prev in win_grey
        )

    for i in tqdm(range(len(imgs) - 1, -1, -1), desc="Culling similar frames"):
        grey = cv2.cvtColor(
            cv2.imread(os.path.join(image_dir, imgs[i])), cv2.COLOR_BGR2GRAY
        )
        if not win_imgs:
            win_imgs.append(imgs[i])
            win_masks.append(masks[i])
            win_grey.append(grey)
        elif is_novel(grey):
            if len(win_imgs) == window:
                kept_imgs.append(win_imgs.pop(0))
                kept_masks.append(win_masks.pop(0))
                win_grey.pop(0)
            win_imgs.append(imgs[i])
            win_masks.append(masks[i])
            win_grey.append(grey)

    kept_imgs.extend(win_imgs)
    kept_masks.extend(win_masks)
    return kept_imgs, kept_masks


def get_segmentation_model(num_classes) -> torch.nn.Module:
    """Build the Deeplabv3 model with MobileNetV3-Large backbone"""
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
    return model
