"""
Reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import os
import random

import torch
from tqdm import tqdm
import utils
from helpers import (
    DATA_IMAGES,
    DATA_TARGETS,
    NUM_CLASSES,
    OUTPUT_ROOT,
    MultiObjectMaskDataset,
    get_device,
    get_segmentation_model,
)


NUM_EPOCHS = 2
TRAIN_PARTITION = 0.7
TRAIN_LR = 5e-3
TRAIN_MOMENTUM = 9e-1
TRAIN_WEIGHT_DECAY = 5e-4
VAL_PARTITION = 0.2

NUM_WORKERS = 4


if __name__ == "__main__":
    # get torch device
    device = get_device()

    # create directory for saving checkpoints if it doesn't exist
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # our dataset has two classes only - background and object
    imgs = list(sorted(os.listdir(DATA_IMAGES)))
    masks = list(sorted(os.listdir(DATA_TARGETS)))
    indices = list(range(len(imgs)))
    train_indices = indices[: int(len(indices) * TRAIN_PARTITION)]
    val_indices = indices[
        int(len(indices) * TRAIN_PARTITION) : int(
            len(indices) * (TRAIN_PARTITION + VAL_PARTITION)
        )
    ]

    # split the dataset in train and test set
    random.shuffle(train_indices)
    train_imgs = [imgs[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    val_imgs = [imgs[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices]

    # setup preprocessing and reading of images and targets
    dataset_train = MultiObjectMaskDataset(
        train_transforms=True,
        imgs=train_imgs,
        image_dir=DATA_IMAGES,
        target_dir=DATA_TARGETS,
        masks=train_masks,
    )
    dataset_val = MultiObjectMaskDataset(
        train_transforms=False,
        imgs=val_imgs,
        image_dir=DATA_IMAGES,
        target_dir=DATA_TARGETS,
        masks=val_masks,
    )

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=8,
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # get the model using our helper function
    model = get_segmentation_model(NUM_CLASSES)

    # move model to device
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAIN_LR,
        momentum=TRAIN_MOMENTUM,
        weight_decay=TRAIN_WEIGHT_DECAY,
    )

    print(f"Training on {device.type}")

    for epoch in range(NUM_EPOCHS):
        model.train()

        for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = [img.to(device) for img in images]
            masks = [mask.to(device) for mask in targets]
            images = torch.stack(images)
            masks = torch.stack(masks)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(
            model.state_dict(),
            os.path.join(OUTPUT_ROOT, f"deeplabv3_epoch_{epoch}.pth"),
        )

    torch.save(model.state_dict(), os.path.join(OUTPUT_ROOT, "deeplabv3_final.pth"))
