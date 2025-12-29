"""
Improvised from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import os
import random

import torch
import utils
from helpers import MultiObjectMaskDataset, get_segmentation_model

if __name__ == "__main__":
    # select torch device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    # device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # create directory for saving checkpoints if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    train_partition = 0.7
    val_partition = 0.2

    # our dataset has two classes only - background and object
    num_classes = 2
    root = os.path.join("data", "dataset")
    imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
    masks = list(sorted(os.listdir(os.path.join(root, "targets"))))
    indices = list(range(len(imgs)))
    train_indices = indices[: int(len(indices) * train_partition)]
    val_indices = indices[
        int(len(indices) * train_partition) : int(
            len(indices) * (train_partition + val_partition)
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
        image_dir=os.path.join("data", "dataset", "images"),
        target_dir=os.path.join("data", "dataset", "targets"),
        masks=train_masks,
    )
    dataset_val = MultiObjectMaskDataset(
        train_transforms=False,
        imgs=val_imgs,
        image_dir=os.path.join("data", "dataset", "images"),
        target_dir=os.path.join("data", "dataset", "targets"),
        masks=val_masks,
    )

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=4, shuffle=False, collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = get_segmentation_model(num_classes)

    # move model to the right device
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            masks = [mask.to(device) for mask in targets]
            images = torch.stack(images)
            masks = torch.stack(masks)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}")
