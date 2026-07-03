import os
import torch
from tqdm import tqdm
import utils
from helpers import (
    MultiObjectMaskDataset,
    cull_similar_frames,
    get_segmentation_model,
    DATA_IMAGES,
    DATA_TARGETS,
    DEVICE,
    OUTPUT_ROOT,
    NUM_CLASSES,
)

NUM_EPOCHS = 2
TRAIN_PARTITION = 0.7
TRAIN_LR = 5e-3
TRAIN_MOMENTUM = 9e-1
TRAIN_WEIGHT_DECAY = 5e-4
VAL_PARTITION = 0.2
BATCH_SIZE = 8
NUM_WORKERS = 4


if __name__ == "__main__":
    # create directory for saving checkpoints if it doesn't exist
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # our dataset has two classes only - background and object
    imgs = sorted(os.listdir(DATA_IMAGES))
    masks = sorted(os.listdir(DATA_TARGETS))

    print(f"Using {DEVICE}")
    # imgs, masks = cull_similar_frames(imgs, masks, DATA_IMAGES)

    indices = list(range(len(imgs)))
    train_indices = indices[: int(len(indices) * TRAIN_PARTITION)]
    val_indices = indices[
        int(len(indices) * TRAIN_PARTITION) : int(
            len(indices) * (TRAIN_PARTITION + VAL_PARTITION)
        )
    ]

    # split the dataset into train and validation sets
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
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # get the model using our helper function
    model = get_segmentation_model(NUM_CLASSES)

    # move model to device
    model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAIN_LR,
        momentum=TRAIN_MOMENTUM,
        weight_decay=TRAIN_WEIGHT_DECAY,
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, targets in tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [train]"
        ):
            images = torch.stack([img.to(DEVICE, non_blocking=True) for img in images])
            masks = torch.stack(
                [mask.to(DEVICE, non_blocking=True) for mask in targets]
            )

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, targets in tqdm(
                data_loader_val, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [val]"
            ):
                images = torch.stack(
                    [img.to(DEVICE, non_blocking=True) for img in images]
                )
                masks = torch.stack(
                    [mask.to(DEVICE, non_blocking=True) for mask in targets]
                )
                outputs = model(images)["out"]
                val_loss += criterion(outputs, masks).item()
        val_loss /= max(len(data_loader_val), 1)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - val loss: {val_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(OUTPUT_ROOT, f"deeplabv3_epoch_{epoch}.pth"),
        )

    torch.save(model.state_dict(), os.path.join(OUTPUT_ROOT, "deeplabv3_final.pth"))
