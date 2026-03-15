import os
import random
import torch
from tqdm import tqdm
import utils
from helpers import (
    MultiObjectMaskDataset,
    get_device,
    get_segmentation_model,
    DATA_IMAGES,
    DATA_TARGETS,
    OUTPUT_ROOT,
    NUM_CLASSES,
)
import cv2 as cv
import skimage.metrics

NUM_EPOCHS = 2
TRAIN_PARTITION = 0.7
TRAIN_LR = 5e-3
TRAIN_MOMENTUM = 9e-1
TRAIN_WEIGHT_DECAY = 5e-4
VAL_PARTITION = 0.2
BATCH_SIZE = 8
NUM_WORKERS = 4


if __name__ == "__main__":
    # get torch device
    device = get_device()

    # create directory for saving checkpoints if it doesn't exist
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # our dataset has two classes only - background and object
    imgs = list(sorted(os.listdir(DATA_IMAGES)))
    masks = list(sorted(os.listdir(DATA_TARGETS)))

#sequential structural similarity test. Barrier is generally around 0.9-0.85
    def ssim_test(greyaccumulator, pic) -> bool:
        for narray in greyaccumulator:
            if skimage.metrics.structural_similarity(narray, pic, data_range=255) > 0.9:
                return False
        return True

#culling mechanism, with a shifting window.
    final, finalm, accumulator, greyaccumulator, maccumulator = [], [], [], [], []
    index = 0
    for lindex in range(len(imgs)):
        llindex = len(imgs) - 1 - lindex
        img_path = os.path.join(DATA_IMAGES, imgs[llindex])
        pic = cv.imread(img_path)
        pic = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
        
        if not accumulator:
            accumulator.append(imgs[llindex])
            greyaccumulator.append(pic)
            maccumulator.append(masks[llindex])
        else:
            if ssim_test(greyaccumulator, pic): 
                if len(accumulator) == 30:
                    final.append(accumulator.pop(0))
                    finalm.append(maccumulator.pop(0))
                    greyaccumulator.pop(0)
                    accumulator.append(imgs[llindex])
                    maccumulator.append(masks[llindex])
                    greyaccumulator.append(pic)
                else:
                    accumulator.append(imgs[llindex])
                    maccumulator.append(masks[llindex])
                    greyaccumulator.append(pic)
    final.extend(accumulator)
    finalm.extend(maccumulator)
    imgs = final
    masks = finalm

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
            images = torch.stack([img.to(device, non_blocking=True) for img in images])
            masks = torch.stack(
                [mask.to(device, non_blocking=True) for mask in targets]
            )

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(OUTPUT_ROOT, f"deeplabv3_epoch_{epoch}.pth"),
        )

    torch.save(model.state_dict(), os.path.join(OUTPUT_ROOT, "deeplabv3_final.pth"))
