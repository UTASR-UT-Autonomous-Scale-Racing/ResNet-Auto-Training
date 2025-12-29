"""
Improvised from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import math
import sys

import torch
import utils
from sklearn.metrics import jaccard_score


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, print_freq, scaler=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    criterion = torch.nn.CrossEntropyLoss()

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [image.to(device) for image in images]

        masks = [t["mask"].to(device) for t in targets]
        masks = torch.stack(masks)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device):
    model.eval()
    iou_scores = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        masks = [mask.to(device) for mask in targets]

        outputs = model(images)["out"]
        preds = outputs.argmax(dim=1)

        for pred, true in zip(preds, masks):
            pred_flat = pred.flatten().cpu().numpy()
            true_flat = true.flatten().cpu().numpy()
            iou_scores.append(jaccard_score(true_flat, pred_flat, average="macro"))

    mean_iou = sum(iou_scores) / len(iou_scores)
    return mean_iou
