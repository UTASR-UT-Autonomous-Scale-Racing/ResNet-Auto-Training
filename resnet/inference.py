"""
Reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import os

import cv2
import onnxruntime as ort
import torch
import torch.onnx
from helpers import (
    DATA_IMAGES,
    NUM_CLASSES,
    OUTPUT_ROOT,
    get_device,
    get_segmentation_model,
    get_transform,
)

ONNX_NAME = "deeplabv3_final_onnx"


def export_to_onnx(
    model,
    input_shape=(3, 400, 640),
    onnx_path=os.path.join(OUTPUT_ROOT, f"{ONNX_NAME}.onnx"),
):
    """Export PyTorch model to ONNX format"""
    model.eval()
    model.to("cpu")

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to("cpu")  # force cpu

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=12,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"Model exported to {onnx_path}")


def inference_real_time_test_onnx(onnx_path, imgs):
    """Inference using ONNX model"""
    print("ONNX Inference")
    transform = get_transform(train=False)

    # Load ONNX model
    ort_session = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    for img_path in imgs:
        image_raw = cv2.imread(img_path)
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = transform(image)  # Apply the same transformations as during training
        x = (
            image.unsqueeze(0).numpy().astype("float32")
        )  # ONNX Runtime requires CPU numpy array

        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: x}
        ort_outputs = ort_session.run(None, ort_inputs)

        logits = torch.tensor(ort_outputs[0])
        pred = logits.argmax(dim=1).squeeze(0).numpy()

        overlay = image.numpy().transpose(1, 2, 0)[:, :, ::-1].copy()
        overlay[pred == 1] = [0, 255, 0]
        blended = cv2.addWeighted(
            overlay.astype("uint8"),
            0.6,
            image.numpy().transpose(1, 2, 0)[:, :, ::-1].astype("uint8"),
            0.4,
            0,
        )
        cv2.imshow("Semantic Segmentation", blended)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    device = get_device()

    imgs = [
        os.path.join(DATA_IMAGES, img) for img in list(sorted(os.listdir(DATA_IMAGES)))
    ]
    indices = list(range(len(imgs)))
    test_imgs = [imgs[i] for i in indices[int(len(indices) * 0.9) :]]

    # Export to ONNX (run once)
    onnx_path = os.path.join(OUTPUT_ROOT, f"{ONNX_NAME}.onnx")
    if not os.path.exists(onnx_path):
        model = get_segmentation_model(NUM_CLASSES)
        model.load_state_dict(
            torch.load(os.path.join(OUTPUT_ROOT, "deeplabv3_final.pth"))
        )
        model.to(device)
        export_to_onnx(model, onnx_path=onnx_path)

    # Use ONNX inference
    inference_real_time_test_onnx(onnx_path, test_imgs)
