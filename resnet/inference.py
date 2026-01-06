import cv2
import numpy as np
import onnxruntime as ort
import os
import torch
import torch.onnx
from helpers import (
    DATA_IMAGES,
    NUM_CLASSES,
    OUTPUT_ROOT,
    get_device,
    get_segmentation_model,
)

ONNX_NAME = "deeplabv3_final_onnx"
PROVIDERS = [
    # "TensorrtExecutionProvider",  # use only when on jetson
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


def export_to_onnx(
    model,
    input_shape=(3, 360, 640),
    onnx_path=os.path.join(OUTPUT_ROOT, f"{ONNX_NAME}.onnx"),
) -> None:
    """Export PyTorch model to ONNX format"""
    model.eval()
    model.to("cpu")

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to("cpu")  # force cpu

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=12,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"Model exported to {onnx_path}")


def inference_real_time_test_onnx(onnx_path, providers, imgs, device) -> None:
    """Inference using ONNX model"""

    print("ONNX Inference")

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path, providers=providers)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    for img_path in imgs:
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            continue

        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = ((img_rgb.to(device) - mean) / std).unsqueeze(0)
        x = img_tensor.contiguous().cpu().numpy()

        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: x}
        ort_outputs = ort_session.run(None, ort_inputs)

        pred = np.argmax(ort_outputs[0][0], axis=0)

        # visualization
        vis = (img_tensor[0] * std + mean).clamp(0, 1).cpu().numpy()
        vis = (vis * 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]

        overlay = vis.copy()
        overlay[pred == 1] = [0, 255, 0]  # greeeeeeeeeeeen

        blended = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)

        cv2.imshow("Semantic Segmentation", blended)
        if cv2.waitKey(1) & 0xFF == 27:  # esc
            break

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
    inference_real_time_test_onnx(onnx_path, PROVIDERS, test_imgs, device=get_device())
