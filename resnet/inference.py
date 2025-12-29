"""
Improvised from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import cv2
import onnxruntime as ort
import torch
import torch.onnx
from helpers import get_segmentation_model, get_transform
from torchvision.transforms import v2 as T

onnx_name = "semantic_segmentation_deeplabv3"


def export_to_onnx(
    model,
    device,
    input_shape=(3, 400, 640),
    onnx_path=f"checkpoints/{onnx_name}.onnx",
):
    """Export PyTorch model to ONNX format"""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)

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
    )
    print(f"Model exported to {onnx_path}")


def inference_real_time_test_onnx(onnx_path, device, imgs):
    """Inference using ONNX model"""
    print("ONNX Inference")
    transform = get_transform(train=False)

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)

    for img_path in imgs:
        image_raw = cv2.imread(img_path)
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = T.ToTensor()(image)
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
    import os

    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    # device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_classes = 2
    root = os.path.join("data", "dataset")
    imgs = [
        os.path.join(root, "images", img)
        for img in list(sorted(os.listdir(os.path.join(root, "images"))))
    ]
    indices = list(range(len(imgs)))
    test_imgs = [imgs[i] for i in indices[int(len(indices) * 0.9) :]]

    # Export to ONNX (run once)
    onnx_path = os.path.join("checkpoints", f"{onnx_name}.onnx")
    if not os.path.exists(onnx_path):
        model = get_segmentation_model(num_classes)
        checkpoints_path = os.path.join("checkpoints", f"{onnx_name}.pth")
        model.load_state_dict(torch.load(checkpoints_path))
        model.to(device)
        export_to_onnx(model, device, onnx_path=onnx_path)

    # Use ONNX inference
    inference_real_time_test_onnx(onnx_path, device, test_imgs)
