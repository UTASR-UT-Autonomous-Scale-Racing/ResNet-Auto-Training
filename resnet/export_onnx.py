"""Convert a Deeplabv3 .pth checkpoint into an ONNX model.

Run from this directory (ResNet-Auto-Training/resnet/):

    python export_onnx.py <pth_path> <onnx_path>

Example:
    python export_onnx.py checkpoints/deeplabv3_final.pth deeplabv3_final.onnx

The exported ONNX has dynamic batch / height / width axes, so the same
file works for any input resolution the wrapper resizes to.
"""

import argparse
import torch

from helpers import NUM_CLASSES, get_segmentation_model


def export(pth_path: str, onnx_path: str, input_shape=(3, 360, 640)) -> None:
    model = get_segmentation_model(NUM_CLASSES)
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        (dummy,),
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
    print(f"Exported {pth_path} -> {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pth_path", help="Path to the .pth checkpoint")
    parser.add_argument("onnx_path", help="Where to write the .onnx file")
    args = parser.parse_args()
    export(args.pth_path, args.onnx_path)
