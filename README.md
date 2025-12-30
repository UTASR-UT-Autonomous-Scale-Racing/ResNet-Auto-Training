# ResNet Auto Training

This project simplifies the process of data labeling by automating the creation of color masks using SAM2 prompts. Users can manually correct errors in the generated masks, and SAM2 further automates the masking process to produce ground truth values for ResNet training.

Note: uses the MobileNetV3-Large backbone.

# Installation

Just run `install.sh`.

1. **Install SAM2**
   `git clone https://github.com/facebookresearch/sam2.git`
   `cd sam2`
   `pip install -e .`

1. **Download checkpoints**
   `mkdir -p checkpoints/`
   `cd checkpoints/`
   `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`

# Instructions

1. **Generate frames and prompts**
   `python masking/generate_frames_and_prompts.py`

1. **Generates masks using SAM2**
   `python masking/generate_sam2_masks.py`

1. **Training**
   `python resnet/train.py`

1. **Inference**
   `python resnet/inference.py`

**Summary of Outputs:**

- `data/frames_and_prompts`: Contains extracted JPG images from the video and prompts for each frame 
- `data/sam2_masked_frames`: Contains masked images with pixel values of 0 and 1
- `checkpoints/deeplabv3_epoch_N.pth`: The trained model at epoch N
- `checkpoints/deeplabv3_final.pth`: The final trained model
- `checkpoints/deeplabv3_final_onnx.onnx`: The final trained model in ONNX format
