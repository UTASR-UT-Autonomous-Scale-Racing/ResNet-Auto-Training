# ResNet Auto Training

This project simplifies the process of data labeling by automating the creation of color masks using SAM2 prompts. Users can manually correct errors in the generated masks, and SAM2 further automates the masking process to produce ground truth values for ResNet training.

Note: uses the MobileNetV3-Large backbone.

# Instructions

1. `cd masking`
1. **Install SAM2**
   `git clone https://github.com/facebookresearch/sam2.git`
   `cd sam2`
   `pip install -e .`
1. **Download checkpoints**
   `mkdir -p checkpoints/`
   `cd checkpoints/`
   `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`
1. **Create JPG folder**
   `python masking/generate_frames_and_prompts.py`
1. **Run video masking**
   `python masking/generate_sam2_masks.py`
1. **Train ResNet**
   `cd resnet/`
   `python resnet/train.py`
1. **Run Inference**
   `python resnet/inference.py`

**Summary of Outputs:**

- `data/frames_and_prompts`: Contains extracted JPG images from the video and prompts for each frame.
- `data/sam2_masked_frames`: Contains masked images with pixel values of 0 and 1.
- `checkpoints/semantic_segmentation_deeplabv3.onnx`: Contains the trained Deeplabv3 model.
