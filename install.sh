git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

cd
mkdir -p checkpoints/
cd checkpoints/
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
