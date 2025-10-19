## Getting Started

### Data sources

1. [CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)
2. [Diverse Dermatology Images](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965)

### Installation

1. Install dependencies
```
# Set up and activate a virtual environment
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate

# Install necessary packages
pip3 install --upgrade pip
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install "transformers @ git+https://github.com/huggingface/transformers@01734dba842c29408c96caa5c345c9e415c7569b"
pip3 install open-flamingo==2.0.1
pip3 install open-clip-torch==2.23.0
pip3 install pandas
pip3 install einops
pip3 install "accelerate @ git+https://github.com/huggingface/accelerate@4d583ad6a1f13d1d7617e6a37f791ec01a68413a"
pip3 install numpy==1.24.2
pip3 install scikit-learn==1.2.2
pip3 install protobuf==3.20.3
pip3 install timm==0.6.12
```

2. Then the 7B Llama base model needs to be installed separately. Download from this repository -- https://huggingface.co/baffo32/decapoda-research-llama-7B-hf, and then point the llama_path variable in the flamingo.py model definition for the MedFlamingo model to llama_path = '/path/to/decapoda-research-llama-7B-hf'

## Source code

The core VCR code is in the src/interpretability/vcr.py script. It makes use of helper files in src/models (such as the core flamingo model code), and in the src/interpretability/clip.py script (for the CLIP embeddings)

We include two experiment scripts (one for CheXpert, one for DDI) to demonstrate how we used VCR to generate explanations. These are found in /src/experiments/