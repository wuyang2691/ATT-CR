# ATT-CR  
## Code Release Notice  

This repository contains the official implementation of the paper:  

> [**ATT-CR: Adaptive Triangular Transformer for Cloud Removal**](https://ieeexplore.ieee.org/document/11119343)  
> *Yang Wu, Ye Deng, Pengna Li, Wenli Huang, Kangyi Wu, Xiaomeng Xin, and Jinjun Wang*  

---

## 1. Repository Overview  

This codebase provides the **training**, **testing**, and **inference** implementations of the ATT-CR model for remote sensing image cloud removal.  

**Main components:**  
- **Model architecture:** `basicsr/models/archs/ATTCR_arch.py`  
- **Training & testing code:** `basicsr/`  
- **Experiment configuration files:** `option/`  
- **Pre-trained model checkpoints:** `experiments/` (e.g., RICE1, RICE2, T-Cloud, SEN12MS)  
- **Shell scripts for quick runs:** `train_scripts/`, `test_scripts/`  

---

## 2. Environment Setup  

We provide two dependency files: **`environment.yml`** (Conda) and **`requirements.txt`** (Pip).  

### 2.1 Prerequisites  
- **Hardware:** NVIDIA GPU with CUDA support (e.g., RTX 3090 / 4090)  
- **Software:** Linux/Ubuntu recommended, Conda, CUDA-compatible PyTorch  

### 2.2 Installation  

```bash
# Clone the repository
git clone https://github.com/wuyang2691/ATT-CR.git
cd ATT-CR-main

# Create and activate Conda environment
conda env create -f environment.yml
conda activate ATT_CR

# (Optional) Upgrade with pip dependencies
pip install -r requirements.txt --upgrade
```
### 2.3 Datasets download
- ```RICE-I```: It consists of 500 pairs of filmy and cloud-free images obtained from Google Earth. 
[RICE-I](https://github.com/BUPTLdy/RICE_DATASET)

- ```RICE-II```: It consists of 736 pairs of images captured by Landsat 8 OLI/TIRS, including cloudy, cloudless, and mask images. The mask images were created using the Landsat Level-1 quality band to identify regions affected by clouds, cloud shadows, and cirrus clouds. The cloudless images were captured at the same location as the corresponding cloud images with a maximum interval of 15 days. [RICE-II](https://github.com/BUPTLdy/RICE_DATASET)

- ```T-CLOUD```: The T-CLOUD, a real scene thin cloud dataset captured from Landsat 8 RGB images, contains 2,939 image pairs. The cloudy images and their clear counterparts are separated by one satellite re-entry period (16 days). These images are carefully selected with similar lighting conditions and are cropped into $256 \times 256$ patches. The dataset is split into 2351 images for training and 588 for testing. 
[T-CLOUD](https://github.com/haidong-Ding/Cloud-Removal)

- ```SEN12MS-CR```: It contains approximately 110,000 samples from 169 distinct, non-overlapping regions across various continents and meteorological seasons. Each sample includes a pair of Sentinel-2 images, one cloudy and one cloud-free, along with the corresponding Sentinel-1 synthetic aperture radar (SAR) image. [SEN12MS-CR](https://mediatum.ub.tum.de/1554803)

Download training and testing datasets and put them into the corresponding folders of ./dataset.

---

## 3. Training  

### 3.1 Configure Dataset Paths  
1. Prepare your dataset (RICE1, RICE2, T-Cloud, SEN12MS).  
2. Modify the corresponding YAML config file under `option/`.  
   - Example: for **T-Cloud**, edit `option/TCloud.yml`  
   - Update:
     ```yaml
     dataroot_gt: /path/to/ground_truth
     dataroot_lq: /path/to/cloudy_images
     ```

### 3.2 Run Training  

Option 1: **Shell Script**  
```bash
# Activate environment
source activate ATT_CR

# Run T-Cloud training
bash train_scripts/TCloud_train.sh
```

Option 2: **Direct Command**  
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun   --nproc_per_node=4   --master_port=3457   basicsr/train.py -opt option/TCloud.yml --launcher pytorch
```

---

## 4. Testing / Inference  

### 4.1 Pre-trained Models  
Pre-trained checkpoints are available in [`experiments/`]( https://pan.baidu.com/s/1ZGsZvL_aiT2tZxYO66uwpw?pwd=d1iq ).  
- Example: `experiments/RICE1/models/model_best.pth`  
- If using your own trained models, update the '--weights' in the .sh file.  

### 4.2 Run Testing  

Option 1: **Shell Script**  
```bash
bash test_scripts/TCloud.sh
```

Option 2: **Direct Command**  
```bash
CUDA_VISIBLE_DEVICES=0  python  basicsr/test.py --opt_yml option/TCloud.yml  --result_dir ./output/TCloud  --weights  ./experiments/T-Cloud/models/model_best.pth  --input_dir ./dataset/T-Cloud/test/input  --input_truth_dir ./dataset/T-Cloud/test/target
```

Special case for **SEN12MS dataset**:  
```bash
python basicsr/senms_predict.py
```

---

## 5. Citation & Acknowledgment  

If you find this work useful, please consider citing our paper and giving the repository a ⭐ to stay updated.  

```bash
@article{wu2025attcr,
  title={ATT-CR: Adaptive Triangular Transformer for Cloud Removal},
  author={Wu, Yang and Deng, Ye and Li, Pengna and Huang, Wenli and Wu, Kangyi and Xin, Xiaomeng and Wang, Jinjun},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={ATT-CR: Adaptive Triangular Transformer for Cloud Removal}, 
  year={2025},
  volume={18},
  number={},
  pages={20595-20610},
}
```