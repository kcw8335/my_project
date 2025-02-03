# Fairness-Aware Multimodal Learning in Automatic Video Interview Assessment

This repository contains the implementation of the paper:

**"Fairness-Aware Multimodal Learning in Automatic Video Interview Assessment"**

> 📄 [Read the full paper here](https://ieeexplore.ieee.org/abstract/document/10287972)

## 🔍 Overview
This project addresses fairness issues in automated video interview assessment systems that utilize multimodal data (video, audio, and text). Our method aims to reduce bias by minimizing the Wasserstein distance between sensitive groups while balancing fairness and preformance. The model is trained using an adversarial learning approach and a fairness-aware loss function.

## 📌 Features
- **Multimodal Learning**: Uses video, audio, and text features.
- **Fairness-Aware Training**: Implements adversarial learning to mitigate bias.
- **Wasserstein Distance Optimization**: Ensures fairness by minimizing distribution differences.
- **Customizable Fairness-Accuracy Tradeoff**: Adjustable hyperparameters to balance fairness and model performance.

## 📂 Repository Structure
```
MY_PROJECT/
│── src/
│   ├── callback/         # PyTorch Lightning callbacks
│   ├── datamodule/       # PyTorch Lightning data module
│   ├── dataset/          # Custom dataset handling
│   ├── loss/             # Loss functions
│   ├── metric/           # Evaluation PyTorch metrics
│   ├── module/           # PyTorch Lightning components
│   ├── network/          # Model architectures
│   ├── task/             # Training and evaluation scripts
│── README.md             # Project documentation
│── requirements.txt      # Dependencies
```

## 🚀 Installation
To install the required dependencies, run:
```bash
# Clone this repository
git clone https://github.com/kcw8335/my_project.git
cd src

# Create and activate conda environment
conda create --name fairness_env python=3.10.15 -y
conda activate fairness_env

# Install required dependencies
pip install -r requirements.txt
```

## 🔧 Usage
### 🎯 Data Preparation
For details on dataset preparation, please refer to the paper: "Fairness-Aware Multimodal Learning in Automatic Video Interview Assessment".

- Video files for each interview session.
- Corresponding audio extracted from video.
- Transcribed text from the spoken content.
- Labels: Interview assessment score and sensitive attributes (e.g., gender, age).

This paper used two datasets:
- **Hiring Recommendation (HR) Dataset** (real-world job interview dataset)
- **First Impressions (FI) Dataset** (benchmark dataset with video-based assessments)

### 📊 Training the Model
To train the model, run:
```bash
python ./task/chalearn/train.py
```
For configuration details, please refer to `config.yaml`.

## 📖 Citation
If you use this code, please cite our paper:
```bibtex
@article{kim2023fairness,
  title={Fairness-aware multimodal learning in automatic video interview assessment},
  author={Kim, Changwoo and Choi, Jinho and Yoon, Jongyeon and Yoo, Daehun and Lee, Woojin},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```

## 📬 Contact
For any inquiries, please contact [Changwoo KIM](note8335@dgu.ac.kr).
