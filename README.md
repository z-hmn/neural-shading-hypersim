# Neural Shading with Hypersim

### Zoe Homan  
Barnard College — Computer Science  
Deep Learning for Computer Graphics Final Project
Prof. Corey Toler-Franklin

This project explores neural shading as a supervised image-to-image learning problem using Apple’s Hypersim dataset.

A U-Net convolutional neural network is trained to predict diffuse illumination from:

- surface normals
- depth maps
- diffuse reflectance

The predicted illumination is then recombined with reflectance to reconstruct diffuse rendered appearance.

## Features

- U-Net illumination prediction
- Multi-buffer input pipeline
- Reconstruction-aware training
- Gradient loss experiments
- Camera-space vs world-space normal experiments
- AWS GPU training workflow

## Repository Structure

```text
.
├── scripts/
├── samples_ai_001_001.json
├── samples_multiscene_normalCam.json
├── samples_multiscene_normalWorld.json
├── requirements.txt
└── README.md
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses Apple’s Hypersim dataset:

https://github.com/apple/ml-hypersim

The dataset is not included in this repository because of its size.

## Main Training Scripts

Baseline training:

```bash
python scripts/train_eval.py
```

Camera-space normals:

```bash
python scripts/train_eval_normalCam.py
```

World-space normals + reconstruction loss:

```bash
python scripts/train_eval_normalWorldRecon.py
```

World-space normals + reconstruction + gradient loss:

```bash
python scripts/train_normWorld_Recon_Grad.py
```

## Model

The model is implemented in:

```text
scripts/model.py
```

Input tensor:

```text
7 x 256 x 256
```

- normals (3)
- depth (1)
- reflectance (3)

Output tensor:

```text
3 x 256 x 256
```

representing RGB diffuse illumination.

## Notes

This repository contains the project code and sample index files, but does not include:

- Hypersim scene data
- cached tensors
- trained checkpoints

The project was developed locally for early experiments and later trained on AWS using NVIDIA Tesla T4 GPUs.
