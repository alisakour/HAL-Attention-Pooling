# Enhancing HAL Representations via Attention-Based Pooling

![HAL-ATTN Demo](header.gif)

## Overview
This repository contains the official PyTorch implementation of our paper, which introduces a hybrid architecture that bridges classical distributional semantics (HAL, 1996) with modern deep learning. By replacing mean pooling with a **temperature-scaled additive attention mechanism**, the model dynamically prioritizes context-salient words over structural noise.

## Key Results
Our evaluation on the IMDB dataset demonstrates a significant improvement in both convergence speed and final accuracy:

| Model | Pooling Strategy | Initial Acc (Epoch 1) | Peak Test Accuracy |
|---|---|---|---|
| Baseline | Mean Pooling | 63.83% | 75.64% |
| **Proposed** | **Attention Pooling** | **78.70%** | **82.38%** |

**Improvement:** The proposed model yields an absolute accuracy gain of **+6.74 percentage points** and surpasses the baseline's peak accuracy in its very first epoch.

## Installation & Usage
To replicate our results, clone the repository and run the training pipeline:
```bash
# 1. Clone the repository
git clone https://github.com/alisakour/HAL-Attention-Pooling.git
cd HAL-Attention-Pooling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the end-to-end training and evaluation script
python train.py
```

## Authors
- **Ali Sakour**
- **Zoalfekar Sakour**

*Undergraduate Students, Dept. of Computer and Automatic Control Engineering, Tishreen University, Syria.*