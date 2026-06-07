# Electromagnetic Inverse Scattering Imaging with UNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![GitHub stars](https://img.shields.io/github/stars/zuer115/Electromagnetic-Inverse-Scattering-Imaging-with-UNet.svg)](https://github.com/zuer115/Electromagnetic-Inverse-Scattering-Imaging-with-UNet/stargazers)

English | [中文](./README_CN.md)

This repository contains the official implementation of our research on high-resolution 2D and 3D microwave imaging. By deeply coupling the **Distorted Born Approximation (DBA)** physical priors with a customized **Bilinear U-Net**, we effectively solve the highly non-linear and ill-posed Electromagnetic Inverse Scattering Problem (ISP).

## Key Features
- **Matching Liquid Model**: Introduces an adjustable background permittivity (default $\epsilon_\mathrm{r} = 1.5$) to physically eliminate boundary reflections and scattering clutter.
- **Physical Prior Extraction**: Utilizes Distorted Born Approximation (DBA) and Illumination/Sensitivity Correction to map scattered fields into highly accurate initial spatial features.
- **Artifact-Free Architecture**: Replaces traditional transposed convolutions with Bilinear/Trilinear upsampling, completely eradicating checkerboard artifacts in both 2D and 3D space.
- **High-Performance Computing (HPC)**: Fully optimized PyTorch backend (`complex64`, `bfloat16` AMP, batched GPU solvers) capable of generating and training massive datasets in minutes.

## Repository Structure
```text
.
├── main.py            # Universal CLI inference engine (2D/3D)
├── gen_data.py        # 2D dataset generator (Method of Moments)
├── gen_data_3d.py     # 3D dataset generator (Fibonacci Spherical Lattice)
├── train.py           # 2D network training script
├── train_3d.py        # 3D network training script
├── evaluate.py        # 2D evaluation and visualization script
├── evaluate_3d.py     # 3D tomographic slicing & voxel rendering script
├── train_data/        # Pre-generated 2D and 3D training/testing datasets (.npy)
├── models/            # Pre-trained checkpoints (2D Full, 2D Half, 3D Full)
├── output/            # Directory containing evaluation outputs and images
└── tests/             # Test suite
```

## Requirements
- Python 3.8+
- PyTorch >= 2.0.0 (CUDA enabled recommended)
- NumPy, SciPy, Matplotlib

## Usage Guide

### 1. Universal Inference CLI (`main.py`)
We provide a ready-to-use command-line interface for deploying the trained models on new data.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | *(required)* | Input scattered field `.npy` file |
| `-w, --weights` | `models/full_best_model.pth` | Trained model `.pth` file |
| `-o, --out_dir` | `output` | Output directory |
| `-d, --dim` | `2d` | Data dimension: `2d` or `3d` |
| `-m, --mode` | `full_circle` | Antenna array configuration: `full_circle` or `half_circle` |
| `-f, --formats` | `npy png` | Output formats, e.g., `npy png pdf svg` |
| `-b, --batch_size` | `16` | Inference batch size |
| `--cpu` | *(flag)* | Use CPU *only* |

**Example (2D Full-Circle):**
```bash
python main.py -i train_data/full_test_X.npy -w models/full_best_model.pth -d 2d -m full_circle -f npy pdf png
```

### 2. Dataset Generation
Generate high-fidelity electromagnetic scattered fields using GPU-accelerated Method of Moments (MoM).

#### 2a. 2D Data Generator (`gen_data.py`)

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `-a, --array_type` | *(required)* | Antenna array: `full_circle` or `half_circle` |
| `--eps_bg` | `1.5` | Background relative permittivity |
| `-o, --out_dir` | `train_data` | Output directory for datasets |
| `-b, --batch_size` | `16` | Forward solve batch size |
| `--num_train_clean` | `10000` | Number of clean training samples |
| `--num_train_noisy` | `3000` | Number of noisy training samples |
| `--num_test` | `2000` | Number of test samples |
| `--train_clean_prefix` | `{type}_train_clean` | Filename prefix for clean training set |
| `--train_noisy_prefix` | `{type}_train_noisy` | Filename prefix for noisy training set |
| `--test_prefix` | `{type}_test` | Filename prefix for test set |
| `--cpu` | *(flag)* | Use CPU *only* |

#### 2b. 3D Data Generator (`gen_data_3d.py`)

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--eps_bg` | `1.5` | Background relative permittivity |
| `-o, --out_dir` | `train_data` | Output directory for datasets |
| `-b, --batch_size` | `1` | Forward solve batch size |
| `--num_train` | `2000` | Number of training samples |
| `--num_test` | `500` | Number of test samples |
| `--train_prefix` | `sphere_train` | Output prefix for training set |
| `--test_prefix` | `sphere_test` | Output prefix for test set |
| `--cpu` | *(flag)* | Use CPU *only* |

**Examples:**
```bash
# 2D full-aperture (default sizes)
python gen_data.py -a full_circle

# 2D limited-view, custom background and sample counts
python gen_data.py -a half_circle --eps_bg 1.5 -b 8 --num_train_clean 5000 --num_test 1000

# 3D full space
python gen_data_3d.py
```

### 3. Model Training
Train the Bilinear U-Net models from scratch.

#### 3a. 2D Training (`train.py`)

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `-a, --array_type` | *(required)* | Antenna array: `full_circle` or `half_circle` |
| `--eps_bg` | `1.5` | Background relative permittivity |
| `-o, --out_dir` | `models` | Output directory for model weights |
| `-b, --batch_size` | `128` | Training batch size |
| `-e, --epochs` | `400` | Number of training epochs |
| `--train_clean_path` | `train_data/{type}_train_clean` | Path stem for clean training data |
| `--train_noisy_path` | `train_data/{type}_train_noisy` | Path stem for noisy training data |
| `--test_path` | `train_data/{type}_test` | Path stem for test data |
| `--no_noisy` | *(flag)* | Skip loading noisy training data |
| `--cpu` | *(flag)* | Use CPU *only* |

#### 3b. 3D Training (`train_3d.py`)

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--eps_bg` | `1.5` | Background relative permittivity |
| `-o, --out_dir` | `models` | Output directory for model weights |
| `-b, --batch_size` | `64` | Training batch size |
| `-e, --epochs` | `250` | Number of training epochs |
| `--train_path` | `train_data/sphere_train` | Path stem for training data |
| `--test_path` | `train_data/sphere_test` | Path stem for test data |
| `--cpu` | *(flag)* | Use CPU *only* |

**Examples:**
```bash
# 2D full-circle training
python train.py -a full_circle

# 2D training with custom samples and paths
python train.py -a half_circle --eps_bg 1.5 -b 64 -e 200 --no_noisy

# 3D training
python train_3d.py
```

### 4. Evaluation and Visualization
Generate academic-grade comparison plots (Ground Truth vs. DBA BP vs. U-Net vs. Error).

#### 4a. 2D Evaluation (`evaluate.py`)

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--eps_bg` | `1.5` | Background relative permittivity |
| `-o, --out_dir` | `output` | Output directory for plots |
| `--test_full_path` | `train_data/full_test` | Path stem for full-circle test data |
| `--test_half_path` | `train_data/half_test` | Path stem for half-circle test data |
| `--full_model` | `models/full_best_model.pth` | Path to full-circle model weights |
| `--half_model` | `models/half_best_model.pth` | Path to half-circle model weights |
| `-n, --num_images` | `3` | Number of sample images per array type |
| `--no_full` | *(flag)* | Skip full-circle evaluation |
| `--no_half` | *(flag)* | Skip half-circle evaluation |
| `--cpu` | *(flag)* | Use CPU *only* |

#### 4b. 3D Evaluation (`evaluate_3d.py`)

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--eps_bg` | `1.5` | Background relative permittivity |
| `-o, --out_dir` | `output` | Output directory for plots |
| `--test_path` | `train_data/sphere_test` | Path stem for test data |
| `-w, --model_path` | `models/sphere_best_model.pth` | Path to model weights |
| `-n, --num_images` | `3` | Number of sample images |
| `--no_slices` | *(flag)* | Skip 2D slice plots |
| `--no_voxels` | *(flag)* | Skip 3D voxel plots |
| `--cpu` | *(flag)* | Use CPU *only* |

**Examples:**
```bash
# Full 2D evaluation
python evaluate.py

# Half-circle only, with custom model path
python evaluate.py --no_full --half_model models/half_best_model.pth -n 5

# 3D evaluation — slices only
python evaluate_3d.py --no_voxels -n 2
```

## Results Showcase
Our model successfully achieves highly accurate quantitative dielectric constant regression and boundary preservation, even under the extremely challenging Limited-View (Half-Circle) configuration. See the `output/` directory for detailed visualizations.