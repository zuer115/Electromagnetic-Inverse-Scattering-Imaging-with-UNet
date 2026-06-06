# 基于UNet的电磁逆散射成像

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![GitHub stars](https://img.shields.io/github/stars/zuer115/Electromagnetic-Inverse-Scattering-Imaging-with-UNet.svg)](https://github.com/zuer115/Electromagnetic-Inverse-Scattering-Imaging-with-UNet/stargazers)

[English](./README.md) | 中文

本项目提供了一种融合物理机制与深度神经网络的高分辨电磁成像开源框架。通过将**扭曲玻恩近似（Distorted Born Approximation, DBA）**物理先验与定制化的**无伪影双线性 U-Net**深度耦合，我们有效解决了微波成像中高度非线性和极度病态的反问题（ISP）。

## 核心亮点
- **匹配液物理模型**：引入可调背景介电常数（默认 $\epsilon_\mathrm{r} = 1.5$），从物理底层彻底消除虚拟边界反射与电磁杂波干扰。
- **物理先验提取**：利用 DBA 伴随投影算子与空间灵敏度校正（Sensitivity Correction），将散射场直接映射为高精度空间拓扑特征。
- **无网格伪影架构**：摒弃传统转置卷积，采用双线性/三线性插值上采样，从数学底层 100% 根除 2D 及 3D 空间中的网格伪影（Checkerboard Artifacts）。
- **极致 HPC 加速**：基于 PyTorch 的纯 GPU 物理引擎，支持 `complex64` 矩阵并发求解与 `bfloat16` 混合精度训练，极大缩短数据生成与训练周期。

## 仓库结构
```text
.
├── main.py            # 通用高分辨重构推理引擎 (支持命令行极简调用)
├── gen_data.py        # 2D 物理数据集生成脚本 (全孔径/受限视角)
├── gen_data_3d.py     # 3D 物理数据集生成脚本 (斐波那契球面天线阵列)
├── train.py           # 2D 网络模型训练脚本
├── train_3d.py        # 3D 网络模型训练脚本
├── evaluate.py        # 2D 模型评估与可视化脚本
├── evaluate_3d.py     # 3D 模型评估、断层切片与真 3D 体素透明渲染脚本
├── train_data/        # 预生成的 2D 及 3D 训练与测试数据集 (.npy)
├── models/            # 训练完毕的最优模型权重 (2D全孔径, 2D半孔径, 3D全孔径)
└── output/            # 存放评估代码自动生成的高清学术对比图谱
```

## 环境依赖
- Python 3.8+
- PyTorch >= 2.0.0 (强烈建议配置 CUDA)
- NumPy, SciPy, Matplotlib

## 使用指南

### 1. 通用推理引擎 (`main.py`)
我们提供了一个开箱即用的命令行工具，可直接将预训练模型应用于新数据集的批量推理。

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i, --input` | *(必填)* | 输入的散射场 `.npy` 文件路径 |
| `-w, --weights` | `models/full_best_model.pth` | 训练好的模型权重 `.pth` 文件路径 |
| `-o, --out_dir` | `output` | 结果保存目录 |
| `-d, --dim` | `2d` | 数据维度：`2d` 或 `3d` |
| `-m, --mode` | `full_circle` | 天线阵列排布：`full_circle` 或 `half_circle` |
| `-f, --formats` | `npy png` | 输出格式列表，例如 `npy png pdf svg` |
| `-b, --batch_size` | `16` | 推理批大小 |
| `--cpu` | *(标志)* | 仅使用CPU运行 |

**调用示例 (2D 全孔径):**
```bash
python main.py -i train_data/full_test_X.npy -w models/full_best_model.pth -d 2d -m full_circle -f npy pdf png
```

### 2. 数据集生成
利用 GPU 加速的矩量法（MoM）生成包含多重散射的高保真电磁物理数据集。

#### 2a. 2D 数据生成 (`gen_data.py`)

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-a, --array_type` | *(必填)* | 天线阵列：`full_circle` 或 `half_circle` |
| `--eps_bg` | `1.5` | 背景相对介电常数 |
| `-o, --out_dir` | `train_data` | 数据集输出目录 |
| `-b, --batch_size` | `16` | 正向求解批大小 |
| `--num_train_clean` | `10000` | 无噪声训练样本数 |
| `--num_train_noisy` | `3000` | 含噪声训练样本数 |
| `--num_test` | `2000` | 测试样本数 |
| `--train_clean_prefix` | `{type}_train_clean` | 无噪声训练集文件名前缀 |
| `--train_noisy_prefix` | `{type}_train_noisy` | 含噪声训练集文件名前缀 |
| `--test_prefix` | `{type}_test` | 测试集文件名前缀 |
| `--cpu` | *(标志)* | 仅使用CPU运行 |

#### 2b. 3D 数据生成 (`gen_data_3d.py`)

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--eps_bg` | `1.5` | 背景相对介电常数 |
| `-o, --out_dir` | `train_data` | 数据集输出目录 |
| `-b, --batch_size` | `1` | 正向求解批大小 |
| `--num_train` | `2000` | 训练样本数 |
| `--num_test` | `500` | 测试样本数 |
| `--train_prefix` | `sphere_train` | 训练集输出前缀 |
| `--test_prefix` | `sphere_test` | 测试集输出前缀 |
| `--cpu` | *(标志)*| 仅使用CPU运行 |

**调用示例:**
```bash
# 2D 全孔径（默认参数）
python gen_data.py -a full_circle

# 2D 受限视角，自定义背景介电常数和样本数
python gen_data.py -a half_circle --eps_bg 1.5 -b 8 --num_train_clean 5000 --num_test 1000

# 3D 全空间
python gen_data_3d.py
```

### 3. 模型训练
在生成数据后，可直接启动网络的从零训练。

#### 3a. 2D 模型训练 (`train.py`)

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-a, --array_type` | *(必填)* | 天线阵列：`full_circle` 或 `half_circle` |
| `--eps_bg` | `1.5` | 背景相对介电常数 |
| `-o, --out_dir` | `models` | 模型权重输出目录 |
| `-b, --batch_size` | `128` | 训练批大小 |
| `-e, --epochs` | `400` | 训练轮数 |
| `--train_clean_path` | `train_data/{type}_train_clean` | 无噪声训练数据路径前缀 |
| `--train_noisy_path` | `train_data/{type}_train_noisy` | 含噪声训练数据路径前缀 |
| `--test_path` | `train_data/{type}_test` | 测试数据路径前缀 |
| `--no_noisy` | *(标志)* | 跳过含噪声训练数据 |
| `--cpu` | *(标志)* | 仅使用CPU运行 |

#### 3b. 3D 模型训练 (`train_3d.py`)

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--eps_bg` | `1.5` | 背景相对介电常数 |
| `-o, --out_dir` | `models` | 模型权重输出目录 |
| `-b, --batch_size` | `64` | 训练批大小 |
| `-e, --epochs` | `250` | 训练轮数 |
| `--train_path` | `train_data/sphere_train` | 训练数据路径前缀 |
| `--test_path` | `train_data/sphere_test` | 测试数据路径前缀 |
| `--cpu` | *(标志)* | 仅使用CPU运行 |

**调用示例:**
```bash
# 2D 全孔径训练
python train.py -a full_circle

# 2D 训练，自定义参数
python train.py -a half_circle --eps_bg 1.5 -b 64 -e 200 --no_noisy

# 3D 训练
python train_3d.py
```

### 4. 评估与学术作图
自动读取测试集与模型权重，生成达到 SCI 期刊出版标准的重构对比图。

#### 4a. 2D 评估 (`evaluate.py`)

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--eps_bg` | `1.5` | 背景相对介电常数 |
| `-o, --out_dir` | `output` | 图像输出目录 |
| `--test_full_path` | `train_data/full_test` | 全孔径测试数据路径前缀 |
| `--test_half_path` | `train_data/half_test` | 半孔径测试数据路径前缀 |
| `--full_model` | `models/full_best_model.pth` | 全孔径模型权重路径 |
| `--half_model` | `models/half_best_model.pth` | 半孔径模型权重路径 |
| `-n, --num_images` | `3` | 每种阵列显示的样本图像数 |
| `--no_full` | *(标志)* | 跳过全孔径评估 |
| `--no_half` | *(标志)* | 跳过半孔径评估 |
| `--cpu` | *(标志)* | 仅使用CPU运行 |

#### 4b. 3D 评估 (`evaluate_3d.py`)

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--eps_bg` | `1.5` | 背景相对介电常数 |
| `-o, --out_dir` | `output` | 图像输出目录 |
| `--test_path` | `train_data/sphere_test` | 测试数据路径前缀 |
| `-w, --model_path` | `models/sphere_best_model.pth` | 模型权重路径 |
| `-n, --num_images` | `3` | 样本图像数 |
| `--no_slices` | *(标志)* | 跳过 2D 断层切片图 |
| `--no_voxels` | *(标志)* | 跳过 3D 体素渲染图 |
| `--cpu` | *(标志)* | 仅使用CPU运行 |

**调用示例:**
```bash
# 完整 2D 评估
python evaluate.py

# 仅评估半孔径，指定模型路径
python evaluate.py --no_full --half_model models/half_best_model.pth -n 5

# 3D 评估 — 仅生成切片图
python evaluate_3d.py --no_voxels -n 2
```

## 结果展示
本框架在完美解决纯数据驱动方法引起的"均值坍塌"与"伪影"问题的同时，实现了对目标绝对介电常数的极高精度定量回归。即便在极具挑战的单侧缺失视角（Half-Circle）下，网络依然表现出卓越的非线性抗逆泛化能力。详细高清图表请运行评估脚本或查看 `output/` 文件夹。