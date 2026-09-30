# SCM VI-ReID 独立实验版本

六个目录分别包含完整代码，每个目录可单独复制到独立服务器或 GPU 作业运行，不依赖其他版本目录。入口均为 `run_variant.sh`。

| 目录 | 内容 |
| --- | --- |
| `SCM_for_VI-ReID-Code-improved` | 改进基线 |
| `SCM_for_VI-ReID-v1-hc` | 基线 + HC 中心三元组损失 |
| `SCM_for_VI-ReID-v2-circle` | 基线 + Circle Pair Loss |
| `SCM_for_VI-ReID-v3-mixstyle` | 基线 + 跨模态 MixStyle |
| `SCM_for_VI-ReID-v4-adamw-cosine` | 基线 + AdamW/余弦学习率 |
| `SCM_for_VI-ReID-v5-combo` | HC + MixStyle + AdamW/余弦组合 |

## 服务器下载

这是私有仓库，服务器需使用有访问权限的 GitHub 账号登录：

```bash
gh auth login
gh repo clone oiooi111/SCM-VI-ReID-Variants
cd SCM-VI-ReID-Variants
```

已配置 GitHub SSH 密钥时也可以：

```bash
git clone git@github.com:oiooi111/SCM-VI-ReID-Variants.git
```

## 环境与数据

优先使用你运行原版代码的 Python/CUDA 环境。需要相互兼容、支持服务器显卡的 `torch` 和 `torchvision`；其余依赖可以在所选版本目录运行 `python3 -m pip install -r requirements-extra.txt`。

依赖文件根据源码导入整理，尚未在 GPU 上验证。代码使用 `Image.ANTIALIAS`，因此约束 Pillow < 10；使用旧环境时建议 Python 3.10，并保留已有可工作的 PyTorch/torchvision 版本。

首次启动会下载 CLIP RN50 预训练权重，需联网或预先准备好 `~/.cache/clip/RN50.pt`。仓库不含预训练权重、训练权重、数据集或日志。

SYSU-MM01 路径末尾要保留 `/`，目录需要原图和 `exp/train_id.txt`、`exp/val_id.txt`、`exp/test_id.txt`。先在任意一个版本目录运行预处理（六个目录均包含相同脚本）：

```bash
cd SCM_for_VI-ReID-v1-hc
python3 preprocess_sysu.py --data-path /data/SYSU-MM01/
```

预处理按你最终提供的脚本合并 train + val，使用 RGB 相机 1/2/4/5 和红外相机 3/6，将原图缩放为 144×288，统一按身份排序重标号，在数据集根目录生成六个文件：

```text
train_rgb_resized_img.npy
train_rgb_resized_label.npy
train_rgb_resized_path.npy
train_ir_resized_img.npy
train_ir_resized_label.npy
train_ir_resized_path.npy
```

当前六个监督训练版本读取其中四个图像与标签文件；路径文件同时保存，供需要图像路径的流程使用。已有输出时脚本默认停止；如果曾用旧的 192×384 脚本生成数据，需加 `--overwrite` 重新生成。六份代码共用同一数据集目录时只需预处理一次。训练时 `--sysu_data_path` 仍需以 `/` 结尾。

## 独立运行一个版本

例如在某个独立 GPU 作业内运行 V1：

```bash
cd SCM_for_VI-ReID-v1-hc
bash run_variant.sh --sysu_data_path /data/SYSU-MM01/
```

其他版本只需进入对应目录，运行同名脚本。脚本沿用调度器设置的 `CUDA_VISIBLE_DEVICES`；未设置时默认使用 GPU 0。每份代码各自保存到本目录的 `experiments/` 下。

重复实验请指定不同输出目录，避免覆盖：

```bash
bash run_variant.sh --sysu_data_path /data/SYSU-MM01/ \
  --seed 2 --output_path experiments/v1_hc_seed2
```

RegDB 示例（使用对应 trial 的数据）：

```bash
bash run_variant.sh --dataset regdb --pid_num 206 \
  --regdb_data_path /data/RegDB/ --trial 1
```

## 保留的训练和评测行为

每个 epoch 评测，按测试 Rank-1 保存最佳模型；默认 140 个 Stage-3 epoch。保持现有三阶段训练、翻转测试和各版本独立输出。启动脚本默认关闭自动恢复，避免新实验误加载旧输出。

本次上传验证范围为源码语法、Shell 入口、文件范围与版本独立性；尚未完成服务器训练或实测精度验证。上传过程中未修改六份代码的训练逻辑。
