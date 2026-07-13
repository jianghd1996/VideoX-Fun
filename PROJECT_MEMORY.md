# VideoX-Fun 项目工作记忆

## 项目概况
- **仓库**: jianghd1996/VideoX-Fun（fork from aigc-apps/VideoX-Fun）
- **本地路径**: /home/admin/VideoX-Fun
- **当前分支**: main
- **最新提交**: a02255f (2026-06-03)

## 开发环境
- **本机**（联网）: 改代码、写脚本、git push
- **训练机**（不联网，有 GPU/数据/模型）: git pull、跑训练、debug
- 工作流: 本机改 → push → 训练机 pull → 运行 → 反馈报错 → 本机修

## 当前开发方向
基于现有的 Mask Concat 方案，继续开发 Wan2.2 Control LoRA 训练。

### 输入数据
- 原始 RGB 视频（GT）
- 控制信号视频（Control）
- 打标结果（Caption）
- Mask 信息（可选，没有时 fallback 到全 1）

### 已有工具
- `scripts/generate_dataset_json.py` — 数据集 JSON 生成脚本
  - 支持 GT/控制信号/caption 自动匹配
  - 输出符合 ImageVideoControlDataset 格式

## 上游对比
- 上游领先 2 个提交（Lens model, Self-Forcing），跟 Control LoRA 无关
- 本地领先 42 个提交（Mask 方案、验证系统、训练流程改造）
- 关键 bug 修复待 cherry-pick: 7851e14 (motion_sub_loss 维度), 8c34acc (control_latents 传参)

## 代码结构备忘
- **Dataset**: `videox_fun/data/dataset_image_video.py` → ImageVideoControlDataset
- **训练脚本**: `scripts/wan2.2_fun/train_control_lora.py`
- **训练启动**: `scripts/wan2.2_fun/train_control_lora.sh`
- **Transformer**: `videox_fun/models/wan_transformer3d.py` → WanTransformer3DModel
- **Mask 逻辑**: mask_concat_channels 参数控制（默认 0=关闭，>0=启用 Mask Concat）
  - 输入: mask_for_concat concat 到 noisy_latents channel 维
  - Loss: loss_mask 加权（已知 1.0，未知 0.1）
  - Fallback: 无 mask_path 时用全 1 mask

## TODO / 下一步
- [ ] 等用户提供实际数据路径，测试 generate_dataset_json.py
- [ ] 考虑 cherry-pick 上游 bug 修复
- [ ] 根据训练机反馈调试
