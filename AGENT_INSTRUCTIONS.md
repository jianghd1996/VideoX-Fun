# 训练机 Agent 执行指令

> 执行命令 → 输出存文件 → 把文件内容反馈回来。
> 不要修改任何代码文件。

## 环境

- 仓库: `/mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun`
- 模型: `/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control`
- 数据: `/mnt/DataPart/jianghongda/dataset/livephoto`（GT: static, 控制: static_gs_render）

---

## Step 1: 修改 shell 脚本（路径 + GPU + 进程数）

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun
cp scripts/wan2.2_fun/train_control_lora.sh scripts/wan2.2_fun/train_control_lora.sh.bak

sed -i 's|/cache/02_model/Wan2.2-Fun-5B-Control/|/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|datasets/internal_datasets/|/mnt/DataPart/jianghongda/dataset/livephoto|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|/cache/00_data/metadata.json|datasets/dataset.json|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|/cache/00_data/test_data_zhuan/|/mnt/DataPart/jianghongda/dataset/livephoto/static|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|CUDA_VISIBLE_DEVICES=0,1,2,3|CUDA_VISIBLE_DEVICES=5,6|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|accelerate launch --mixed_precision|accelerate launch --num_processes=2 --mixed_precision|g' scripts/wan2.2_fun/train_control_lora.sh

cat scripts/wan2.2_fun/train_control_lora.sh | tee /tmp/step1.log
```

**反馈**: 把 `/tmp/step1.log` 内容贴回来。

---

## Step 2: 在 tmux 里启动正式训练

```bash
tmux new-session -d -s train -c /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun "bash scripts/wan2.2_fun/train_control_lora.sh > train.log 2>&1"
```

等 60 秒后检查：

```bash
sleep 60
tail -50 train.log | tee /tmp/step2.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv | tee /tmp/step2_gpu.log
```

**反馈**: 把 `/tmp/step2.log` 和 `/tmp/step2_gpu.log` 内容贴回来。

---

## 注意事项

1. **不要修改任何代码文件**
2. 报错时把完整 traceback 存到文件再贴
3. 不确定的事情，反馈回来
