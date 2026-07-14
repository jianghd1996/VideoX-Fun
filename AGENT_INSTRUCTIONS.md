# 训练机 Agent 执行指令

> 执行命令 → 输出存文件 → 把文件内容反馈回来。
> 不要修改任何代码文件。

## 环境

- 仓库: `/mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun`
- 模型: `/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control`
- 数据: `/mnt/DataPart/jianghongda/dataset/livephoto`（GT: static, 控制: static_gs_render）
- Python 环境: `/mnt/DataPart/jianghongda/env/wan2.2/`

---

## Step 1: 停止现有训练（如果有）

```bash
tmux kill-session -t train 2>/dev/null; pkill -f train_control_lora.py 2>/dev/null; echo "stopped" | tee /tmp/step0.log
```

---

## Step 2: 修改 shell 脚本（虚拟环境 + 路径 + GPU + 进程数）

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun
cp scripts/wan2.2_fun/train_control_lora.sh scripts/wan2.2_fun/train_control_lora.sh.bak

# 在开头添加虚拟环境激活
sed -i '1i source /mnt/DataPart/jianghongda/env/wan2.2/bin/activate' scripts/wan2.2_fun/train_control_lora.sh

# 替换路径
sed -i 's|/cache/02_model/Wan2.2-Fun-5B-Control/|/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|datasets/internal_datasets/|/mnt/DataPart/jianghongda/dataset/livephoto|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|/cache/00_data/metadata.json|datasets/dataset.json|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|/cache/00_data/test_data_zhuan/|/mnt/DataPart/jianghongda/dataset/livephoto/static|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|CUDA_VISIBLE_DEVICES=0,1,2,3|CUDA_VISIBLE_DEVICES=5,6|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|accelerate launch --mixed_precision|accelerate launch --num_processes=2 --mixed_precision|g' scripts/wan2.2_fun/train_control_lora.sh

# 显示修改结果
head -10 scripts/wan2.2_fun/train_control_lora.sh | tee /tmp/step2_head.log
grep -E "MODEL_NAME|DATASET|CUDA_VISIBLE|validation" scripts/wan2.2_fun/train_control_lora.sh | tee /tmp/step2.log
```

**反馈**: 把 `/tmp/step2_head.log` 和 `/tmp/step2.log` 内容贴回来。

---

## Step 3: 在 tmux 里启动训练

```bash
tmux new-session -d -s train -c /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun "bash scripts/wan2.2_fun/train_control_lora.sh 2>&1 | tee train.log"
```

等 60 秒后检查：

```bash
sleep 60
tail -80 train.log | tee /tmp/step3.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv | tee /tmp/step3_gpu.log
```

**反馈**: 把 `/tmp/step3.log` 和 `/tmp/step3_gpu.log` 内容贴回来。

---

## Step 4: 持续监控

```bash
# 查看最近训练日志
tail -50 train.log | tee /tmp/step4.log

# 检查是否有 division by zero 错误
grep -i "division by zero\|error\|Error\|ERROR" train.log | tail -20 | tee /tmp/step4_error.log

# GPU 状态
nvidia-smi | tee /tmp/step4_gpu.log
```

**反馈**: 把 `/tmp/step4.log`、`/tmp/step4_error.log`、`/tmp/step4_gpu.log` 内容贴回来。

---

## 注意事项

1. **不要修改任何代码文件**（.py 文件）
2. 报错时把完整 traceback 存到文件再贴
3. 如果有 division by zero 错误，把相关行前后 10 行也贴出来
4. 不确定的事情，反馈回来