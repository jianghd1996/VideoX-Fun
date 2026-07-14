# 训练机 Agent 执行指令

> 本文件由开发机上的 agent 编写，供训练机上的 agent 执行。
> 代码已更新并推送到 GitHub（commit 305be16），请先 pull 再执行。

## 环境信息

- 仓库路径: `/mnt/DataPart/jianghd1996/VideoX-Fun-dev/VideoX-Fun`
- 当前分支: `main`
- 工作流: 先 git pull → 逐步执行 → 完整反馈输出

---

## Task 1: 同步最新代码

已修复 `scripts/generate_dataset_json.py` 中的两个 bug：
- caption 文件格式不支持 list-of-objects
- `Path(key).stem` 把目录路径吃掉了，导致 0 匹配

```bash
cd /mnt/DataPart/jianghd1996/VideoX-Fun-dev/VideoX-Fun
git pull origin main
```

**反馈**: git pull 的结果，确认 commit 是 `305be16` 或更新。

---

## Task 2: 重新生成 dataset.json

用修复后的脚本重新跑：

```bash
cd /mnt/DataPart/jianghd1996/VideoX-Fun-dev/VideoX-Fun

DEBUG=1 python scripts/generate_dataset_json.py \
    --gt_root /mnt/DataPart/jianghongda/dataset/livephoto/static \
    --control_root /mnt/DataPart/jianghongda/dataset/livephoto/static_control \
    --caption_file /mnt/DataPart/jianghongda/dataset/livephoto/static_captions_dict.json \
    --output dataset.json
```

**需要反馈的信息**:
1. 完整的终端输出
2. Matched 数量是否接近 4868
3. 生成的 dataset.json 前 3 条: `head -30 dataset.json`
4. 总条目数: `python -c "import json; print(len(json.load(open('dataset.json'))))"`

---

## Task 3: 配置 accelerate

上次反馈 accelerate 未配置，训练前必须完成。

```bash
# 方案 A：交互式配置（推荐 4 卡训练）
accelerate config
# 选择: multi-GPU, 4 GPUs, DeepSpeed 或 FSDP 根据需求, mixed precision fp16

# 方案 B：如果交互不方便，直接写配置文件
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '0,1,2,3'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
```

**反馈**: `cat ~/.cache/huggingface/accelerate/default_config.yaml` 的完整内容。

---

## Task 4: 验证训练脚本可以正常解析参数

```bash
cd /mnt/DataPart/jianghd1996/VideoX-Fun-dev/VideoX-Fun

# 先确认模型路径存在
ls -la /cache/02_model/Wan2.2-Fun-5B-Control/ | head -20

# 确认数据路径存在
ls -la /cache/00_data/metadata.json 2>/dev/null
ls -la datasets/internal_datasets/ 2>/dev/null

# 测试训练脚本参数解析（上次失败了）
python scripts/wan2.2_fun/train_control_lora.py --help 2>&1 | head -50
```

**反馈**: 以上三个命令的完整输出。如果 `--help` 报错，贴完整 traceback。

---

## Task 5: 干运行测试（不实际训练）

如果 Task 4 的 `--help` 成功了，尝试加载模型验证配置：

```bash
cd /mnt/DataPart/jianghd1996/VideoX-Fun-dev/VideoX-Fun

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/wan2.2_fun/train_control_lora.py \
    --pretrained_model_name_or_path /cache/02_model/Wan2.2-Fun-5B-Control/ \
    --train_data_dir datasets/internal_datasets/ \
    --train_data_meta /cache/00_data/metadata.json \
    --output_dir output_dir_wan2.2_5b_control_lora \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 2 \
    --logging_steps 1 \
    2>&1 | head -100
```

> 注意: 这里加了 `--max_train_steps 2`，只跑 2 步验证流程能走通，不会真正训练。

**反馈**: 完整的输出，特别关注：
- 是否有 import 错误
- 模型是否成功加载
- 数据是否正确读取

---

## 反馈格式

每个 Task 执行完后，按以下格式反馈：

```
### Task X 执行结果

**状态**: 成功 / 失败
**输出**:
<完整终端输出>
**备注**: <任何异常或观察>
```

---

## 注意事项

1. **不要修改任何代码文件**，只执行命令
2. 遇到报错，完整复制 traceback
3. 路径不存在，用 `find` 搜索实际路径
4. 不确定的事情，把问题反馈回来，由开发机 agent 处理
