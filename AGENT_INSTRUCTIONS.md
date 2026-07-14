# 训练机 Agent 执行指令

> 本文件由开发机上的 agent 编写，供训练机上的 agent 执行。
> 上次反馈已处理：路径已修正，训练脚本路径已更新。

## 环境信息

- 仓库路径: `/mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun`
- 当前分支: `main`
- 模型路径: `/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control`
- 数据根目录: `/mnt/DataPart/jianghongda/dataset/livephoto`
  - GT 视频: `static`
  - 控制信号: `static_gs_render`
  - Captions: `static_captions.json`

---

## Task 1: 同步最新代码

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun
git pull origin main
```

**反馈**: git pull 结果，确认 commit 更新。

---

## Task 2: 用已生成的 dataset.json 替换训练脚本里的路径

dataset.json 已经生成成功（4868 条匹配），现在需要把它拷贝到训练脚本能读到的位置，并修改训练脚本参数。

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun

# 把 dataset.json 拷贝到仓库内（训练脚本的 train_data_meta 参数指向这里）
cp dataset.json datasets/dataset.json

# 确认文件存在
ls -lh datasets/dataset.json
head -20 datasets/dataset.json
```

**反馈**: 确认文件已拷贝，内容正确。

---

## Task 3: 修改训练脚本 shell 文件

训练脚本 `scripts/wan2.2_fun/train_control_lora.sh` 里的路径需要更新。

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun

# 备份原文件
cp scripts/wan2.2_fun/train_control_lora.sh scripts/wan2.2_fun/train_control_lora.sh.bak

# 用 sed 替换路径
sed -i 's|/cache/02_model/Wan2.2-Fun-5B-Control/|/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|datasets/internal_datasets/|/mnt/DataPart/jianghongda/dataset/livephoto|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|/cache/00_data/metadata.json|datasets/dataset.json|g' scripts/wan2.2_fun/train_control_lora.sh
sed -i 's|/cache/00_data/test_data_zhuan/|/mnt/DataPart/jianghongda/dataset/livephoto/static|g' scripts/wan2.2_fun/train_control_lora.sh

# 验证替换结果
grep -E "MODEL_NAME|DATASET|validation_data_dir" scripts/wan2.2_fun/train_control_lora.sh
```

**反馈**: grep 输出，确认路径已正确替换。

---

## Task 4: 干运行测试（2步验证）

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun

# 先确认模型目录有文件
ls /mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control/ | head -20

# 干运行：只跑 2 步，验证模型加载 + 数据读取正常
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision=bf16 scripts/wan2.2_fun/train_control_lora.py \
    --config_path=config/wan2.2/wan_civitai_5b.yaml \
    --pretrained_model_name_or_path=/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control \
    --train_data_dir=/mnt/DataPart/jianghongda/dataset/livephoto \
    --train_data_meta=datasets/dataset.json \
    --image_sample_size=1440 \
    --video_sample_size=1440 \
    --token_sample_size=1440 \
    --video_sample_stride=1 \
    --video_sample_n_frames=81 \
    --train_batch_size=1 \
    --video_repeat=1 \
    --gradient_accumulation_steps=1 \
    --dataloader_num_workers=4 \
    --max_train_steps=2 \
    --checkpointing_steps=1000 \
    --learning_rate=1e-04 \
    --seed=42 \
    --output_dir=output_dir_test \
    --gradient_checkpointing \
    --mixed_precision=bf16 \
    --adam_weight_decay=3e-2 \
    --adam_epsilon=1e-10 \
    --vae_mini_batch=1 \
    --max_grad_norm=0.05 \
    --random_hw_adapt \
    --training_with_video_token_length \
    --enable_bucket \
    --uniform_sampling \
    --train_mode=control_ref \
    --control_ref_image=random \
    --add_inpaint_info \
    --mask_concat_channels 1 \
    --control_mask_ratio=0.3 \
    --add_full_ref_image_in_self_attention \
    --rank=64 \
    --network_alpha=32 \
    --target_name=q,k,v,ffn.0,ffn.2 \
    --use_peft_lora \
    --low_vram \
    2>&1 | head -150
```

> 注意：去掉了 `--validation_data_dir` 和 `--num_inference_steps`，先不跑验证，只验证训练流程能跑通。

**反馈**: 完整输出，特别关注：
- 是否有 import / 路径错误
- 模型是否成功加载（应该看到 `Loading model...` 之类的日志）
- 数据是否正确读取（应该看到 `Loaded X samples` 或类似日志）
- 是否在 2 步后正常结束

---

## Task 5: 如果 Task 4 成功，正式训练

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun

# 确认 Task 4 没问题后，用完整参数跑正式训练
nohup bash scripts/wan2.2_fun/train_control_lora.sh > train.log 2>&1 &

# 监控训练进度
tail -f train.log
```

**反馈**: 
- `tail train.log` 前 50 行
- GPU 占用情况: `nvidia-smi`
- 确认训练已正常启动

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
