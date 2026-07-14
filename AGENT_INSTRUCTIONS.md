# 训练机 Agent 执行指令

> 代码已修改，训练机只需 git pull + 验证。
> 测试通过后不要启动训练，把启动命令反馈回来。

## Step 1: 同步代码

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun
git pull origin main 2>&1 | tee /tmp/step1.log
```

**反馈**: `/tmp/step1.log` 内容

---

## Step 2: 验证 shell 脚本路径

```bash
cat scripts/wan2.2_fun/train_control_lora.sh | head -20 | tee /tmp/step2.log
```

**反馈**: `/tmp/step2.log` 内容，确认路径正确。

---

## Step 3: 干运行测试（2步）

```bash
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun
source /mnt/DataPart/jianghongda/env/wan2.2/bin/activate

CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes=2 --mixed_precision=bf16 scripts/wan2.2_fun/train_control_lora.py --config_path=config/wan2.2/wan_civitai_5b.yaml --pretrained_model_name_or_path=/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-Fun-5B-Control --train_data_dir=/mnt/DataPart/jianghongda/dataset/livephoto --train_data_meta=datasets/dataset.json --image_sample_size=1440 --video_sample_size=1440 --token_sample_size=1440 --video_sample_stride=1 --video_sample_n_frames=81 --train_batch_size=1 --video_repeat=1 --gradient_accumulation_steps=1 --dataloader_num_workers=4 --max_train_steps=2 --checkpointing_steps=1000 --learning_rate=1e-04 --seed=42 --output_dir=output_dir_test --gradient_checkpointing --mixed_precision=bf16 --adam_weight_decay=3e-2 --adam_epsilon=1e-10 --vae_mini_batch=1 --max_grad_norm=0.05 --random_hw_adapt --training_with_video_token_length --enable_bucket --uniform_sampling --train_mode=control_ref --control_ref_image=random --add_inpaint_info --mask_concat_channels 1 --control_mask_ratio=0.3 --add_full_ref_image_in_self_attention --rank=64 --network_alpha=32 --target_name=q,k,v,ffn.0,ffn.2 --use_peft_lora --low_vram > /tmp/step3.log 2>&1

tail -30 /tmp/step3.log | tee /tmp/step3_tail.log
grep -E "ERROR|error|Traceback|loaded|Loaded|step" /tmp/step3.log | head -20 | tee /tmp/step3_summary.log
echo "Exit code: $?" | tee /tmp/step3_exit.log
```

**反馈**: `/tmp/step3_tail.log`、`/tmp/step3_summary.log`、`/tmp/step3_exit.log` 内容

---

## Step 4: 测试通过后反馈

测试通过后，反馈以下内容：

```
✅ 测试通过

启动命令（在 tmux 里执行）：
cd /mnt/DataPart/jianghongda/VideoX-Fun-dev/VideoX-Fun
source /mnt/DataPart/jianghongda/env/wan2.2/bin/activate
bash scripts/wan2.2_fun/train_control_lora.sh
```

**不要启动训练**，把启动命令贴回来即可。