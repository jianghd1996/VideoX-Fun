# 训练机 Agent 执行指令

> 本文件由开发机上的 agent 编写，供训练机上的 agent 执行。
> 执行完每一步后，将完整输出反馈给开发机 agent。

## 环境信息

- 仓库路径: 请执行 `find / -maxdepth 4 -name "VideoX-Fun" -type d 2>/dev/null` 确认
- 当前分支: `main`
- 工作流: git pull → 执行命令 → 反馈输出

---

## Task 1: 同步代码

```bash
cd <VideoX-Fun仓库路径>
git pull origin main
```

**反馈**: 输出 git pull 的结果，确认是否成功拉取了最新代码（commit f4d1c60 或更新）。

---

## Task 2: 测试数据生成脚本

当用户提供了 GT 视频路径、控制信号路径、caption 文件后，执行：

```bash
# 先用 DEBUG 模式跑，看详细输出
DEBUG=1 python scripts/generate_dataset_json.py \
    --gt_root <GT视频根目录> \
    --control_root <控制信号根目录> \
    --caption_file <caption文件路径> \
    --output dataset.json
```

**需要反馈的信息**:
1. 完整的终端输出
2. 生成的 dataset.json 前几条内容: `head -50 dataset.json`
3. 总条目数: `python -c "import json; print(len(json.load(open('dataset.json'))))"`

如果报错，请把完整的 traceback 贴出来。

---

## Task 3: 检查数据目录结构

如果数据匹配有问题，先检查目录结构：

```bash
# 查看 GT 目录结构（前3层）
find <GT视频根目录> -maxdepth 3 -type f | head -20

# 查看控制信号目录结构
find <控制信号根目录> -maxdepth 3 -type f | head -20

# 查看 caption 文件内容前几行
head -20 <caption文件路径>
```

**反馈**: 把以上三个命令的输出贴出来。

---

## Task 4: 检查训练脚本配置

确认训练脚本的参数配置：

```bash
cat scripts/wan2.2_fun/train_control_lora.sh
```

**反馈**: 贴出完整的 shell 脚本内容，特别注意:
- `--pretrained_model_name_or_path`
- `--ann_path` (数据 JSON 路径)
- `--data_root` (数据根目录)
- `--output_dir`
- GPU 相关参数

---

## Task 5: 干运行测试（不实际训练）

在正式训练前，先做一次干运行检查:

```bash
# 检查 Python 环境和依赖
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('PyTorch:', torch.__version__)"

# 检查 accelerate 配置
accelerate config  # 或者 cat ~/.cache/huggingface/accelerate/default_config.yaml

# 检查训练脚本是否能正常解析参数
python scripts/wan2.2_fun/train_control_lora.py --help 2>&1 | head -30
```

**反馈**: 贴出以上命令的输出。

---

## 反馈格式要求

每次执行完任务后，请按以下格式反馈:

```
## Task X 执行结果

**状态**: 成功 / 失败
**命令**: <执行的命令>
**输出**:
<完整输出>
**备注**: <任何异常或观察>
```

---

## 注意事项

1. 不要修改任何代码文件，只执行命令
2. 如果遇到报错，完整复制 traceback
3. 如果路径不存在，用 `find` 命令搜索
4. 不确定的事情，问开发机 agent（把问题反馈回来）
