# 咪咕仝学平台 - 分步执行指南

前置条件：平台实例已创建（4090 24GB + vLLM 0.19.0 + Ubuntu 22.04），conda_env 已激活。

## 方式 A：一键脚本（推荐）

如果能把本地 `数字人竞赛/` 目录推到平台实例，直接执行：

```bash
cd ~/数字人竞赛
bash platform_train.sh
# 约 6-10 小时，完成后自动验证
```

## 方式 B：分步复制粘贴（没有推目录工具时）

### Step 1 — 建目录

```bash
source activate conda_env
cd ~
mkdir -p 数字人竞赛/{participant/{configs,templates},data,checkpoints,docs}
cd 数字人竞赛
```

### Step 2 — 生成 SFT 训练数据（把下面两段代码直接贴进实例里跑）

```bash
pip install -q modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -q peft trl datasets accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

把以下内容保存为 `build_sft.py`（或直接平台上 nano 写）：

```python
import json
from pathlib import Path

EMOTIONS = ["joy","gratitude","relaxed","care","pride","neutral","surprise","mixed","sadness","loneliness","anxiety","anger","fear","disgust","shame","helplessness"]
PERSONALITY = sorted(["extroverted","introverted","open","conservative","high_conscientiousness","casual","agreeable","assertive","emotionally_stable","sensitive"])
INTERESTS = sorted(["study_exam","programming_technology","reading_writing","film_animation","music","games","sports_fitness","travel_outdoor","pets","social","career_development","art_design"])
STYLES = sorted(["brief","detailed","colloquial","formal","direct","indirect","humorous","rational","high_emotional_expression","low_emotional_expression","emoji_user"])

SP = f"""你正在参加数字情感陪伴比赛。根据截止到最新用户消息的对话历史，完成下一轮 Assistant 回复预测。

只输出一个 JSON object，不要输出 Markdown 或解释。object 必须且只能包含：
- response_text: 非空中文回复，应自然、贴合上下文和用户偏好；
- emotion_label: 最新用户消息的情绪，只能从 {", ".join(EMOTIONS)} 中选择；
- user_profile: object，包含 personality_traits、interests、style 三个字符串数组；
- memory_refs: 固定输出 []，因为测试集没有公开 memory bank。

画像标签只能使用以下枚举；没有充分证据时宁可输出空数组：
personality_traits: {", ".join(PERSONALITY)}
interests: {", ".join(INTERESTS)}
style: {", ".join(STYLES)}
"""

def render(history, sid):
    lines = [f"{t['role']}: {t['content']}" for t in history]
    return "样本 ID: " + sid + "\n对话历史:\n" + "\n".join(lines)

OFFICIAL = Path("~/官方数据/数字人综合情感陪伴对话模型/训练-验证-数据集").expanduser()
DST = Path("data"); DST.mkdir(exist_ok=True)

for split in ['train', 'val']:
    src = OFFICIAL / split / f"{split}_public.jsonl"
    dst = DST / f"sft_{split}.jsonl"
    n = 0
    with open(src, encoding='utf-8') as fin, open(dst, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip(): continue
            row = json.loads(line)
            turns = row['turns']
            conv_no = row['conversation_id'].split('_')[-1]
            for pp in row['prediction_points']:
                tid = pp['target_user_turn_id']
                history = [t for t in turns if t['turn_id'] <= tid]
                tgt = pp['target']
                target = {
                    "response_text": tgt["response_text"],
                    "emotion_label": tgt["emotion_label"],
                    "user_profile": {
                        "personality_traits": tgt["user_profile"]["personality_traits"],
                        "interests": tgt["user_profile"]["interests"],
                        "style": tgt["user_profile"]["style"],
                    },
                    "memory_refs": [],
                }
                sid = f"{split}_{conv_no}_t{tid}"
                sample = {
                    "id": sid,
                    "messages": [
                        {"role": "system", "content": SP},
                        {"role": "user", "content": render(history, sid)},
                        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                    ],
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
    print(f"{split}: {n} 样本")
```

然后：
```bash
python3 build_sft.py
```

### Step 3 — 下载基座模型

```bash
python3 -c "
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='participant/model')
print('done:', p)
"
```

### Step 4 — 训练（最耗时，约 6-10h）

把本地 `train_sft.py` 推过去，或者直接复制粘贴保存为 `train_sft.py`。然后：

```bash
export HF_HUB_OFFLINE=0
python3 train_sft.py \
    --output_dir checkpoints/lora_7b_v1 \
    --model_dir participant/model \
    --train_file data/sft_train.jsonl \
    --val_file data/sft_val.jsonl \
    --per_device_batch 4 \
    --grad_accum 4 \
    --epochs 3 \
    --lr 1e-4 \
    --max_length 1024 \
    --lora_r 32 \
    --eval_steps 500 \
    --save_steps 500
```

### Step 5 — 合并 LoRA + 覆盖 participant/model/

```python
# save as merge.py
import shutil
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(".").resolve()
MODEL_DIR = ROOT / "participant" / "model"
ADAPTER = ROOT / "checkpoints" / "lora_7b_v1" / "lora_adapter"

print("[merge] 加载基座 + LoRA ...")
base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float16, device_map="auto")
merged = PeftModel.from_pretrained(base, str(ADAPTER))
merged = merged.merge_and_unload()

# 备份基座以防需要回退
import os, time
backup = MODEL_DIR.parent / "model_backup_base"
if not backup.exists():
    backup.mkdir()
    for f in list(MODEL_DIR.iterdir()):
        if f.suffix in ('.safetensors', '.bin') and f.name.startswith('model'):
            shutil.copy2(f, backup / f.name)

merged.save_pretrained(str(MODEL_DIR), safe_serialization=True)
AutoTokenizer.from_pretrained(str(ADAPTER)).save_pretrained(str(MODEL_DIR))
print("[merge] 已覆盖 participant/model/")
```

```bash
python3 merge.py
```

### Step 6 — 用官方 harness 验证

```bash
# 改 config 为 vLLM 后端
python3 -c "
import json
cfg = json.load(open('participant/configs/inference_config.json'))
cfg['backend'] = 'openai_compatible'
cfg['model_path'] = './model'
cfg['hardware_label'] = 'NVIDIA GeForce RTX 4090 24GB (vLLM 0.19.0)'
json.dump(cfg, open('participant/configs/inference_config.json','w'), indent=2, ensure_ascii=False)
"

# 跑官方 harness（vLLM 会自动启动）
bash participant/start.sh ~/官方数据/数字人综合情感陪伴对话模型/test_inference_data.jsonl result/

# 看结果
cat result/performance_report.json | python3 -m json.tool
cat result/submission.jsonl
```

### Step 7 — 打包镜像 + 提交

```bash
docker build -t migudm-competition/team-name:v1 .
docker push <咪咕 registry>/team-name:v1

# 然后在赛事平台提交该镜像 tag
```

## 训练日志留存（专家评审 10 分）

训练输出会自动打到 stdout，**务必 tee 到文件**：

```bash
python3 train_sft.py ... 2>&1 | tee logs/train_run_$(date +%Y%m%d_%H%M%S).log
```

每 500 步会自动 eval + save，会在 `checkpoints/lora_7b_v1/` 生成 `trainer_state.json` 和 `eval_results.json`，这两份就是专家评审要看的**训练日志和调优过程**。

所有日志文件最终需要打包到镜像的 `/root/AI技术赛道-数字人综合情感对话模型-队名/` 目录下。
