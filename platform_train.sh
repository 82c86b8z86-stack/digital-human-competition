#!/usr/bin/env bash
# 咪咕仝学平台全量训练脚本（4090 24GB 单卡）
# 执行顺序: 数据 → 依赖 → 训练 → 合并 → 验证
# 在平台实例上: bash platform_train.sh

set -euo pipefail
source activate conda_env

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=0

PROJ_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${PROJ_DIR}/participant/model"
TRAIN_SCRIPT="${PROJ_DIR}/train_sft.py"
DATA_DIR="${PROJ_DIR}/data"
OUT_DIR="${PROJ_DIR}/checkpoints"

echo "========================================="
echo " [1/5] 下载数据"
echo "========================================="

# 如果还没有训练/验证 jsonl，从官方数据目录复制
if [[ -d "${PROJ_DIR}/官方数据/数字人综合情感陪伴对话模型/训练-验证-数据集" ]]; then
    echo "从官方数据目录复制训练/验证 jsonl"
    mkdir -p "${DATA_DIR}"
    python3 - <<PYEOF
import json
from pathlib import Path
SRC = Path("${PROJ_DIR}/官方数据/数字人综合情感陪伴对话模型/训练-验证-数据集")
DST = Path("${DATA_DIR}")
DST.mkdir(parents=True, exist_ok=True)

# 逐行生成 SFT 训练样本（对齐官方 harness prompt）
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

for split in ['train', 'val']:
    src = SRC / split / f"{split}_public.jsonl"
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
    print(f"{split}: {n} 样本 -> {dst}")
PYEOF
else
    echo "[WARN] 官方数据目录不存在，跳过 SFT 数据生成"
fi

echo "========================================="
echo " [2/5] 下载基座模型 Qwen2.5-7B-Instruct"
echo "========================================="
if [[ -d "${MODEL_DIR}/config.json" ]]; then
    echo "模型已存在，跳过下载"
else
    pip install -q modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    python3 -c "
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='${MODEL_DIR}')
print('done:', p)
"
fi

echo "========================================="
echo " [3/5] 安装训练依赖"
echo "========================================="
pip install -q peft trl datasets accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "========================================="
echo " [4/5] 全量 LoRA 训练"
echo "========================================="
python3 "${TRAIN_SCRIPT}" \
    --output_dir "${OUT_DIR}/lora_7b_v1" \
    --model_dir "${MODEL_DIR}" \
    --train_file "${DATA_DIR}/sft_train.jsonl" \
    --val_file "${DATA_DIR}/sft_val.jsonl" \
    --per_device_batch 4 \
    --grad_accum 4 \
    --epochs 3 \
    --lr 1e-4 \
    --max_length 1024 \
    --lora_r 32 \
    --eval_steps 500 \
    --save_steps 500

echo "========================================="
echo " [5/5] 合并 LoRA 并验证"
echo "========================================="
python3 - <<PYEOF
import json, os, sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("${PROJ_DIR}")
MODEL_DIR = ROOT / "participant" / "model"
ADAPTER = ROOT / "checkpoints" / "lora_7b_v1" / "lora_adapter"
MERGED = ROOT / "participant" / "model_merged"

print("[merge] 加载基座 + LoRA ...")
base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float16, device_map="auto")
merged = PeftModel.from_pretrained(base, str(ADAPTER))
merged = merged.merge_and_unload()
merged.save_pretrained(str(MERGED), safe_serialization=True)
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
tok.save_pretrained(str(MERGED))
print(f"[merge] 已保存到 {MERGED}")

# 把合并后的完整模型复制覆盖 participant/model/（评测脚本从这里加载）
import shutil
for item in MERGED.iterdir():
    dst = MODEL_DIR / item.name
    if dst.exists(): dst.unlink()
    shutil.copy2(item, dst)
print("[merge] 已覆盖 participant/model/")
PYEOF

echo ""
echo "========================================="
echo " [DONE] 训练完成，可打包 Docker 镜像提交"
echo " 验证命令: bash /root/participant/start.sh /root/test_inference_data.jsonl /root/result"
echo "========================================="
