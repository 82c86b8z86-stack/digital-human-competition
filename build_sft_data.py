"""把官方 train/val 数据转换为 SFT 训练样本（与官方 run_inference.py 的提示格式逐字对齐）。"""
import json, sys
from pathlib import Path

SRC = Path(r"D:\0001\数字人赛题\官方数据\数字人综合情感陪伴对话模型\训练-验证-数据集")
OUT = Path(r"D:\0001\C题\数字人竞赛\data")
OUT.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise",
            "mixed", "sadness", "loneliness", "anxiety", "anger", "fear", "disgust",
            "shame", "helplessness"]
PERSONALITY = sorted(["extroverted", "introverted", "open", "conservative", "high_conscientiousness", "casual", "agreeable", "assertive", "emotionally_stable", "sensitive"])
INTERESTS = sorted(["study_exam", "programming_technology", "reading_writing", "film_animation", "music", "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design"])
STYLES = sorted(["brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous", "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user"])

SYSTEM_PROMPT = f"""你正在参加数字情感陪伴比赛。根据截止到最新用户消息的对话历史，完成下一轮 Assistant 回复预测。

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

def convert(src_path, out_path, id_prefix):
    n = 0; mismatch = 0
    with open(src_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            row = json.loads(line)
            turns = row["turns"]
            conv_no = row["conversation_id"].split("_")[-1]
            for pp in row["prediction_points"]:
                tid = pp["target_user_turn_id"]
                history = [t for t in turns if t["turn_id"] <= tid]
                aid = pp["assistant_response_turn_id"]
                ref_turn = next((t for t in turns if t["turn_id"] == aid), None)
                tgt = pp["target"]
                if ref_turn and ref_turn["content"] != tgt["response_text"]:
                    mismatch += 1
                sid = f"{id_prefix}_{conv_no}_t{tid}"
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
                sample = {
                    "id": sid,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": render(history, sid)},
                        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                    ],
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n += 1
    print(f"{src_path.name} -> {out_path.name}: {n} 样本, response 与 turns 不一致 {mismatch} 处")

convert(SRC/"train"/"train_public.jsonl", OUT/"sft_train.jsonl", "train")
convert(SRC/"val"/"val_public.jsonl", OUT/"sft_val.jsonl", "val")
