import json, collections, statistics
from pathlib import Path
BASE = Path(r"D:\0001\数字人赛题\官方数据\数字人综合情感陪伴对话模型\训练-验证-数据集")

def analyze(path, name):
    emo = collections.Counter(); pers = collections.Counter(); inter = collections.Counter(); sty = collections.Counter()
    n_conv = 0; n_pp = 0; hist_lens = []; resp_lens = []; mem_use = 0; prof_empty = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line); n_conv += 1
            turns = {t["turn_id"]: t for t in row["turns"]}
            for pp in row["prediction_points"]:
                n_pp += 1
                hist_lens.append(pp["target_user_turn_id"])
                t = pp["target"]
                emo[t["emotion_label"]] += 1
                resp_lens.append(len(t["response_text"]))
                if t["memory_refs"]: mem_use += 1
                p = t["user_profile"]
                for v in p["personality_traits"]: pers[v] += 1
                for v in p["interests"]: inter[v] += 1
                for v in p["style"]: sty[v] += 1
                for k in ("personality_traits","interests","style"):
                    if not p[k]: prof_empty[k] += 1
    print(f"== {name}: {n_conv} 对话, {n_pp} 预测点")
    print(f"  情感分布: {dict(emo.most_common())}")
    print(f"  目标轮位置 min/med/max: {min(hist_lens)}/{statistics.median(hist_lens)}/{max(hist_lens)}")
    print(f"  回复字符数 min/med/p95/max: {min(resp_lens)}/{statistics.median(resp_lens)}/{sorted(resp_lens)[int(len(resp_lens)*0.95)]}/{max(resp_lens)}")
    print(f"  memory_refs 非空占比: {mem_use/n_pp:.1%}")
    print(f"  画像空数组率: { {k: f'{v/n_pp:.1%}' for k,v in prof_empty.items()} }")
    print(f"  personality: {dict(pers.most_common())}")
    print(f"  interests: {dict(inter.most_common())}")
    print(f"  style: {dict(sty.most_common())}")

analyze(BASE/"train"/"train_public.jsonl", "train")
analyze(BASE/"val"/"val_public.jsonl", "val")
