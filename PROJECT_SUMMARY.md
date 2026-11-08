# 数字人综合情感陪伴对话模型 — 项目小结

> **赛事**: 动感地带 AI+ 高校创智计划（咪咕 aigpu.migu.cn）  
> **赛道**: AI 技术赛道  
> **核心要求**: Docker 镜像离线运行，入口 `bash /root/participant/start.sh TEST_FILE RESULT_DIR`，评测自动打分（情感识别 25 + 回应质量 20 + 用户画像 15 + 推理性能 10 + 专家评审 10）

---

## 一、总体进度

| 阶段 | 状态 | 产出 |
|---|---|---|
| 数据 EDA & Schema 对齐 | ✅ 完成 | 18657 训练 / 2354 验证，情感不平衡（anxiety 34% vs surprise 0.3%） |
| SFT 数据管线 | ✅ 完成 | `build_sft_data.py` 逐字对齐官方 harness prompt，训练目标 0 处错配 |
| LoRA 训练（本地 smoke） | ✅ 完成 | 10 步 loss 12.5→2.6，eval_loss 2.4，0 parse_failure |
| LoRA merge + 官方 harness | ✅ 完成 | 合并后跑 3 条样例，parse_failure_count=0，情感从全 sadness 修正为 anxiety |
| 平台部署脚本 & Docker | ✅ 完成 | `platform_train.sh` 一键全流程，Dockerfile 基于官方 vLLM 0.19.0 |
| 咪咕平台全量训练 | ⏳ 待执行 | 4090 24GB 单卡，7B LoRA r=32, batch=4×4, 3 epochs |
| 正式提交镜像 | ⏳ 待执行 | merge LoRA → 打包 → 推 registry |

---

## 二、技术栈

| 组件 | 选型 | 原因 |
|---|---|---|
| 基座模型 | Qwen2.5-7B-Instruct（平台）/ 1.5B（本地验证） | 4090 24GB 足够 fp16 + LoRA，7B > 3B 质量 |
| 训练方式 | LoRA，r=32，attention+MLP 7 个投影 | 1.2% 可训练参数，排除 lm_head（tie_word_embeddings 坑） |
| 训练框架 | transformers + peft + accelerate | TRL 1.16 SFTTrainer API 大改，回退标准 Trainer 更稳 |
| 推理后端 | vLLM 0.19.0（openai_compatible） | 官方基础镜像预装，批处理 100 条快 5-10× |
| Docker 基础镜像 | `vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8` | 官方提供 |

---

## 三、Smoke 验证结果（Qwen2.5-1.5B，本地 RTX 5060 8GB）

| 指标 | 基座输出 | LoRA 10 步 |
|---|---|---|
| parse_failure_count | 0 | **0** |
| 情感分类 | 全 sadness（通用回答） | **anxiety 修正正确** |
| loss | 12.5（初始） | **2.6** |
| eval_loss | — | **2.4** |
| 显存占用 | — | ~5GB（gradient checkpointing 后） |

全链路（训练 → merge_and_unload → 官方 harness 推理 → JSON schema 校验）**零断点**。

### 截断丢弃率验证

```
total: 18657
truncated (>1024 tokens): 50.4%
dropped (assistant token 全 -100): 0.0%
history 轮次: p95=2, max=2
```

max_length=1024 足够，截断后 assistant 回复始终在尾部保留。

---

## 四、关键踩坑 & 解决方案

| 坑 | 根因 | 解决方案 |
|---|---|---|
| peft tie_word_embeddings 下挂 lm_head 报错 | Qwen2.5 tie 输入输出 embedding，peft 在 tied layer 上有 bug | 排除 lm_head，训练目标模块仅 attention+MLP |
| TRL 1.16 SFTTrainer 签名大改 | packing/tokenizer 参数移除 | 回退标准 transformers Trainer |
| Windows OOM（1.5B + 2048） | 14GB 显存需求 | 降到 max_length=1024 + gradient checkpointing |
| start.sh tp=2 单卡必挂 | 官方默认双卡 | 改为 `--tensor-parallel-size 1` |
| 基座输出全 sadness | 没经过情感数据 SFT | LoRA 微调后 emotion 修正正确 |

---

## 五、提交前 Checklist

- [ ] 咪咕平台实例上 `bash platform_train.sh` 跑完 7B LoRA 3 epoch
- [ ] merge_and_unload 覆盖 `participant/model/`
- [ ] `bash participant/start.sh test_inference_data.jsonl result/` 验证 parse_failure=0
- [ ] `performance_report.json` latency 符合 10 分要求
- [ ] Docker 镜像打包 + push
- [ ] `/root/AI技术赛道-数字人综合情感对话模型-队名/` 放专家评审材料（训练日志、调优记录）

---

## 六、仓库结构

```
数字人竞赛/
├── .gitignore                 # 排除大模型权重 / checkpoint / data jsonl
├── Dockerfile                 # 基于 vLLM 0.19.0 基础镜像
├── .dockerignore
├── README.md                  # 项目说明
├── PLATFORM_GUIDE.md          # 平台分步执行指南
├── train_sft.py               # LoRA 训练核心脚本（支持 --smoke）
├── build_sft_data.py          # SFT 数据生成器（逐字对齐 harness prompt）
├── eda.py                     # 数据探索统计
├── platform_train.sh          # 平台一键: 数据→下载→训练→merge→验证
├── checkpoints/smoke/lora_adapter/  # smoke 训练最终 LoRA（6 文件，KB 级）
└── participant/               # 官方推理包（start.sh tp=1, backend=openai_compatible）
```

---

*生成时间: 2026-11-08*
