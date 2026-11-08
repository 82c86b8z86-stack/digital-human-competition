# 数字人综合情感陪伴对话模型 - 提交清单

## 最终目录结构（Docker 镜像内）

```
/root/
├── participant/                    ← 官方推理工具 v2（唯一入口 start.sh）
│   ├── start.sh                    ← 已改 tp=1（4090 单卡）
│   ├── run_inference.py            ← 官方推理脚本（原样）
│   ├── requirements.txt            ← 官方依赖（原样）
│   ├── configs/
│   │   └── inference_config.json   ← backend=openai_compatible, max_new_tokens=512
│   ├── model/                      ← 训练后 merge_and_unload 的完整 HF 模型
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer*.json
│   └── templates/                  ← 官方 schema（原样）
├── test_inference_data.jsonl       ← 官方样例（Docker 内验证用）
└── AI技术赛道-数字人综合情感对话模型-队名/  ← 专家评审 10 分材料
    ├── 技术方案.md
    ├── 训练日志.csv
    └── 实验记录.md
```

## 提交流程

```bash
# 平台实例上
bash platform_train.sh          # 一键: 数据→依赖→训练→合并→验证

# 本地（或平台实例）打包
docker build -t <your_team_tag>:v1 .
docker push <registry>/<your_team_tag>:v1
```

## 关键参数汇总

| 项 | 值 | 原因 |
|---|---|---|
| 基座 | Qwen2.5-7B-Instruct | 24GB 足够，7B 质量 > 3B |
| 训练方式 | LoRA r=32, 目标模块 attention+MLP | 1.2% 可训练参数 |
| 训练 batch | per_device=4, grad_accum=4, 有效=16 | 24GB 显存余量充足 |
| max_length | 1024 | 50% 样本截断但 0 丢弃 |
| lr | 1e-4, cosine | LoRA 常用值 |
| epochs | 3 | 约 2300 步 |
| 推理后端 | openai_compatible（vLLM 0.19.0） | 批处理 100 条快 5-10x |
| max_new_tokens | 512 | 训练集 p95=806 字符≈400 token |
| tp | 1 | 单卡 4090 |

## Smoke 验证结果（本地 RTX 5060 8GB, Qwen2.5-1.5B）

| 指标 | 基座 | 10 步 LoRA 微调 |
|---|---|---|
| parse_failure | 0 | 0 |
| 情感正确率 | sadness 全错 | anxiety 修对 |
| loss | 12.5 (初始) | 2.6 |
| eval_loss | — | 2.4 |

全链路（训练→merge→harness 推理→JSON schema 校验）**零断点**。
