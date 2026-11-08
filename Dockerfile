# 数字人综合情感陪伴对话模型 - 提交镜像
# 基于组委会提供的 vLLM 0.19.0 基础镜像构建
# 构建: docker build -t <team_image> .
# 运行: 平台会执行 bash /root/participant/start.sh TEST_FILE RESULT_DIR

FROM vLLM:0.19.0-ubuntu22.04-p3.12-cuda12.8

WORKDIR /root

# ---- 训练后需要的推理依赖 ----
# transformers/accelerate 基础镜像的 vLLM 可能已包含，pip install 会自动跳过已安装版本
RUN pip install --no-cache-dir \
        "transformers>=4.40" \
        "accelerate>=0.29" \
        "peft>=0.15" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && rm -rf /root/.cache/pip

# ---- 复制参赛者产出的 participant 包 ----
# 注意: participant/model/ 必须是合并 LoRA 后的完整 HF 模型（非 LoRA adapter）
COPY participant/ /root/participant/

# ---- 官方推理样例数据（可选，用于本地 docker 内验证）----
COPY 官方数据/数字人综合情感陪伴对话模型/test_inference_data.jsonl /root/test_inference_data.jsonl

# ---- 专家评审资料（技术方案、实验设计、训练日志）----
# 放在 /root/AI技术赛道-数字人综合情感对话模型-队名/
COPY docs/ /root/AI技术赛道-数字人综合情感对话模型-队名/

# 默认禁用 HF 联网，确保离线推理
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# 健康检查（平台评测前会探测）
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python3 -c "import os; assert os.path.isdir('/root/participant/model'), 'model missing'" || exit 1

CMD ["/bin/bash"]
