# MiMo-V2.5-ASR Batch Inference Server

这是一个类似 vLLM 动态批处理思路的高吞吐 ASR 服务：多个音频转写请求进入异步队列，由后台批处理器按时间窗口或批大小自动聚合后送入 MiMo-V2.5-ASR 推理。

## 架构

```text
Client
   ↓
FastAPI (/v1/audio/transcriptions)
   ↓
Async Queue
   ↓
Batch Scheduler (每 50ms 或达到 batch size 时触发)
   ↓
MiMo-V2.5-ASR
   ↓
Results
```

## 目录结构

```text
mimo-asr-server/
├── app.py
├── batcher.py
├── model.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── docs/
    └── spec.md
```

## 依赖

核心依赖在 `requirements.txt` 中：

```txt
fastapi
uvicorn[standard]
pydantic
python-multipart
soundfile
numpy
transformers
torch
```

如果接入官方 MiMo 代码，需要把官方推理依赖同步加入 `requirements.txt`。

## 构建

```bash
docker build -t mimo-asr-server .
```

## 启动

```bash
docker run --gpus all -d \
  -p 8000:8000 \
  -v /mnt/data/models/MiMo-V2.5-ASR:/models/MiMo-V2.5-ASR \
  --name mimo-asr \
  mimo-asr-server
```

也可以使用 compose：

```bash
docker compose up -d
```

## 健康检查

```bash
curl http://localhost:8000/health
```

返回：

```json
{"status":"ok"}
```

## 转写测试

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@test.wav"
```

返回：

```json
{
  "text": "hello world"
}
```

## OpenAI Python SDK 调用

服务路径使用 `/v1/audio/transcriptions`，可以按 OpenAI Python SDK 的 base URL 方式调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

with open("test.wav", "rb") as f:
    resp = client.audio.transcriptions.create(
        model="mimo-asr",
        file=f,
    )

print(resp.text)
```

## 性能调优

### batch size

- A800 80GB：建议 `16` 到 `64`
- H100：可根据显存和模型实现提升到 `128`

### max_wait_ms

- `20ms`：低延迟优先
- `50ms`：延迟和吞吐平衡
- `100ms`：吞吐优先

### 进程和切片

- 单 GPU 建议 1 个推理进程，避免多个进程争抢显存。
- 长音频建议先做 VAD 切片，再批量送入转写，整体吞吐更高。

## 模型接入说明

`model.py` 当前是模型适配层，`_transcribe_one` 使用占位返回值。接入正式 MiMo 推理时，需要替换为官方模型加载和推理逻辑，例如：

```python
# from src.mimo_audio.mimo_audio import MimoAudio
# self.model = MimoAudio(model_path=model_path)
# return self.model.asr_sft(wav_path)
```

## 预期性能

以下是 A800 80GB 上的方向性预估，实际结果取决于模型结构、音频时长分布、切片策略和真实 batch 推理实现。

| 场景 | 预计吞吐 |
| --- | ---: |
| 单请求 | 5x-15x realtime |
| Batch 16 | 20x-60x realtime |
| Batch 32 | 40x-100x realtime |
