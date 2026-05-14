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
accelerate>=1.9.0
torch==2.6.0
torchaudio==2.6.0
transformers==4.49.0
fastapi>=0.116.1
uvicorn[standard]>=0.35.0
pydantic>=2.11.7
python-multipart
librosa>=0.11.0
scipy>=1.15.3,<1.16
triton==3.2.0
soundfile
numpy
gradio==5.46.1
zhon==2.1.1
```

`requirements.txt` 已合入官方 MiMo-V2.5-ASR 的基础依赖，并保留服务所需的 `python-multipart`。镜像内还需要存在 `src.mimo_audio.mimo_audio` 可导入的官方源码。官方文档要求 Python 3.12、CUDA >= 12.0；如使用 `flash-attn`，建议按运行环境安装匹配 wheel。

## 构建

```bash
docker build -t mimo-asr-server .
```

## 启动

```bash
docker run --gpus '"device=3"' -d \
  -p 28203:8000 \
  -v /mnt/data/models/XiaomiMiMo/MiMo-V2.5-ASR:/models/MiMo-V2.5-ASR:ro \
  -v /mnt/data/models/XiaomiMiMo/MiMo-Audio-Tokenizer:/models/MiMo-Audio-Tokenizer:ro \
  -e MIMO_MODEL_PATH=/models/MiMo-V2.5-ASR \
  -e MIMO_TOKENIZER_PATH=/models/MiMo-Audio-Tokenizer \
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

`model.py` 已接入官方 MiMo 推理入口，默认导入 `src.mimo_audio.mimo_audio.MimoAudio`，并调用 `asr_sft(wav_path)` 转写。

默认路径：

- 模型：`/models/MiMo-V2.5-ASR`
- 音频 tokenizer：`/models/MiMo-Audio-Tokenizer`

可通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MIMO_MODEL_PATH` | `/models/MiMo-V2.5-ASR` | MiMo-V2.5-ASR 权重目录 |
| `MIMO_TOKENIZER_PATH` | `/models/MiMo-Audio-Tokenizer` | MiMo-Audio-Tokenizer 目录 |
| `MIMO_SOURCE_PATH` | 空 | 官方源码路径，需包含 `src/` 包 |
| `MIMO_AUDIO_TAG` | 空 | 如果官方 `asr_sft` 支持 `audio_tag` 参数，则自动传入 |
| `ASR_MAX_BATCH_SIZE` | `16` | 最大动态 batch 大小 |
| `ASR_MAX_WAIT_MS` | `50` | 动态 batch 最大等待毫秒数 |
| `ASR_TMP_DIR` | `/tmp/mimo_asr` | 上传音频临时目录 |

如果镜像内没有官方 MiMo 源码，服务启动时会报错提示安装源码或设置 `MIMO_SOURCE_PATH`，不会再返回占位文本。

## 预期性能

以下是 A800 80GB 上的方向性预估，实际结果取决于模型结构、音频时长分布、切片策略和真实 batch 推理实现。

| 场景 | 预计吞吐 |
| --- | ---: |
| 单请求 | 5x-15x realtime |
| Batch 16 | 20x-60x realtime |
| Batch 32 | 40x-100x realtime |
