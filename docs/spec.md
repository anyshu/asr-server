# MiMo-V2.5-ASR Server Spec

## 目标

提供一个基于 FastAPI 的 MiMo-V2.5-ASR 批量转写服务，对外暴露音频转写 HTTP 接口，并通过 Docker 运行在具备 NVIDIA GPU 的环境中。

该服务采用类似 vLLM 的动态批处理架构：多个请求进入异步队列，由后台 `BatchScheduler` 在达到批大小或等待窗口到期时统一提交给模型推理，以提升吞吐。

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

## 模块职责

### `app.py`

- 创建 FastAPI 应用。
- 在 lifespan 中启动和停止 `BatchScheduler`。
- 提供 `/health` 健康检查接口。
- 提供 `/v1/audio/transcriptions` 文件上传转写接口。
- 将上传文件临时写入 `/tmp/mimo_asr`，请求完成后删除。

### `batcher.py`

- 维护异步请求队列。
- 将请求封装为 `RequestItem`，通过 future 返回结果。
- 按 `max_batch_size` 和 `max_wait_ms` 聚合批次。
- 调用 `model.transcribe_batch` 执行批量转写。
- 将转写结果或异常回填到对应请求。

### `model.py`

- 作为 MiMo-V2.5-ASR 的模型适配层。
- 通过单线程 `ThreadPoolExecutor` 执行同步推理，避免阻塞事件循环。
- 默认导入官方 `src.mimo_audio.mimo_audio.MimoAudio`，调用 `asr_sft` 执行真实转写。

## 服务接口

### `GET /health`

用于服务健康检查。

响应示例：

```json
{"status":"ok"}
```

### `POST /v1/audio/transcriptions`

上传单个音频文件并返回转写文本。

请求：

- Content-Type: `multipart/form-data`
- 字段：`file`

响应：

```json
{"text":"transcription result"}
```

## OpenAI SDK 兼容调用

服务路径为 `/v1/audio/transcriptions`，可通过 OpenAI Python SDK 设置 `base_url` 调用：

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

当前接口只使用上传文件，`model` 参数主要用于 SDK 调用兼容。

## 批处理调度

默认配置：

- 最大批大小：`16`
- 最大等待时间：`50ms`

调度流程：

1. 从队列中等待首个请求。
2. 记录当前批次截止时间。
3. 在截止时间前继续从队列聚合请求。
4. 达到 `max_batch_size` 或超过 `max_wait_ms` 后触发推理。
5. 将模型结果按请求顺序回填。

异常处理：批量转写失败时，将异常回填到该批次所有未完成的 future。

## 模型适配

默认模型路径：

```text
/models/MiMo-V2.5-ASR
```

`MimoASRModel` 默认导入官方推理入口：

```python
from src.mimo_audio.mimo_audio import MimoAudio
```

初始化时传入：

- `model_path`
- `tokenizer_path`

单文件转写调用：

```python
self.model.asr_sft(wav_path)
```

如果官方 `asr_sft` 签名支持 `audio_tag`，服务会在配置 `MIMO_AUDIO_TAG` 后自动传入该参数。

如果官方代码支持真实 GPU batch 推理，应优先在 `_transcribe_batch_sync` 中实现批量推理，而不是循环调用单文件推理。

## 运行配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MIMO_MODEL_PATH` | `/models/MiMo-V2.5-ASR` | MiMo-V2.5-ASR 权重目录 |
| `MIMO_TOKENIZER_PATH` | `/models/MiMo-Audio-Tokenizer` | MiMo-Audio-Tokenizer 目录 |
| `MIMO_SOURCE_PATH` | 空 | 官方源码路径，需包含 `src/` 包 |
| `MIMO_AUDIO_TAG` | 空 | 可选音频标签参数 |
| `ASR_MAX_BATCH_SIZE` | `16` | 最大动态 batch 大小 |
| `ASR_MAX_WAIT_MS` | `50` | 动态 batch 最大等待毫秒数 |
| `ASR_TMP_DIR` | `/tmp/mimo_asr` | 上传音频临时目录 |

## 部署

容器基于 `nvidia/cuda:12.4.1-runtime-ubuntu22.04`，安装 Python、pip 和 ffmpeg，并通过 uvicorn 启动服务。

默认端口：`8000`

默认模型挂载：

```text
/mnt/data/models/MiMo-V2.5-ASR -> /models/MiMo-V2.5-ASR
```

构建：

```bash
docker build -t mimo-asr-server .
```

运行：

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

## 性能调优

### `max_batch_size`

- A800 80GB：建议 `16` 到 `64`
- H100：可根据显存和模型实现提升到 `128`

### `max_wait_ms`

- `20ms`：低延迟优先
- `50ms`：延迟和吞吐平衡
- `100ms`：吞吐优先

### 进程策略

单 GPU 建议 1 个推理进程，避免多个进程争抢显存。多 GPU 支持需要在进程、端口、调度或模型层面做额外设计。

### 长音频处理

长音频建议先做 VAD 切片，再将切片结果送入批处理队列，通常能提升整体吞吐并降低单请求尾延迟。

## 运行约束

- 运行环境需要 NVIDIA GPU 和可用的容器 GPU runtime。
- 镜像内需要包含官方 MiMo 源码，或挂载源码目录并通过 `MIMO_SOURCE_PATH` 指向包含 `src/` 的路径。
- 运行时需要同时提供 MiMo-V2.5-ASR 权重和 MiMo-Audio-Tokenizer。
- 上传文件会临时写入 `/tmp/mimo_asr`，请求结束后删除。
- Dockerfile 会单独安装 `flash-attn`，因为官方音频 tokenizer 依赖 `flash_attn`，且该包需要在 torch 安装后使用 `--no-build-isolation` 安装。
- 如果官方 MiMo 代码有额外依赖，需要同步补充到 `requirements.txt` 或 Dockerfile。

## 后续增强方向

- 支持 URL 输入。
- 返回句级或词级时间戳。
- 增加 Prometheus metrics。
- 扩展 OpenAI Whisper API 兼容参数。
- 支持多 GPU 和分布式队列。
