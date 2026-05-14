import os
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from batcher import BatchScheduler
from model import MimoASRModel

TMP_DIR = os.getenv("ASR_TMP_DIR", "/tmp/mimo_asr")
MODEL_PATH = os.getenv("MIMO_MODEL_PATH", "/models/MiMo-V2.5-ASR")
TOKENIZER_PATH = os.getenv("MIMO_TOKENIZER_PATH", "/models/MiMo-Audio-Tokenizer")
SOURCE_PATH = os.getenv("MIMO_SOURCE_PATH")
AUDIO_TAG = os.getenv("MIMO_AUDIO_TAG") or None
MAX_BATCH_SIZE = int(os.getenv("ASR_MAX_BATCH_SIZE", "16"))
MAX_WAIT_MS = int(os.getenv("ASR_MAX_WAIT_MS", "50"))

os.makedirs(TMP_DIR, exist_ok=True)

model = MimoASRModel(
    MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    audio_tag=AUDIO_TAG,
    source_path=SOURCE_PATH,
)
scheduler = BatchScheduler(model=model, max_batch_size=MAX_BATCH_SIZE, max_wait_ms=MAX_WAIT_MS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await scheduler.start()
    yield
    await scheduler.stop()


app = FastAPI(lifespan=lifespan)


class TranscriptionResponse(BaseModel):
    text: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "tokenizer_path": TOKENIZER_PATH,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_wait_ms": MAX_WAIT_MS,
    }


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    model_name: str | None = Form(default=None, alias="model"),
):
    suffix = os.path.splitext(file.filename or "audio.wav")[-1]
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}{suffix}")

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        text = await scheduler.submit(tmp_path)
        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
