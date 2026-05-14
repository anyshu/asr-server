import os
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from model import MimoASRModel
from batcher import BatchScheduler

TMP_DIR = "/tmp/mimo_asr"
os.makedirs(TMP_DIR, exist_ok=True)

model = MimoASRModel("/models/MiMo-V2.5-ASR")
scheduler = BatchScheduler(model=model, max_batch_size=16, max_wait_ms=50)


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
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio.wav")[-1]
    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}{suffix}")

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        text = await scheduler.submit(tmp_path)
        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
