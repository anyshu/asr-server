import asyncio
import inspect
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class MimoASRModel:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = "/models/MiMo-Audio-Tokenizer",
        audio_tag: str | None = None,
        source_path: str | None = None,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.audio_tag = audio_tag
        self.pool = ThreadPoolExecutor(max_workers=1)

        self._add_source_path(source_path or os.getenv("MIMO_SOURCE_PATH"))
        self.model = self._load_model()
        self._asr_signature = inspect.signature(self.model.asr_sft)

    def _add_source_path(self, source_path: str | None) -> None:
        if source_path and source_path not in sys.path:
            sys.path.insert(0, source_path)

    def _load_model(self) -> Any:
        try:
            from src.mimo_audio.mimo_audio import MimoAudio
        except ImportError as exc:
            raise RuntimeError(
                "MiMo official source code is not available. Install the XiaomiMiMo/MiMo-V2.5-ASR "
                "repository in the image, or mount it and set MIMO_SOURCE_PATH to the directory that "
                "contains the src/ package."
            ) from exc

        try:
            return MimoAudio(
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
            )
        except TypeError:
            # Keep compatibility if the official constructor changes names.
            return MimoAudio(self.model_path, self.tokenizer_path)

    def _transcribe_one(self, wav_path: str) -> str:
        kwargs = {}
        if self.audio_tag and "audio_tag" in self._asr_signature.parameters:
            kwargs["audio_tag"] = self.audio_tag

        result = self.model.asr_sft(wav_path, **kwargs)
        return self._normalize_result(result)

    def _normalize_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("text", "transcription", "result"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
        return str(result)

    def _transcribe_batch_sync(self, wav_paths: list[str]) -> list[str]:
        return [self._transcribe_one(p) for p in wav_paths]

    async def transcribe_batch(self, wav_paths: list[str]) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.pool,
            self._transcribe_batch_sync,
            wav_paths,
        )
