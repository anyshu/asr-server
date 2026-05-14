import asyncio
from concurrent.futures import ThreadPoolExecutor


class MimoASRModel:
    def __init__(self, model_path: str):
        # Replace with official MiMo model loading logic.
        self.model_path = model_path
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _transcribe_one(self, wav_path: str) -> str:
        # Replace with:
        # return self.model.asr_sft(wav_path)
        return f"transcribed text for {wav_path}"

    def _transcribe_batch_sync(self, wav_paths: list[str]) -> list[str]:
        return [self._transcribe_one(p) for p in wav_paths]

    async def transcribe_batch(self, wav_paths: list[str]) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.pool,
            self._transcribe_batch_sync,
            wav_paths,
        )
