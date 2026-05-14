import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestItem:
    wav_path: str
    future: asyncio.Future


class BatchScheduler:
    def __init__(self, model, max_batch_size: int = 16, max_wait_ms: int = 50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue[RequestItem] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, wav_path: str) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(RequestItem(wav_path, future))
        return await future

    async def _run(self):
        while True:
            first = await self.queue.get()
            batch = [first]

            deadline = asyncio.get_running_loop().time() + self.max_wait_ms / 1000.0

            while len(batch) < self.max_batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            wav_paths = [x.wav_path for x in batch]

            try:
                results = await self.model.transcribe_batch(wav_paths)
                for item, text in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(text)
            except Exception as e:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
