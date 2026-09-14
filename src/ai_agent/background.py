import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic


JobFactory = Callable[[], Awaitable[str]]
CompletionHandler = Callable[["BackgroundJob"], Awaitable[None]]


@dataclass(slots=True)
class BackgroundJob:
    """State for one asynchronous command or model request."""

    id: int
    title: str
    kind: str = "task"
    status: str = "running"
    output: str = ""
    error: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class BackgroundTaskManager:
    """Run, track, and cancel bounded asynchronous work."""

    def __init__(self, on_finish: CompletionHandler | None = None) -> None:
        self._on_finish = on_finish
        self._jobs: dict[int, BackgroundJob] = {}
        self._next_id = 1

    def start(
        self,
        title: str,
        factory: JobFactory,
        *,
        kind: str = "task",
    ) -> BackgroundJob:
        job = BackgroundJob(id=self._next_id, title=title, kind=kind)
        self._next_id += 1
        self._jobs[job.id] = job
        job.task = asyncio.create_task(self._run(job, factory))
        return job

    def list(self) -> list[BackgroundJob]:
        return list(sorted(self._jobs.values(), key=lambda job: job.id))

    def cancel(self, job_id: int) -> BackgroundJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"Background job {job_id} does not exist")
        if job.status != "running" or job.task is None:
            raise ValueError(f"Background job {job_id} is already {job.status}")
        job.task.cancel()
        return job

    async def shutdown(self) -> None:
        tasks = []
        for job in self._jobs.values():
            if job.status == "running" and job.task is not None:
                job.task.cancel()
                tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, job: BackgroundJob, factory: JobFactory) -> None:
        started = monotonic()
        try:
            job.output = await factory()
            job.status = "completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
        except Exception as error:
            job.status = "failed"
            job.error = str(error)
        finally:
            job.duration_ms = round((monotonic() - started) * 1000)
            if self._on_finish is not None:
                await self._on_finish(job)
