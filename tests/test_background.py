import asyncio
import unittest

from ai_agent.background import BackgroundTaskManager


class BackgroundTaskManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_job_records_output_and_notifies(self) -> None:
        finished = asyncio.Event()

        async def on_finish(job: object) -> None:
            finished.set()

        async def operation() -> str:
            return "result"

        manager = BackgroundTaskManager(on_finish)
        job = manager.start("test job", operation)
        await finished.wait()

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.output, "result")
        self.assertIsNotNone(job.duration_ms)

    async def test_running_job_can_be_cancelled(self) -> None:
        started = asyncio.Event()

        async def operation() -> str:
            started.set()
            await asyncio.Event().wait()
            return "unreachable"

        manager = BackgroundTaskManager()
        job = manager.start("long job", operation)
        await started.wait()
        manager.cancel(job.id)
        await job.task

        self.assertEqual(job.status, "cancelled")

    async def test_failed_job_records_error_and_notifies(self) -> None:
        finished = asyncio.Event()

        async def on_finish(_job: object) -> None:
            finished.set()

        async def operation() -> str:
            raise RuntimeError("command failed")

        manager = BackgroundTaskManager(on_finish)
        job = manager.start("failing job", operation)
        await finished.wait()
        await job.task

        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "command failed")
        self.assertIsNotNone(job.duration_ms)

    async def test_cancel_rejects_unknown_and_finished_jobs(self) -> None:
        async def operation() -> str:
            return "done"

        manager = BackgroundTaskManager()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            manager.cancel(99)
        job = manager.start("quick job", operation)
        await job.task

        with self.assertRaisesRegex(ValueError, "already completed"):
            manager.cancel(job.id)

    async def test_shutdown_cancels_every_running_job(self) -> None:
        started = [asyncio.Event(), asyncio.Event()]

        def operation(index: int):
            async def run() -> str:
                started[index].set()
                await asyncio.Event().wait()
                return "unreachable"

            return run

        manager = BackgroundTaskManager()
        jobs = [
            manager.start("first", operation(0)),
            manager.start("second", operation(1)),
        ]
        await asyncio.gather(*(event.wait() for event in started))

        await manager.shutdown()

        self.assertEqual([job.status for job in jobs], ["cancelled", "cancelled"])


if __name__ == "__main__":
    unittest.main()
