"""Backtest scheduling with a strict simulation budget and crash recovery.

Agent -> BacktestScheduler -> Simulator -> WQBClient

The scheduler owns the execution group: concurrency cap, simulation budget,
FIRST_COMPLETED sliding-window refill, pending/started/completed/failed
tracking, structured logging and atomic checkpointing. The Simulator is only
responsible for a single submit -> poll -> fetch cycle.
"""

import json
import logging
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .state import Experiment

logger = logging.getLogger("wqb.scheduler")


class BacktestScheduler:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        client,
        simulator,
        *,
        max_concurrent=3,
        budget=None,
        poll_timeout_sec=900,
        checkpoint_path=None,
        checkpoint_every=1,
    ):
        self.client = client
        self.simulator = simulator
        self.max_concurrent = max_concurrent
        self.budget = budget  # None == unlimited
        self.poll_timeout_sec = poll_timeout_sec
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every

        self._jobs = {}  # job_id -> Experiment
        self._queue = deque()  # pending job_ids (FIFO)
        self._running = set()  # job_ids currently in flight
        self._completed = {}  # job_id -> Experiment (DONE)
        self._failed = {}  # job_id -> Experiment (FAILED)
        self._submitted = 0
        self._since_checkpoint = 0
        self.created_at = time.time()
        self.updated_at = time.time()

    # ---- job intake ----

    def add_jobs(self, experiments):
        """Queue experiments. Experiments whose expression is already recorded
        as completed in a checkpoint are restored instead of re-executed."""
        for exp in experiments:
            if exp.id in self._jobs:
                continue
            prior = self._completed_by_expression(exp.expression)
            if prior is not None:
                self._restore_result(exp, prior)
                self._completed[exp.id] = exp
                logger.info(
                    "BACKTEST_JOB_REUSED job_id=%s expression=%s "
                    "(previously completed, not re-run)",
                    exp.id,
                    exp.expression[:60],
                )
                continue
            self._jobs[exp.id] = exp
            if exp.id in self._failed:
                continue
            self._queue.append(exp.id)

    def _completed_by_expression(self, expression):
        for exp in self._completed.values():
            if exp.expression == expression:
                return exp
        return None

    @staticmethod
    def _restore_result(target, prior):
        target.status = prior.status
        target.metrics = prior.metrics
        target.error = prior.error
        target.alpha_id = prior.alpha_id

    # ---- execution ----

    def run(self):
        self._resume()
        self._drop_completed_from_queue()
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            running = {}
            self._fill_pipeline(pool, running)
            while self._queue or running:
                if not running:
                    break
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in done:
                    job_id = running.pop(fut)
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        exp = self._jobs[job_id]
                        exp.status = "FAILED"
                        exp.error = f"{type(exc).__name__}: {exc}"
                    self._finalize(job_id)
                self._fill_pipeline(pool, running)

        # Anything still queued ran out of simulation budget.
        for job_id in self._queue:
            self._jobs[job_id].status = "SKIPPED"
        self._queue.clear()
        self._checkpoint()
        self.updated_at = time.time()
        return list(self._jobs.values())

    def _drop_completed_from_queue(self):
        """After resume, restore results for queued jobs whose expression was
        already completed and keep the rest."""
        remaining = deque()
        for job_id in self._queue:
            exp = self._jobs[job_id]
            prior = self._completed_by_expression(exp.expression)
            if prior is not None:
                self._restore_result(exp, prior)
                self._completed[job_id] = exp
                logger.info(
                    "BACKTEST_JOB_REUSED job_id=%s expression=%s "
                    "(previously completed, not re-run)",
                    job_id,
                    exp.expression[:60],
                )
                continue
            remaining.append(job_id)
        self._queue = remaining

    def _budget_left(self):
        if self.budget is None:
            return None
        return self.budget - self._submitted

    def _fill_pipeline(self, pool, running):
        while self._queue:
            left = self._budget_left()
            if left is not None and left <= 0:
                break
            if len(running) >= self.max_concurrent:
                break
            job_id = self._queue.popleft()
            exp = self._jobs[job_id]
            self._submitted += 1
            self._running.add(job_id)
            logger.info(
                "BACKTEST_JOB_STARTED job_id=%s expression=%s",
                job_id,
                exp.expression[:60],
            )
            running[pool.submit(self._simulate_job, job_id)] = job_id

    def _simulate_job(self, job_id):
        exp = self._jobs[job_id]
        self.simulator.simulate(exp, poll_timeout_sec=self.poll_timeout_sec)
        return exp

    def _finalize(self, job_id):
        exp = self._jobs[job_id]
        self._running.discard(job_id)
        if exp.status == "DONE":
            self._completed[job_id] = exp
            logger.info(
                "BACKTEST_JOB_COMPLETED job_id=%s expression=%s sharpe=%s fitness=%s",
                job_id,
                exp.expression[:60],
                (exp.metrics or {}).get("sharpe"),
                (exp.metrics or {}).get("fitness"),
            )
        else:
            self._failed[job_id] = exp
            logger.warning(
                "BACKTEST_JOB_FAILED job_id=%s expression=%s error=%s",
                job_id,
                exp.expression[:60],
                exp.error,
            )
        self._since_checkpoint += 1
        if self.checkpoint_every and self._since_checkpoint >= self.checkpoint_every:
            self._checkpoint()
            self._since_checkpoint = 0

    # ---- checkpoint / resume ----

    def _checkpoint(self):
        if not self.checkpoint_path:
            return
        self.updated_at = time.time()
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "submitted": self._submitted,
            "completed": [
                exp.to_dict() for exp in self._completed.values()
            ],
            "failed": [exp.to_dict() for exp in self._failed.values()],
        }
        tmp = self.checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.checkpoint_path)
        logger.info("CHECKPOINT_SAVED path=%s completed=%d failed=%d",
                    self.checkpoint_path, len(self._completed),
                    len(self._failed))

    def _resume(self):
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return
        with open(self.checkpoint_path) as f:
            data = json.load(f)
        self._submitted = data.get("submitted", 0)
        for edict in data.get("completed", []):
            exp = Experiment.from_dict(edict)
            self._jobs[exp.id] = exp
            self._completed[exp.id] = exp
        for edict in data.get("failed", []):
            exp = Experiment.from_dict(edict)
            self._jobs[exp.id] = exp
            self._failed[exp.id] = exp
        logger.info(
            "SCHEDULER_RESUMED path=%s completed=%d failed=%d submitted=%d",
            self.checkpoint_path, len(self._completed), len(self._failed),
            self._submitted,
        )

    # ---- introspection ----

    def completed_experiments(self):
        return list(self._completed.values())

    def failed_experiments(self):
        return list(self._failed.values())
