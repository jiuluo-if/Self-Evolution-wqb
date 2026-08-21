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
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .failures import FailureKind, classify_error
from .state import Experiment

logger = logging.getLogger("wqb.scheduler")


class BacktestScheduler:
    SCHEMA_VERSION = 2

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
        # Expressions submitted to the backend in a previous run but never
        # finalized (process crashed mid-flight). They must never be submitted
        # again: the backend already owns a simulation for them.
        self._no_retry = set()
        self._since_checkpoint = 0
        self._schema_version = self.SCHEMA_VERSION
        # Worker threads checkpoint() right after a successful backend submit
        # to persist the recoverable progress_url; the lock keeps concurrent
        # checkpoints from interleaving on the tmp file.
        self._checkpoint_lock = threading.Lock()
        self.created_at = time.time()
        self.updated_at = time.time()

    # ---- job intake ----

    def add_jobs(self, experiments):
        """Queue experiments. Experiments whose expression already has a
        recorded outcome (completed, or submitted before a crash) are restored
        instead of being executed again."""
        for exp in experiments:
            if exp.id in self._jobs:
                continue
            if self._restore_prior(exp):
                logger.info(
                    "BACKTEST_JOB_REUSED job_id=%s expression=%s "
                    "(previously handled, not re-run)",
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

    def _failed_by_expression(self, expression):
        for exp in self._failed.values():
            if exp.expression == expression:
                return exp
        return None

    def _restore_prior(self, exp):
        """If this expression was handled in a previous run (completed, or
        submitted to the backend but never finalized before a crash), restore
        the recorded outcome on `exp` and return True."""
        prior = self._completed_by_expression(exp.expression)
        if prior is not None:
            self._restore_result(exp, prior)
            self._completed[exp.id] = exp
            return True
        if exp.expression in self._no_retry:
            prior = self._failed_by_expression(exp.expression)
            if prior is not None:
                self._restore_result(exp, prior)
            else:
                exp.status = "FAILED"
                exp.error = "crash_mid_flight: not re-submitted"
            self._failed[exp.id] = exp
            return True
        return False

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
            # Jobs restored from a checkpoint (submitted/polling, or pending
            # with a reserved budget slot) are already in self._running; kick
            # them off first so they never go through the budget reserve path
            # again.
            for job_id in list(self._running):
                running[pool.submit(self._run_job, job_id)] = job_id
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
        already handled in a previous run and keep the rest."""
        remaining = deque()
        for job_id in self._queue:
            exp = self._jobs[job_id]
            if self._restore_prior(exp):
                logger.info(
                    "BACKTEST_JOB_REUSED job_id=%s expression=%s "
                    "(previously handled, not re-run)",
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
            # Reserve the budget slot for this job. The value survives crashes
            # via the checkpoint and is never re-minted, so a resumed job that
            # already owns a slot is never charged twice.
            self._submitted += 1
            self._running.add(job_id)
            exp.status = "PENDING"
            logger.info(
                "BACKTEST_JOB_STARTED job_id=%s expression=%s",
                job_id,
                exp.expression[:60],
            )
            # Persist the in-flight state BEFORE submitting to the backend so
            # a crash before this point resumes as a plain (re-)submit, and a
            # crash after submit+checkpoint resumes as a poll of the existing
            # simulation.
            self._checkpoint()
            running[pool.submit(self._run_job, job_id)] = job_id

    def _run_job(self, job_id):
        """Worker step for one job: submit if not yet submitted, persist the
        progress_url immediately, then poll to completion. Restored jobs with
        an existing progress_url skip straight to polling.

        Exactly-once submit window: before the POST is issued the worker
        checkpoints the job as SUBMITTING. From that instant until the
        progress_url is persisted, the backend may or may not own a
        simulation, so a crash (or a lost POST response) is resumed as an
        ambiguous outcome and never re-submitted. A job still marked PENDING
        in the checkpoint provably never started its submit, so it is safe to
        re-submit on resume.
        """
        exp = self._jobs[job_id]
        try:
            if exp.status == "PENDING":
                # Durable marker that the submit attempt is starting; after
                # this point a crash leaves an ambiguous backend outcome.
                exp.status = "SUBMITTING"
                self._checkpoint()
                try:
                    self.simulator.submit(exp, poll_timeout_sec=self.poll_timeout_sec)
                except Exception as exc:  # noqa: BLE001
                    if self._submit_outcome_known(exc):
                        raise
                    self._mark_submit_unknown(exp, exc)
                    return exp
                # The backend owns a simulation now; make the progress_url
                # durable before polling so a crash resumes the poll instead
                # of creating a duplicate simulation.
                self._checkpoint()
            if exp.status in ("SUBMITTED", "POLLING"):
                self.simulator.poll(exp, poll_timeout_sec=self.poll_timeout_sec)
        except Exception as exc:  # noqa: BLE001
            exp.error = f"{type(exc).__name__}: {exc}"
            exp.status = "FAILED"
        return exp

    @staticmethod
    def _submit_outcome_known(exc):
        """A confirmed rejection means the backend did NOT accept the
        submission; anything else (timeout, network error, 5xx, missing
        Location, or an unexpected exception) is ambiguous. Typed client
        errors carry a `.kind`; plain exceptions are classified by text."""
        kind = getattr(exc, "kind", None) or classify_error(str(exc))
        return kind in (
            FailureKind.SYNTAX,
            FailureKind.AUTH,
            FailureKind.DATA,
            FailureKind.RATE_LIMIT,
        )

    def _mark_submit_unknown(self, exp, exc):
        """A submit whose backend outcome is unknown: the expression may have
        been accepted, so it must never be submitted again and its budget slot
        stays consumed."""
        exp.status = "SUBMIT_UNKNOWN"
        exp.error = f"submit_unknown: {type(exc).__name__}: {exc}"
        self._no_retry.add(exp.expression)
        logger.warning(
            "BACKTEST_JOB_SUBMIT_UNKNOWN job_id=%s expression=%s error=%s",
            exp.id, exp.expression[:60], exp.error,
        )

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
        with self._checkpoint_lock:
            self.updated_at = time.time()
            data = {
                "schema_version": self.SCHEMA_VERSION,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "submitted": self._submitted,
                "running": [
                    self._jobs[job_id].to_dict() for job_id in self._running
                ],
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
        self._schema_version = data.get("schema_version", 1)
        self._submitted = data.get("submitted", 0)
        # Jobs added by the caller (agent) via add_jobs already carry the
        # fresh Experiment objects; a checkpoint entry for the same expression
        # must be merged into that object so the caller observes the outcome.
        queued = {}
        for job_id in self._queue:
            expr = self._jobs[job_id].expression
            queued.setdefault(expr, job_id)
        for edict in data.get("completed", []):
            exp = Experiment.from_dict(edict)
            if exp.expression in queued:
                # Merge onto the caller's object and drop it from the queue so
                # it is never submitted again.
                target_id = queued[exp.expression]
                target = self._jobs[target_id]
                self._restore_result(target, exp)
                self._completed[target_id] = target
                continue
            self._jobs[exp.id] = exp
            self._completed[exp.id] = exp
        for edict in data.get("failed", []):
            exp = Experiment.from_dict(edict)
            if self._is_no_retry(exp):
                # The backend outcome is unknown or already owned; the
                # expression must never be submitted again. Keep the recorded
                # failure and leave the budget slot consumed.
                self._no_retry.add(exp.expression)
                self._restore_failed(queued, exp)
                continue
            if exp.progress_url:
                # A confirmed submission that failed while polling: resume by
                # polling the existing simulation, never by POSTing again.
                self._restore_failed(queued, exp, resume_poll=True)
                continue
            # A confirmed rejection (syntax/auth/data/rate-limit): keep the
            # recorded failure. Re-queuing is preserved as pre-existing
            # behavior for such rejections.
            self._restore_failed(queued, exp)
        for edict in data.get("running", []):
            exp = Experiment.from_dict(edict)
            if exp.expression in queued:
                target_id = queued[exp.expression]
                target = self._jobs[target_id]
                target.progress_url = exp.progress_url
                target.alpha_id = exp.alpha_id
                target.metrics = exp.metrics
                self._queue.remove(target_id)
                self._restore_running(target_id, target, exp)
            else:
                self._jobs[exp.id] = exp
                self._restore_running(exp.id, exp, exp)
        logger.info(
            "SCHEDULER_RESUMED path=%s schema=%d completed=%d failed=%d "
            "midflight=%d submitted=%d",
            self.checkpoint_path, self._schema_version, len(self._completed),
            len(self._failed), len(self._no_retry), self._submitted,
        )

    def _restore_failed(self, queued, exp, resume_poll=False):
        """Merge a checkpoint FAILED entry back onto the caller's job object
        (or keep it as an orphan) and, for confirmed submissions that failed
        during polling, route it back through _restore_running so the existing
        simulation is polled instead of re-submitted."""
        if exp.expression in queued:
            target_id = queued[exp.expression]
            target = self._jobs[target_id]
            self._restore_result(target, exp)
            if resume_poll:
                target.progress_url = exp.progress_url
                target.alpha_id = exp.alpha_id
                self._queue.remove(target_id)
                self._restore_running(target_id, target, exp)
            else:
                self._failed[target_id] = target
            return
        if resume_poll:
            self._jobs[exp.id] = exp
            self._restore_running(exp.id, exp, exp)
            return
        self._jobs[exp.id] = exp
        self._failed[exp.id] = exp

    @staticmethod
    def _is_no_retry(exp):
        """True when the persisted failure means the expression must never be
        submitted again: an unknown submit outcome, or a previous resume that
        already marked a mid-flight job as failed."""
        error = exp.error or ""
        return (
            exp.status == "SUBMIT_UNKNOWN"
            or "crash_mid_flight" in error
            or "submit_unknown" in error
        )

    def _restore_running(self, job_id, target, snapshot):
        """Route a restored in-flight job by its recoverable state:
        - progress_url present -> continue polling the existing simulation
        - PENDING, no url       -> the worker never started the submit (it
                                   checkpoints SUBMITTING first), so the
                                   backend never received it; re-submit with
                                   no new budget charge
        - SUBMITTING / SUBMIT_UNKNOWN, no url -> the submit may have reached
          the backend; the outcome is unknown, so never re-submit and keep the
          budget slot consumed
        - anything else (legacy RUNNING / v1 PENDING / in-flight submit) ->
          cannot be recovered; fail it and never submit the expression again.
        """
        if snapshot.progress_url:
            target.status = (
                snapshot.status
                if snapshot.status in ("SUBMITTED", "POLLING")
                else "POLLING"
            )
            self._running.add(job_id)
            logger.info(
                "BACKTEST_JOB_RECOVERED job_id=%s expression=%s "
                "resume=poll url=%s",
                job_id, target.expression[:60], snapshot.progress_url,
            )
        elif snapshot.status == "PENDING" and self._schema_version >= 2:
            target.status = "PENDING"
            self._running.add(job_id)
            logger.info(
                "BACKTEST_JOB_RECOVERED job_id=%s expression=%s resume=submit",
                job_id, target.expression[:60],
            )
        elif snapshot.status in ("SUBMITTING", "SUBMIT_UNKNOWN"):
            target.status = "SUBMIT_UNKNOWN"
            target.error = (
                snapshot.error
                or "submit_unknown: crash while submitting; backend outcome unknown"
            )
            target.progress_url = snapshot.progress_url
            self._failed[job_id] = target
            self._no_retry.add(target.expression)
            logger.warning(
                "BACKTEST_JOB_SUBMIT_UNKNOWN job_id=%s expression=%s "
                "marked submit_unknown, not re-submitted",
                job_id, target.expression[:60],
            )
        else:
            target.status = "FAILED"
            target.error = (
                "crash_mid_flight: simulation was submitted but never finalized"
            )
            target.progress_url = snapshot.progress_url
            self._failed[job_id] = target
            self._no_retry.add(target.expression)
            logger.warning(
                "BACKTEST_JOB_MIDFLIGHT job_id=%s expression=%s "
                "marked failed, not re-submitted",
                job_id, target.expression[:60],
            )

    # ---- introspection ----

    def completed_experiments(self):
        return list(self._completed.values())

    def failed_experiments(self):
        return list(self._failed.values())
