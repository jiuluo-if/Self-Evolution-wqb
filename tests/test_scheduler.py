import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wqb_agent.client import WQBSimulationError
from wqb_agent.scheduler import BacktestScheduler
from wqb_agent.simulator import Simulator
from wqb_agent.state import Experiment


class FakeClient:
    def __init__(self, latency=0.02):
        self.latency = latency
        self.counter = 0
        self.submissions = []
        self._expr_by_url = {}
        self._alpha_expr = {}
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def submit_simulation(self, expression, settings, alpha_type="REGULAR"):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(self.latency)
            self.counter += 1
            url = f"progress-{self.counter}"
            self._expr_by_url[url] = expression
            self.submissions.append(expression)
            return url
        finally:
            with self._lock:
                self._active -= 1

    def poll_progress(self, progress_url, timeout_sec=900):
        time.sleep(self.latency)
        expression = self._expr_by_url[progress_url]
        alpha_id = f"alpha-{abs(hash(progress_url))}"
        self._alpha_expr[alpha_id] = expression
        return alpha_id

    def get_alpha(self, alpha_id):
        expression = self._alpha_expr.get(alpha_id, "")
        return {
            "is": {
                "sharpe": 1.0,
                "fitness": 1.0,
                "turnover": 0.5,
                "margin": 0.1,
                "returns": 0.05,
                "checks": [{"name": "limitations", "pass": True}],
            },
            "regular": expression,
        }


def make_experiments(exprs, round_no=1):
    return [Experiment(round_no, "h", e, {}, []) for e in exprs]


class TestSchedulerConcurrency(unittest.TestCase):
    def test_concurrency_never_exceeds_limit(self):
        client = FakeClient()
        sim = Simulator(client)
        scheduler = BacktestScheduler(client, sim, max_concurrent=3)
        jobs = make_experiments([f"rank(field{i})" for i in range(7)])
        scheduler.add_jobs(jobs)
        scheduler.run()
        self.assertLessEqual(client.max_active, 3)
        self.assertTrue(all(j.status == "DONE" for j in jobs))
        self.assertEqual(client.counter, 7)

    def test_fill_after_first_completed_refills_pipeline(self):
        # More jobs than workers: completion must immediately refill so every
        # job finishes and concurrency never spikes above the cap.
        client = FakeClient(latency=0.03)
        sim = Simulator(client)
        scheduler = BacktestScheduler(client, sim, max_concurrent=2)
        jobs = make_experiments([f"rank(f{i})" for i in range(9)])
        scheduler.add_jobs(jobs)
        scheduler.run()
        self.assertEqual(client.counter, 9)
        self.assertEqual(sum(1 for j in jobs if j.status == "DONE"), 9)
        self.assertLessEqual(client.max_active, 2)


class TestSchedulerBudget(unittest.TestCase):
    def test_budget_limits_submissions(self):
        client = FakeClient()
        sim = Simulator(client)
        scheduler = BacktestScheduler(client, sim, max_concurrent=3, budget=3)
        jobs = make_experiments([f"rank(f{i})" for i in range(6)])
        scheduler.add_jobs(jobs)
        scheduler.run()
        self.assertEqual(client.counter, 3)
        self.assertEqual(sum(1 for j in jobs if j.status == "DONE"), 3)
        self.assertEqual(sum(1 for j in jobs if j.status == "SKIPPED"), 3)


class TestSchedulerCheckpoint(unittest.TestCase):
    def _path(self, tmpdir):
        return os.path.join(tmpdir, "jobs.json")

    def test_checkpoint_saved_atomically(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        client = FakeClient()
        sim = Simulator(client)
        scheduler = BacktestScheduler(
            client, sim, max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        scheduler.add_jobs(make_experiments([f"rank(f{i})" for i in range(4)]))
        scheduler.run()
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path + ".tmp"))

    def test_checkpoint_is_compact_machine_json(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        client = FakeClient()
        scheduler = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        scheduler.add_jobs(make_experiments([f"rank(f{i})" for i in range(3)]))
        scheduler.run()
        # A checkpoint is a resume artifact, not a document: it must round-trip
        # through json.load and stay compact (no pretty-print indentation).
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data["completed"]), 3)
        with open(path) as f:
            self.assertNotIn("\n  \"completed\"", f.read())

    def test_resume_does_not_rerun_completed(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        exprs = [f"rank(f{i})" for i in range(6)]

        client1 = FakeClient()
        sched1 = BacktestScheduler(
            client1, Simulator(client1), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        sched1.add_jobs(make_experiments(exprs))
        sched1.run()
        self.assertEqual(client1.counter, 6)

        # Simulated crash/restart: fresh scheduler over the same checkpoint,
        # brand-new Experiment objects with the same expressions.
        client2 = FakeClient()
        sched2 = BacktestScheduler(
            client2, Simulator(client2), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs2 = make_experiments(exprs)
        sched2.add_jobs(jobs2)
        sched2.run()
        self.assertEqual(client2.counter, 0, "completed jobs must not re-run")
        self.assertTrue(all(j.status == "DONE" for j in jobs2))
        self.assertEqual([j.metrics["sharpe"] for j in jobs2],
                         [1.0] * 6)

    def test_resume_continues_incomplete_pipeline(self):
        # Jobs never submitted before the crash still run after resume, while
        # the previously-spent simulation budget stays consumed.
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        exprs = [f"rank(f{i})" for i in range(6)]

        client1 = FakeClient()
        sched1 = BacktestScheduler(
            client1, Simulator(client1), max_concurrent=4, budget=4,
            checkpoint_path=path, checkpoint_every=1,
        )
        sched1.add_jobs(make_experiments(exprs))
        sched1.run()
        done1 = {j.expression for j in sched1.completed_experiments()}

        client2 = FakeClient()
        sched2 = BacktestScheduler(
            client2, Simulator(client2), max_concurrent=4, budget=6,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs2 = make_experiments(exprs)
        sched2.add_jobs(jobs2)
        sched2.run()
        self.assertEqual(client2.counter, len(exprs) - len(done1))
        self.assertTrue(all(j.status == "DONE" for j in jobs2))

    def test_completed_job_deduplicated_by_expression(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"

        client1 = FakeClient()
        sched1 = BacktestScheduler(
            client1, Simulator(client1), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        sched1.add_jobs(make_experiments([expr, "rank(f1)"]))
        sched1.run()

        client2 = FakeClient()
        sched2 = BacktestScheduler(
            client2, Simulator(client2), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        sched2.add_jobs(make_experiments([expr, "rank(f2)", "rank(f3)"]))
        sched2.run()
        self.assertEqual(client2.counter, 2, "only the two new expressions run")


class TestSchedulerMidFlight(unittest.TestCase):
    """A crash after submit but before finalize must not re-submit the same
    expression (exactly-once at the backend level)."""

    def _path(self, tmpdir):
        return os.path.join(tmpdir, "jobs.json")

    def _make_checkpoint(self, tmpdir, done_expr, running_expr):
        """Hand-craft a checkpoint as it looks after a mid-flight crash:
        one completed job plus one job submitted but never finalized."""
        import json

        done = make_experiments([done_expr])
        done[0].status = "DONE"
        done[0].metrics = {"sharpe": 1.0, "fitness": 1.0}
        running = make_experiments([running_expr])
        running[0].status = "RUNNING"

        data = {
            "schema_version": 1,
            "submitted": 2,
            "running": [running[0].to_dict()],
            "completed": [done[0].to_dict()],
            "failed": [],
        }
        path = self._path(tmpdir)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_midflight_crash_is_not_resubmitted(self):
        tmpdir = tempfile.mkdtemp()
        done_expr, crashed_expr, fresh_expr = (
            "rank(f0)", "rank(f1)", "rank(f2)")
        path = self._make_checkpoint(tmpdir, done_expr, crashed_expr)

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([done_expr, crashed_expr, fresh_expr])
        sched.add_jobs(jobs)
        sched.run()

        self.assertEqual(
            client.counter, 1,
            "only the never-submitted expression may run",
        )
        self.assertEqual(client.submissions, [fresh_expr])
        by_expr = {j.expression: j for j in jobs}
        self.assertEqual(by_expr[done_expr].status, "DONE")
        self.assertEqual(by_expr[done_expr].metrics["sharpe"], 1.0)
        self.assertEqual(by_expr[crashed_expr].status, "FAILED")
        self.assertIn("crash_mid_flight", by_expr[crashed_expr].error)
        self.assertEqual(by_expr[fresh_expr].status, "DONE")

    def test_midflight_consumed_budget_survives_resume(self):
        # The crash consumed budget on the backend; resuming must not mint
        # that budget back, or the cap would be silently exceeded.
        tmpdir = tempfile.mkdtemp()
        done_expr, crashed_expr = "rank(f0)", "rank(f1)"
        path = self._make_checkpoint(tmpdir, done_expr, crashed_expr)

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=2,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([done_expr, crashed_expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.counter, 0, "nothing left under the budget")

    def test_double_crash_keeps_midflight_out_of_submission(self):
        # A mid-flight job turned FAILED on the first resume, then the process
        # crashed again: the second resume must still refuse to re-submit it.
        import json

        tmpdir = tempfile.mkdtemp()
        crashed_expr = "rank(f1)"
        done = make_experiments(["rank(f0)"])
        done[0].status = "DONE"
        done[0].metrics = {"sharpe": 1.0}
        failed = make_experiments([crashed_expr])
        failed[0].status = "FAILED"
        failed[0].error = "crash_mid_flight: simulation was submitted but never finalized"
        data = {
            "schema_version": 1,
            "submitted": 2,
            "running": [],
            "completed": [done[0].to_dict()],
            "failed": [failed[0].to_dict()],
        }
        path = self._path(tmpdir)
        with open(path, "w") as f:
            json.dump(data, f)

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([crashed_expr, "rank(f9)"])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.submissions, ["rank(f9)"])
        self.assertEqual(
            {j.expression: j.status for j in jobs},
            {crashed_expr: "FAILED", "rank(f9)": "DONE"},
        )

    def test_midflight_state_persisted_before_submit(self):
        # The in-flight job must be in the checkpoint at the moment the backend
        # submit call executes. The worker persists the job as SUBMITTING (no
        # progress_url yet) BEFORE issuing the POST, so a crash during or after
        # the POST is unambiguous: a resume must never re-submit it.
        class BlockingClient(FakeClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.entered = threading.Event()
                self.release = threading.Event()

            def submit_simulation(self, expression, settings, alpha_type="REGULAR"):
                self.entered.set()
                self.release.wait(timeout=5)
                return super().submit_simulation(
                    expression, settings, alpha_type=alpha_type)

        import json

        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        client = BlockingClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=1, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments(["rank(f0)"])
        sched.add_jobs(jobs)

        worker = threading.Thread(target=sched.run)
        worker.start()
        self.assertTrue(client.entered.wait(timeout=5))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data.get("running", [])), 1)
        self.assertEqual(data["running"][0]["expression"], "rank(f0)")
        self.assertEqual(data["running"][0]["status"], "SUBMITTING")
        self.assertIsNone(data["running"][0].get("progress_url"))
        self.assertEqual(data["submitted"], 1)

        client.release.set()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(jobs[0].status, "DONE")
        self.assertEqual(client.submissions, ["rank(f0)"])


class TestExactlyOnceRecovery(unittest.TestCase):
    """The submit/resume invariant: once the backend owns a simulation, a
    crash must resume by polling the existing simulation (identified by the
    persisted progress_url), never by POSTing /simulations again. budget
    `submitted` is never re-minted across restarts."""

    def _path(self, tmpdir):
        return os.path.join(tmpdir, "jobs.json")

    def _write_checkpoint(self, path, submitted, running=None,
                          completed=None, failed=None, schema_version=2):
        import json

        data = {
            "schema_version": schema_version,
            "submitted": submitted,
            "running": [e.to_dict() for e in (running or [])],
            "completed": [e.to_dict() for e in (completed or [])],
            "failed": [e.to_dict() for e in (failed or [])],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def _submitted_exp(self, expr, url):
        exp = make_experiments([expr])[0]
        exp.status = "SUBMITTED"
        exp.progress_url = url
        return exp

    def _polling_exp(self, expr, url):
        exp = self._submitted_exp(expr, url)
        exp.status = "POLLING"
        return exp

    def _pending_exp(self, expr):
        exp = make_experiments([expr])[0]
        exp.status = "PENDING"
        return exp

    def _done_exp(self, expr):
        exp = make_experiments([expr])[0]
        exp.status = "DONE"
        exp.metrics = {"sharpe": 1.0, "fitness": 1.0}
        return exp

    def _seed_url(self, client, expr, url):
        client._expr_by_url[url] = expr

    def test_crash_before_submit_allows_resubmit(self):
        # A PENDING slot with no progress_url means the backend never received
        # a submit: restart must submit it exactly once (no duplicate budget
        # charge, no double submission).
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        self._write_checkpoint(path, submitted=1, running=[self._pending_exp(expr)])

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.submissions, [expr], "exactly one submit")
        self.assertEqual(jobs[0].status, "DONE")
        self.assertEqual(sched._submitted, 1, "slot charged once")

    def test_submitted_job_resumes_with_poll_not_resubmit(self):
        # Submit succeeded and the progress_url was persisted; crash. Restart
        # must poll the existing simulation: zero POST /simulations.
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        url = "progress-7"
        self._seed_url(FakeClient(), expr, url)
        self._write_checkpoint(path, submitted=1,
                               running=[self._submitted_exp(expr, url)])

        client = FakeClient()
        self._seed_url(client, expr, url)
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.submissions, [], "no re-submit after crash")
        self.assertEqual(client.counter, 0)
        self.assertEqual(jobs[0].status, "DONE")
        self.assertEqual(jobs[0].progress_url, url)
        self.assertEqual(jobs[0].metrics["sharpe"], 1.0)

    def test_polling_job_resumes_with_poll(self):
        # Crashed mid-poll with a persisted progress_url: keep polling.
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        url = "progress-3"
        self._write_checkpoint(path, submitted=1,
                               running=[self._polling_exp(expr, url)])

        client = FakeClient()
        self._seed_url(client, expr, url)
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.counter, 0, "poll only, no submit")
        self.assertEqual(jobs[0].status, "DONE")
        self.assertIsNotNone(jobs[0].metrics)

    def test_done_job_never_resubmitted(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        self._write_checkpoint(path, submitted=1,
                               completed=[self._done_exp(expr)])

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.counter, 0)
        self.assertEqual(jobs[0].status, "DONE")

    def test_partial_concurrency_recovers_mixed_states(self):
        # Some jobs finished, one was submitted, one only reserved a slot:
        # restart polls the submitted one, re-submits the reserved one, runs
        # the untouched ones, and never re-runs the finished ones.
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        done_expr, sub_expr, pend_expr, fresh1, fresh2 = (
            "rank(f0)", "rank(f1)", "rank(f2)", "rank(f3)", "rank(f4)")
        url = "progress-9"
        self._write_checkpoint(
            path,
            submitted=4,
            running=[self._submitted_exp(sub_expr, url),
                     self._pending_exp(pend_expr)],
            completed=[self._done_exp(done_expr)],
        )

        client = FakeClient()
        self._seed_url(client, sub_expr, url)
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=4, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([done_expr, sub_expr, pend_expr, fresh1, fresh2])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(
            client.submissions,
            [pend_expr, fresh1, fresh2],
            "submitted job is polled, never re-submitted",
        )
        by_expr = {j.expression: j.status for j in jobs}
        self.assertEqual(by_expr[done_expr], "DONE")
        self.assertEqual(by_expr[sub_expr], "DONE")
        self.assertEqual(by_expr[pend_expr], "DONE")
        self.assertEqual(by_expr[fresh1], "DONE")
        self.assertEqual(by_expr[fresh2], "DONE")

    def test_budget_never_exceeded_across_crash(self):
        # budget=4, 3 slots already consumed before the crash. Restart may
        # submit at most one more job, even though a fresh submission + the
        # polled job both complete.
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        sub_expr, fresh = "rank(f0)", "rank(f1)"
        url = "progress-5"
        self._write_checkpoint(
            path,
            submitted=3,
            running=[self._submitted_exp(sub_expr, url)],
            completed=[self._done_exp("rank(f9)"), self._done_exp("rank(f8)")],
        )

        client = FakeClient()
        self._seed_url(client, sub_expr, url)
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=4, budget=4,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([sub_expr, fresh, "rank(f2)", "rank(f3)"])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.counter, 1, "one fresh slot left")
        self.assertEqual(client.submissions, [fresh])
        by_expr = {j.expression: j.status for j in jobs}
        self.assertEqual(by_expr[sub_expr], "DONE", "polled, not re-submitted")
        self.assertEqual(by_expr[fresh], "DONE")
        self.assertEqual(by_expr["rank(f2)"], "SKIPPED")
        self.assertEqual(by_expr["rank(f3)"], "SKIPPED")
        self.assertLessEqual(sched._submitted, 4)

    def test_progress_url_persisted_before_poll(self):
        # Real window: submit returns, then the scheduler must checkpoint the
        # progress_url before polling. While the worker is blocked inside
        # poll_progress the checkpoint already carries the recoverable URL.
        import json

        class PollBlockingClient(FakeClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.poll_entered = threading.Event()
                self.release = threading.Event()

            def poll_progress(self, progress_url, timeout_sec=900):
                self.poll_entered.set()
                self.release.wait(timeout=5)
                return super().poll_progress(progress_url, timeout_sec=timeout_sec)

        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        client = PollBlockingClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=1, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments(["rank(f0)"])
        sched.add_jobs(jobs)

        worker = threading.Thread(target=sched.run)
        worker.start()
        self.assertTrue(client.poll_entered.wait(timeout=5))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data.get("running", [])), 1)
        self.assertEqual(data["running"][0]["status"], "SUBMITTED")
        self.assertEqual(data["running"][0]["progress_url"], "progress-1")
        self.assertEqual(data["submitted"], 1)

        client.release.set()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(jobs[0].status, "DONE")
        self.assertEqual(client.counter, 1, "exactly one submit")

    def test_validation_job_resumes_with_poll(self):
        # Validation simulations ride the same scheduler state machine, so a
        # validation job that crashed after submit must also be resumed by
        # polling rather than re-submitted.
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(ts_mean(close, 10))"
        url = "progress-21"
        validation_job = self._submitted_exp(expr, url)
        validation_job.mutation = "validation-window-up"
        self._write_checkpoint(path, submitted=1, running=[validation_job])

        client = FakeClient()
        self._seed_url(client, expr, url)
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        jobs[0].mutation = "validation-window-up"
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.counter, 0)
        self.assertEqual(jobs[0].status, "DONE")
        self.assertEqual(jobs[0].mutation, "validation-window-up")


class TestSubmitCrashWindow(unittest.TestCase):
    """The exactly-once submit window: from the moment the worker persists
    SUBMITTING (immediately before the POST) until the progress_url is
    checkpointed, a crash (or a lost POST response) leaves the backend outcome
    unknown. Those jobs must never be re-submitted and their budget slot must
    stay consumed."""

    def _path(self, tmpdir):
        return os.path.join(tmpdir, "jobs.json")

    def _write(self, path, submitted, running=None, completed=None, failed=None):
        import json

        data = {
            "schema_version": 2,
            "submitted": submitted,
            "running": [e.to_dict() for e in (running or [])],
            "completed": [e.to_dict() for e in (completed or [])],
            "failed": [e.to_dict() for e in (failed or [])],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def _pending(self, expr):
        exp = make_experiments([expr])[0]
        exp.status = "PENDING"
        return exp

    def _submitting(self, expr):
        exp = make_experiments([expr])[0]
        exp.status = "SUBMITTING"
        return exp

    def _submitted(self, expr, url):
        exp = make_experiments([expr])[0]
        exp.status = "SUBMITTED"
        exp.progress_url = url
        return exp

    def _done(self, expr):
        exp = make_experiments([expr])[0]
        exp.status = "DONE"
        exp.metrics = {"sharpe": 1.0, "fitness": 1.0}
        return exp

    # Scenario 1: crash before POST /simulations is issued. The worker had
    # not yet persisted SUBMITTING, so the backend provably never received the
    # submit; restart re-submits it exactly once.
    def test_crash_before_post_resubmits_exactly_once(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        self._write(path, submitted=1, running=[self._pending(expr)])

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.submissions, [expr], "exactly one submit")
        self.assertEqual(jobs[0].status, "DONE")
        self.assertEqual(sched._submitted, 1, "no second budget charge")

    # Scenario 2: crash during the POST. The durable state is SUBMITTING with
    # no progress_url; the backend may own a simulation, so resume must not
    # re-submit and the slot stays consumed.
    def test_crash_during_post_not_resubmitted(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        self._write(path, submitted=1, running=[self._submitting(expr)])

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.submissions, [], "ambiguous submit is never re-submitted")
        self.assertEqual(client.counter, 0)
        self.assertEqual(jobs[0].status, "SUBMIT_UNKNOWN")
        self.assertEqual(sched._submitted, 1, "budget slot stays consumed")

    # Scenario 3: POST accepted by BRAIN but the response is lost. Within one
    # process the ambiguous outcome raises immediately (no retry), and a
    # restart keeps refusing to re-submit.
    def test_response_lost_after_accept_is_submit_unknown_not_retried(self):
        class ResponseLostClient(FakeClient):
            def submit_simulation(self, expression, settings, alpha_type="REGULAR"):
                # Backend records the simulation, then the response is lost.
                self.counter += 1
                url = f"progress-{self.counter}"
                self._expr_by_url[url] = expression
                self.submissions.append(expression)
                raise WQBSimulationError("connection reset after response")

        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        client = ResponseLostClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=1, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.counter, 1, "exactly one POST, no ambiguous retry")
        self.assertEqual(jobs[0].status, "SUBMIT_UNKNOWN")
        self.assertIn("submit_unknown", jobs[0].error)

        # Second crash/resume over the same checkpoint: still no re-submit.
        client2 = FakeClient()
        sched2 = BacktestScheduler(
            client2, Simulator(client2), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs2 = make_experiments([expr])
        sched2.add_jobs(jobs2)
        sched2.run()
        self.assertEqual(client2.submissions, [], "still no re-submit after restart")
        self.assertEqual(jobs2[0].status, "SUBMIT_UNKNOWN")

    # Scenario 4: the submit response is received but the post-submit
    # checkpoint has not run yet. The durable state is exactly the SUBMITTING
    # checkpoint written before the POST, which scenario 2 proves is never
    # re-submitted.
    def test_response_received_before_checkpoint_leaves_submitting(self):
        class BlockingClient(FakeClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.entered = threading.Event()
                self.release = threading.Event()

            def submit_simulation(self, expression, settings, alpha_type="REGULAR"):
                self.entered.set()
                self.release.wait(timeout=5)
                return super().submit_simulation(
                    expression, settings, alpha_type=alpha_type)

        import json

        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        client = BlockingClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=1, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments(["rank(f0)"])
        sched.add_jobs(jobs)

        worker = threading.Thread(target=sched.run)
        worker.start()
        self.assertTrue(client.entered.wait(timeout=5))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["running"][0]["status"], "SUBMITTING")
        self.assertIsNone(data["running"][0].get("progress_url"))
        self.assertEqual(data["submitted"], 1)

        client.release.set()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(jobs[0].status, "DONE")

    # Scenario 5: crash during poll with a persisted progress_url. Resume
    # keeps polling the existing simulation; zero POST /simulations.
    def test_crash_during_poll_resumes_with_poll(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        url = "progress-7"
        self._write(path, submitted=1, running=[self._submitted(expr, url)])

        client = FakeClient()
        client._expr_by_url[url] = expr
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.submissions, [], "poll only, no re-submit")
        self.assertEqual(jobs[0].status, "DONE")

    # Scenario 6: completed resume. The outcome is restored, nothing runs.
    def test_completed_resume_restores_without_submit(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        self._write(path, submitted=1, completed=[self._done(expr)])

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.counter, 0)
        self.assertEqual(jobs[0].status, "DONE")
        self.assertEqual(jobs[0].metrics["sharpe"], 1.0)

    # The ambiguous slot is never re-minted across a restart: budget=1 with one
    # already-consumed ambiguous slot leaves nothing for fresh expressions.
    def test_submit_unknown_slot_consumes_budget_across_restart(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        crashed, fresh = "rank(f0)", "rank(f1)"
        self._write(path, submitted=1, running=[self._submitting(crashed)])

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=1,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([crashed, fresh])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.counter, 0, "budget exhausted by the ambiguous slot")
        by_expr = {j.expression: j for j in jobs}
        self.assertEqual(by_expr[crashed].status, "SUBMIT_UNKNOWN")
        self.assertEqual(by_expr[fresh].status, "SKIPPED")
        self.assertEqual(sched._submitted, 1)

    # A double crash after the first resume checkpoints SUBMIT_UNKNOWN: the
    # second restart still refuses to submit the expression.
    def test_double_crash_keeps_submit_unknown_out_of_submission(self):
        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        self._write(path, submitted=1, running=[self._submitting(expr)])

        client1 = FakeClient()
        sched1 = BacktestScheduler(
            client1, Simulator(client1), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        sched1.add_jobs(make_experiments([expr]))
        sched1.run()

        client2 = FakeClient()
        sched2 = BacktestScheduler(
            client2, Simulator(client2), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs2 = make_experiments([expr])
        sched2.add_jobs(jobs2)
        sched2.run()
        self.assertEqual(client2.submissions, [], "still no re-submit after double crash")
        self.assertEqual(jobs2[0].status, "SUBMIT_UNKNOWN")
        self.assertEqual(sched2._submitted, 1)

    # Legacy checkpoints written by the old code recorded PENDING+no-url for
    # jobs whose submit may have reached the backend. Resume must treat them as
    # ambiguous rather than blindly re-submitting.
    def test_legacy_v1_pending_no_url_treated_as_ambiguous(self):
        import json

        tmpdir = tempfile.mkdtemp()
        path = self._path(tmpdir)
        expr = "rank(f0)"
        pending = self._pending(expr)
        with open(path, "w") as f:
            json.dump({
                "schema_version": 1,
                "submitted": 1,
                "running": [pending.to_dict()],
                "completed": [],
                "failed": [],
            }, f)

        client = FakeClient()
        sched = BacktestScheduler(
            client, Simulator(client), max_concurrent=2, budget=10,
            checkpoint_path=path, checkpoint_every=1,
        )
        jobs = make_experiments([expr])
        sched.add_jobs(jobs)
        sched.run()
        self.assertEqual(client.submissions, [], "legacy PENDING is ambiguous, no re-submit")
        self.assertEqual(jobs[0].status, "FAILED")
        self.assertIn("crash_mid_flight", jobs[0].error)
        self.assertEqual(sched._submitted, 1)


if __name__ == "__main__":
    unittest.main()
