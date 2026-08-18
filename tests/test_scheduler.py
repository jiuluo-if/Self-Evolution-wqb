import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        # The RUNNING job must be in the checkpoint at the moment the backend
        # submit call executes, i.e. the checkpoint happens before submit.
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
        self.assertEqual(data["running"][0]["status"], "RUNNING")
        self.assertEqual(data["submitted"], 1)

        client.release.set()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(jobs[0].status, "DONE")


if __name__ == "__main__":
    unittest.main()
