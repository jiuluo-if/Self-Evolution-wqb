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


if __name__ == "__main__":
    unittest.main()
