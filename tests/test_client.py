import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wqb_agent.client import (
    WQBAuthError,
    WQBNotFoundError,
    WQBRejectedError,
    WQBTimeoutError,
    WQBClient,
)
from wqb_agent.simulator import _extract_metrics


class MockResponse:
    def __init__(self, status, json_data=None, headers=None, text=None):
        self.status_code = status
        self._json = json_data
        self.headers = headers or {}
        self.text = text or ""

    def json(self):
        return self._json


class FakeSession:
    """Pops responses in order; an entry may be an exception to raise."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)


def make_client(responses, max_retries=5, base="http://brain.test"):
    client = WQBClient(username="u", password="p", base_url=base,
                       max_retries=max_retries)
    client._set_authenticated(True)
    client._local.session = FakeSession(responses)
    return client


class TestRequestRetryPolicy(unittest.TestCase):
    def test_429_retries_until_success(self):
        client = make_client(
            [
                MockResponse(429, headers={"Retry-After": "0.0"}),
                MockResponse(429, headers={"Retry-After": "0.0"}),
                MockResponse(200, {"results": []}),
            ]
        )
        with patch("wqb_agent.client.time.sleep"):
            resp = client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(client._local.session.calls), 3)

    def test_429_honors_retry_after(self):
        client = make_client(
            [
                MockResponse(429, headers={"Retry-After": "0.0"}),
                MockResponse(200, {"results": []}),
            ]
        )
        with patch("wqb_agent.client.time.sleep") as sleep:
            client._request("GET", "http://brain.test/x", context="t")
        self.assertTrue(sleep.called)

    def test_422_fails_fast_without_retry(self):
        client = make_client(
            [MockResponse(422, text="bad expression")],
            max_retries=5,
        )
        with self.assertRaises(WQBRejectedError):
            client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(len(client._local.session.calls), 1)

    def test_400_fails_fast(self):
        client = make_client([MockResponse(400, text="bad request")])
        with self.assertRaises(WQBRejectedError):
            client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(len(client._local.session.calls), 1)

    def test_403_fails_fast(self):
        client = make_client([MockResponse(403, text="forbidden")])
        with self.assertRaises(Exception):
            client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(len(client._local.session.calls), 1)

    def test_404_fails_fast_as_not_found(self):
        client = make_client([MockResponse(404, text="not found")])
        with self.assertRaises(WQBNotFoundError):
            client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(len(client._local.session.calls), 1)

    def test_5xx_backs_off_then_succeeds(self):
        client = make_client(
            [
                MockResponse(500, text="boom"),
                MockResponse(503, text="busy"),
                MockResponse(200, {"ok": True}),
            ],
            max_retries=5,
        )
        with patch("wqb_agent.client.time.sleep"):
            resp = client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(resp.status_code, 200)

    def test_5xx_exhausted_raises_infra(self):
        client = make_client(
            [MockResponse(500, text="boom")] * 5, max_retries=5
        )
        with patch("wqb_agent.client.time.sleep"):
            with self.assertRaises(Exception) as ctx:
                client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(getattr(ctx.exception, "kind", None), "INFRA")

    def test_401_reauths_and_retries(self):
        client = make_client(
            [
                MockResponse(401, text="unauthorized"),
                MockResponse(200, {}, {"Set-Cookie": "s"}),
                MockResponse(200, {"ok": True}),
            ]
        )
        with patch("wqb_agent.client.time.sleep"):
            resp = client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(resp.status_code, 200)
        # GET (401) -> auth POST -> GET retry
        self.assertEqual(len(client._local.session.calls), 3)

    def test_auth_401_raises_auth_error(self):
        # Authentication endpoint itself rejects credentials.
        client = WQBClient(username="u", password="p", base_url="http://brain.test")
        client._set_authenticated(False)
        client._local.session = FakeSession([MockResponse(401, text="bad creds")])
        with self.assertRaises(WQBAuthError):
            client._ensure_auth()

    def test_connection_error_backs_off(self):
        import requests

        client = make_client(
            [
                requests.exceptions.ConnectionError("refused"),
                requests.exceptions.ConnectionError("refused"),
                MockResponse(200, {"ok": True}),
            ],
            max_retries=5,
        )
        with patch("wqb_agent.client.time.sleep"):
            resp = client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(resp.status_code, 200)

    def test_request_timeout_raises_timeout_kind(self):
        import requests

        client = make_client(
            [requests.exceptions.Timeout("slow")] * 2, max_retries=2
        )
        with patch("wqb_agent.client.time.sleep"):
            with self.assertRaises(Exception) as ctx:
                client._request("GET", "http://brain.test/x", context="t")
        self.assertEqual(getattr(ctx.exception, "kind", None), "TIMEOUT")


class TestSubmitSimulationAmbiguity(unittest.TestCase):
    """submit_simulation is a non-idempotent write: an ambiguous outcome
    (network error, timeout, 5xx) may already have been accepted by BRAIN, so
    it must raise after a single attempt instead of retrying into a duplicate
    simulation. Confirmed rejections (4xx) still fail fast with one call."""

    def _client(self, responses, max_retries=5):
        import requests

        return make_client(responses, max_retries=max_retries)

    def test_ambiguous_connection_error_does_not_retry_post(self):
        import requests

        client = self._client(
            [requests.exceptions.ConnectionError("response lost")] * 5,
            max_retries=5,
        )
        with patch("wqb_agent.client.time.sleep"):
            with self.assertRaises(Exception) as ctx:
                client.submit_simulation("rank(close)", {})
        self.assertEqual(getattr(ctx.exception, "kind", None), "INFRA")
        self.assertEqual(
            len(client._local.session.calls), 1,
            "a non-idempotent POST must never be retried on an ambiguous error",
        )

    def test_ambiguous_timeout_does_not_retry_post(self):
        import requests

        client = self._client(
            [requests.exceptions.Timeout("slow")] * 5, max_retries=5
        )
        with patch("wqb_agent.client.time.sleep"):
            with self.assertRaises(Exception) as ctx:
                client.submit_simulation("rank(close)", {})
        self.assertEqual(getattr(ctx.exception, "kind", None), "TIMEOUT")
        self.assertEqual(len(client._local.session.calls), 1)

    def test_ambiguous_5xx_does_not_retry_post(self):
        client = self._client([MockResponse(500, text="boom")] * 5, max_retries=5)
        with patch("wqb_agent.client.time.sleep"):
            with self.assertRaises(Exception) as ctx:
                client.submit_simulation("rank(close)", {})
        self.assertEqual(getattr(ctx.exception, "kind", None), "INFRA")
        self.assertEqual(len(client._local.session.calls), 1)

    def test_confirmed_rejection_fails_fast_with_one_call(self):
        client = self._client([MockResponse(422, text="bad expression")])
        with self.assertRaises(WQBRejectedError):
            client.submit_simulation("rank(close)", {})
        self.assertEqual(len(client._local.session.calls), 1)

    def test_accept_returns_location(self):
        client = self._client(
            [MockResponse(201, {}, {"Location": "http://brain.test/sims/1"})]
        )
        url = client.submit_simulation("rank(close)", {})
        self.assertEqual(url, "http://brain.test/sims/1")
        self.assertEqual(len(client._local.session.calls), 1)

    def test_missing_location_header_is_ambiguous(self):
        client = self._client([MockResponse(201, {}, {})])
        with self.assertRaises(Exception) as ctx:
            client.submit_simulation("rank(close)", {})
        self.assertEqual(getattr(ctx.exception, "kind", None), "INFRA")


class TestPollProgress(unittest.TestCase):
    def test_poll_422_fails_fast(self):
        client = make_client([MockResponse(422, text="invalid")])
        with self.assertRaises(WQBRejectedError):
            client.poll_progress("http://brain.test/p", timeout_sec=30)
        self.assertEqual(len(client._local.session.calls), 1)

    def test_poll_404_fails_fast(self):
        client = make_client([MockResponse(404, text="gone")])
        with self.assertRaises(WQBNotFoundError):
            client.poll_progress("http://brain.test/p", timeout_sec=30)

    def test_poll_429_then_alpha(self):
        client = make_client(
            [
                MockResponse(429, headers={"Retry-After": "0.0"}),
                MockResponse(200, {"alpha": "a1"}, {}),
            ]
        )
        with patch("wqb_agent.client.time.sleep"):
            alpha_id = client.poll_progress("http://brain.test/p", timeout_sec=30)
        self.assertEqual(alpha_id, "a1")

    def test_poll_timeout(self):
        # A 200 with Retry-After that never yields an alpha id eventually times
        # out instead of looping forever.
        import itertools

        client = make_client([MockResponse(200, {}, {"Retry-After": "0.1"})])
        clock = iter(itertools.chain([0.0], itertools.repeat(2.0)))
        with patch("wqb_agent.client.time.time", side_effect=lambda: next(clock)):
            with patch("wqb_agent.client.time.sleep"):
                with self.assertRaises(Exception) as ctx:
                    client.poll_progress("http://brain.test/p", timeout_sec=0.2)
        self.assertEqual(getattr(ctx.exception, "kind", None), "TIMEOUT")


class TestThreadLocalSession(unittest.TestCase):
    def test_each_thread_gets_its_own_session(self):
        import threading

        client = WQBClient(username="u", password="p", base_url="http://brain.test")
        seen = {}

        def worker():
            sess = client._session()
            seen[threading.current_thread().name] = id(sess)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(seen.values())), 4)


class TestExtractMetrics(unittest.TestCase):
    def test_empty_checks_is_unknown_not_pass(self):
        metrics = _extract_metrics({"is": {"sharpe": 1.0}})
        self.assertIsNone(metrics["passed"])

    def test_all_passing_checks_is_pass(self):
        metrics = _extract_metrics(
            {"is": {"checks": [{"name": "a", "pass": True}]}}
        )
        self.assertTrue(metrics["passed"])

    def test_any_failing_check_is_fail(self):
        metrics = _extract_metrics(
            {"is": {"checks": [{"name": "a", "pass": True},
                               {"name": "b", "pass": False}]}}
        )
        self.assertFalse(metrics["passed"])


if __name__ == "__main__":
    unittest.main()
