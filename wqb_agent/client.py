import logging
import os
import random
import threading
import time

import requests

from .failures import FailureKind, classify_error

logger = logging.getLogger("wqb.client")

BASE_URL = "https://api.worldquantbrain.com"

CREDENTIALS_FILE = os.path.expanduser("~/.brain_credentials.txt")

# Status codes that indicate a permanent, non-retryable rejection.
FAIL_FAST_STATUSES = (400, 403, 404, 422)


class WQBAuthError(Exception):
    kind = FailureKind.AUTH


class WQBRateLimitError(Exception):
    kind = FailureKind.RATE_LIMIT


class WQBRejectedError(Exception):
    kind = FailureKind.SYNTAX


class WQBNotFoundError(Exception):
    kind = FailureKind.DATA


class WQBTimeoutError(Exception):
    kind = FailureKind.TIMEOUT


class WQBSimulationError(Exception):
    kind = FailureKind.INFRA


def load_credentials(username_env="WQB_USERNAME", password_env="WQB_PASSWORD"):
    username = os.environ.get(username_env)
    password = os.environ.get(password_env)
    if username and password:
        return username, password
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE) as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) >= 2:
            return lines[0], lines[1]
    return None, None


class WQBClient:
    """Thread-safe WQB API client.

    Each thread gets its own requests.Session and auth flag (thread-local) so
    concurrent simulations never share a mutable Session. Retry policy:

      401              -> re-authenticate and retry
      429              -> honor Retry-After
      5xx / conn/timeout -> exponential backoff + jitter
      400/403/404/422  -> fail fast with a classified error
    """

    def __init__(self, username=None, password=None, base_url=BASE_URL,
                 max_retries=5):
        self.base_url = base_url.rstrip("/")
        self.username, self.password = username, password
        if self.username is None or self.password is None:
            self.username, self.password = load_credentials()
        if not self.username or not self.password:
            raise WQBAuthError(
                "No credentials found. Set WQB_USERNAME/WQB_PASSWORD "
                "or create ~/.brain_credentials.txt (username, then password)."
            )
        self.max_retries = max_retries
        self._local = threading.local()
        self._auth_lock = threading.Lock()

    # ---- thread-local session ----

    def _session(self):
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({"Accept": "application/json"})
            self._local.session = sess
        return sess

    def _is_authenticated(self):
        return bool(getattr(self._local, "authenticated", False))

    def _set_authenticated(self, value):
        self._local.authenticated = value

    # ---- authentication ----

    def _authenticate(self):
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._session().post(
                    f"{self.base_url}/authentication",
                    json={"username": self.username, "password": self.password},
                    allow_redirects=False,
                    timeout=30,
                )
            except requests.exceptions.RequestException as exc:
                if attempt >= self.max_retries:
                    raise WQBSimulationError(
                        f"Authentication network error after "
                        f"{attempt} attempts: {exc}"
                    ) from exc
                time.sleep(self._backoff(attempt - 1))
                continue
            if resp.status_code == 200 or resp.status_code in (301, 302, 303):
                self._set_authenticated(True)
                return
            if resp.status_code == 401:
                raise WQBAuthError("Authentication rejected by WorldQuant BRAIN (401).")
            if resp.status_code == 429:
                retry = float(resp.headers.get("Retry-After", 5))
                time.sleep(min(retry, 30))
            else:
                raise WQBSimulationError(
                    f"Authentication failed with status {resp.status_code}."
                )
            if attempt >= self.max_retries:
                raise WQBSimulationError(
                    "Authentication failed after repeated rate limiting."
                )

    def _ensure_auth(self):
        if self._is_authenticated():
            return
        with self._auth_lock:
            if not self._is_authenticated():
                self._authenticate()

    # ---- retry helpers ----

    @staticmethod
    def _backoff(attempt, base=1.0, cap=30.0):
        return min(cap, base * (2 ** attempt) + random.uniform(0.0, base))

    def _sleep_retry_after(self, resp):
        try:
            retry = float(resp.headers.get("Retry-After", 5))
        except (TypeError, ValueError):
            retry = 5.0
        time.sleep(min(retry, 30))

    def _classified_exception(self, status_code, text, context):
        kind = classify_error(text, status_code)
        if status_code:
            msg = f"{context} failed with status {status_code}: {text[:300]}"
        else:
            msg = f"{context}: {text[:300]}"
        if kind == FailureKind.AUTH:
            return WQBAuthError(msg)
        if kind == FailureKind.RATE_LIMIT:
            return WQBRateLimitError(msg)
        if kind == FailureKind.SYNTAX:
            return WQBRejectedError(msg)
        if kind == FailureKind.DATA:
            return WQBNotFoundError(msg)
        if kind == FailureKind.TIMEOUT:
            return WQBTimeoutError(msg)
        return WQBSimulationError(msg)

    # ---- unified request ----

    def _request(self, method, url, *, params=None, json=None, timeout=60,
                 accepted=(200,), context="request", retry_ambiguous=True):
        """Single retry policy shared by every HTTP call.

        retry_ambiguous=False is for non-idempotent writes (POST /simulations):
        a network error, timeout, or 5xx after the request may have reached the
        server means we cannot tell whether it was accepted. Retrying would
        risk a duplicate backend object, so the first ambiguous outcome raises
        immediately. 401 re-auth and 429 retries are kept: both are confirmed
        non-acceptance, so re-sending is safe.
        """
        for attempt in range(self.max_retries):
            self._ensure_auth()
            try:
                resp = self._session().request(
                    method, url, params=params, json=json, timeout=timeout
                )
            except requests.exceptions.Timeout as exc:
                if not retry_ambiguous or attempt >= self.max_retries - 1:
                    raise WQBTimeoutError(
                        f"{context} timed out after {self.max_retries} attempts."
                    ) from exc
                time.sleep(self._backoff(attempt))
                continue
            except requests.exceptions.RequestException as exc:
                if not retry_ambiguous or attempt >= self.max_retries - 1:
                    raise WQBSimulationError(
                        f"{context} network error after {self.max_retries} "
                        f"attempts: {exc}"
                    ) from exc
                time.sleep(self._backoff(attempt))
                continue

            if resp.status_code in accepted:
                return resp
            if resp.status_code == 401:
                self._set_authenticated(False)
                self._ensure_auth()
                continue
            if resp.status_code == 429:
                self._sleep_retry_after(resp)
                continue
            if resp.status_code in FAIL_FAST_STATUSES:
                raise self._classified_exception(
                    resp.status_code, resp.text, context
                )
            if resp.status_code >= 500:
                if not retry_ambiguous or attempt >= self.max_retries - 1:
                    raise self._classified_exception(
                        resp.status_code, resp.text, context
                    )
                time.sleep(self._backoff(attempt))
                continue
            raise self._classified_exception(resp.status_code, resp.text, context)
        raise WQBSimulationError(f"{context} exceeded retries.")

    # ---- API ----

    def get_datasets(self):
        resp = self._request("GET", f"{self.base_url}/datasets", context="GET /datasets")
        return resp.json().get("results", [])

    def get_datafields(self, dataset_id, limit=50, offset=0):
        resp = self._request(
            "GET",
            f"{self.base_url}/datasets/{dataset_id}/datafields",
            params={"limit": limit, "offset": offset},
            context=f"GET datafields {dataset_id}",
        )
        payload = resp.json()
        return payload.get("results", []), payload.get("count", 0)

    def submit_simulation(self, expression, settings, alpha_type="REGULAR"):
        body = {"type": alpha_type, "settings": settings, "regular": expression}
        # A POST that is accepted by BRAIN creates a simulation that consumes
        # budget. Never auto-retry an ambiguous outcome (network error,
        # timeout, 5xx): the request may already have been accepted, and a
        # retry would double-submit the same expression.
        resp = self._request(
            "POST",
            f"{self.base_url}/simulations",
            json=body,
            accepted=(201, 200),
            context=f"submit simulation {expression[:60]}",
            retry_ambiguous=False,
        )
        location = resp.headers.get("Location")
        if not location:
            raise WQBSimulationError("Simulation response missing Location header.")
        return location

    def poll_progress(self, progress_url, timeout_sec=900):
        """Poll until the simulation has an alpha id.

        400/403/404/422 fail fast (permanent), 401 re-auths, 429 honors
        Retry-After, 5xx backs off, and the overall deadline is enforced.
        """
        start = time.time()
        while True:
            self._ensure_auth()
            try:
                resp = self._session().get(progress_url, timeout=60)
            except requests.exceptions.RequestException as exc:
                if time.time() - start > timeout_sec:
                    raise WQBTimeoutError("Simulation polling timed out.") from exc
                time.sleep(self._backoff(0))
                continue

            if resp.status_code == 401:
                self._set_authenticated(False)
                self._ensure_auth()
                continue
            if resp.status_code == 429:
                self._sleep_retry_after(resp)
                if time.time() - start > timeout_sec:
                    raise WQBTimeoutError("Simulation polling timed out.")
                continue
            if resp.status_code in FAIL_FAST_STATUSES:
                raise self._classified_exception(
                    resp.status_code, resp.text, "poll_progress"
                )
            if resp.status_code >= 500:
                if time.time() - start > timeout_sec:
                    raise WQBTimeoutError("Simulation polling timed out.")
                time.sleep(self._backoff(0))
                continue

            retry_after = resp.headers.get("Retry-After")
            if resp.status_code == 200 and retry_after is None:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = {}
                alpha_id = payload.get("alpha")
                if alpha_id:
                    return alpha_id
                raise WQBSimulationError(
                    f"Simulation finished without alpha id: {resp.text[:300]}"
                )
            if time.time() - start > timeout_sec:
                raise WQBTimeoutError("Simulation polling timed out.")
            delay = float(retry_after) if retry_after else 5.0
            time.sleep(min(delay, 30))

    def get_alpha(self, alpha_id):
        resp = self._request(
            "GET", f"{self.base_url}/alphas/{alpha_id}", context=f"GET alpha {alpha_id}"
        )
        return resp.json()

    def run_simulation(self, expression, settings, timeout_sec=900):
        progress_url = self.submit_simulation(expression, settings)
        alpha_id = self.poll_progress(progress_url, timeout_sec=timeout_sec)
        return self.get_alpha(alpha_id)
