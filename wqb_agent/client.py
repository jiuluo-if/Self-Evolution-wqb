import json
import os
import time

import requests

BASE_URL = "https://api.worldquantbrain.com"

CREDENTIALS_FILE = os.path.expanduser("~/.brain_credentials.txt")


class WQBAuthError(Exception):
    pass


class WQBRequestError(Exception):
    pass


class WQBSimulationError(Exception):
    pass


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
    def __init__(self, username=None, password=None, base_url=BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.username, self.password = username, password
        if self.username is None or self.password is None:
            self.username, self.password = load_credentials()
        if not self.username or not self.password:
            raise WQBAuthError(
                "No credentials found. Set WQB_USERNAME/WQB_PASSWORD "
                "or create ~/.brain_credentials.txt (username, then password)."
            )
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._authenticated = False

    def _authenticate(self):
        resp = self.session.post(
            f"{self.base_url}/authentication",
            json={"username": self.username, "password": self.password},
            allow_redirects=False,
            timeout=30,
        )
        if resp.status_code == 200:
            self._authenticated = True
            return
        if resp.status_code in (301, 302, 303):
            self._authenticated = True
            return
        if resp.status_code == 401:
            raise WQBAuthError("Authentication rejected by WorldQuant BRAIN (401).")
        if resp.status_code == 429:
            retry = float(resp.headers.get("Retry-After", 5))
            time.sleep(min(retry, 30))
            return self._authenticate()
        raise WQBAuthError(f"Authentication failed with status {resp.status_code}.")

    def _ensure_auth(self):
        if not self._authenticated:
            self._authenticate()

    def _get(self, path, params=None, max_retries=3):
        self._ensure_auth()
        for attempt in range(max_retries):
            resp = self.session.get(f"{self.base_url}{path}", params=params, timeout=60)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                retry = float(resp.headers.get("Retry-After", 5))
                time.sleep(min(retry, 30))
                continue
            if resp.status_code == 401:
                self._authenticated = False
                self._ensure_auth()
                continue
            if resp.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            raise WQBRequestError(
                f"GET {path} failed with status {resp.status_code}: {resp.text[:300]}"
            )
        raise WQBRequestError(f"GET {path} exceeded retries.")

    def get_datasets(self):
        resp = self._get("/datasets")
        return resp.json().get("results", [])

    def get_datafields(self, dataset_id, limit=50, offset=0):
        resp = self._get(
            f"/datasets/{dataset_id}/datafields",
            params={"limit": limit, "offset": offset},
        )
        payload = resp.json()
        return payload.get("results", []), payload.get("count", 0)

    def submit_simulation(self, expression, settings, alpha_type="REGULAR"):
        self._ensure_auth()
        body = {"type": alpha_type, "settings": settings, "regular": expression}
        resp = self.session.post(
            f"{self.base_url}/simulations", json=body, timeout=60
        )
        if resp.status_code == 429:
            retry = float(resp.headers.get("Retry-After", 5))
            time.sleep(min(retry, 30))
            return self.submit_simulation(expression, settings, alpha_type)
        if resp.status_code in (400, 422):
            raise WQBSimulationError(
                f"Simulation rejected ({resp.status_code}): {resp.text[:500]}"
            )
        if resp.status_code == 401:
            self._authenticated = False
            self._ensure_auth()
            return self.submit_simulation(expression, settings, alpha_type)
        if resp.status_code >= 500:
            time.sleep(2)
            return self.submit_simulation(expression, settings, alpha_type)
        if resp.status_code != 201 and resp.status_code != 200:
            raise WQBSimulationError(
                f"Simulation submit failed ({resp.status_code}): {resp.text[:500]}"
            )
        location = resp.headers.get("Location")
        if not location:
            raise WQBSimulationError("Simulation response missing Location header.")
        return location

    def poll_progress(self, progress_url, timeout_sec=900):
        start = time.time()
        while True:
            self._ensure_auth()
            resp = self.session.get(progress_url, timeout=60)
            if resp.status_code == 401:
                self._authenticated = False
                self._ensure_auth()
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
                raise WQBSimulationError("Simulation polling timed out.")
            delay = float(retry_after) if retry_after else 5.0
            time.sleep(min(delay, 30))

    def get_alpha(self, alpha_id):
        resp = self._get(f"/alphas/{alpha_id}")
        return resp.json()

    def run_simulation(self, expression, settings, timeout_sec=900):
        progress_url = self.submit_simulation(expression, settings)
        alpha_id = self.poll_progress(progress_url, timeout_sec=timeout_sec)
        return self.get_alpha(alpha_id)
