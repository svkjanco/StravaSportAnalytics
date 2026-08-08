from __future__ import annotations

"""Strava API client: OAuth + activity listing + cached stream fetching."""
import json
import time
from pathlib import Path

import requests

STRAVA_BASE = "https://www.strava.com/api/v3"
TOKEN_URL = "https://www.strava.com/oauth/token"

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
TOKEN_FILE = CACHE_DIR / "token.json"
STREAM_CACHE = CACHE_DIR / "streams"
STREAM_CACHE.mkdir(exist_ok=True)


def authorize_url(client_id: str) -> str:
    return (
        f"https://www.strava.com/oauth/authorize?client_id={client_id}"
        "&response_type=code&redirect_uri=http://localhost/exchange_token"
        "&approval_prompt=force&scope=read,activity:read_all"
    )


def exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    r = requests.post(TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "code": code, "grant_type": "authorization_code"}, timeout=30)
    r.raise_for_status()
    tok = r.json()
    _save_token(tok)
    return tok


def _save_token(tok: dict) -> None:
    TOKEN_FILE.write_text(json.dumps(tok, indent=2))


def _load_token() -> dict:
    if not TOKEN_FILE.exists():
        raise RuntimeError("No token yet. Run exchange_code(...) once.")
    return json.loads(TOKEN_FILE.read_text())


def _valid_access_token(client_id: str, client_secret: str) -> str:
    tok = _load_token()
    if tok.get("expires_at", 0) - 60 <= time.time():
        r = requests.post(TOKEN_URL, data={
            "client_id": client_id, "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": tok["refresh_token"]}, timeout=30)
        r.raise_for_status()
        tok = r.json()
        _save_token(tok)
    return tok["access_token"]


class StravaClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret

    def _headers(self):
        token = _valid_access_token(self.client_id, self.client_secret)
        return {"Authorization": f"Bearer {token}"}

    def _get(self, path: str, params: dict | None = None):
        while True:
            r = requests.get(f"{STRAVA_BASE}{path}", headers=self._headers(),
                             params=params or {}, timeout=30)
            if r.status_code == 429:
                print("Rate limited, sleeping 900s...")
                time.sleep(900)
                continue
            r.raise_for_status()
            return r.json()

    def list_activities(self, after_epoch: int | None = None,
                        per_page: int = 200) -> list[dict]:
        out, page = [], 1
        while True:
            params = {"per_page": per_page, "page": page}
            if after_epoch:
                params["after"] = after_epoch
            batch = self._get("/athlete/activities", params)
            if not batch:
                break
            out.extend(batch)
            page += 1
        return out

    def get_streams(self, activity_id: int) -> dict:
        cache_f = STREAM_CACHE / f"{activity_id}.json"
        if cache_f.exists():
            return json.loads(cache_f.read_text())

        keys = "time,watts,heartrate,cadence,distance,altitude,velocity_smooth"
        try:
            data = self._get(f"/activities/{activity_id}/streams",
                             {"keys": keys, "key_by_type": "true"})
        except requests.HTTPError as e:
            # activity has no streams (404) -> cache empty so we never retry
            if e.response is not None and e.response.status_code == 404:
                cache_f.write_text(json.dumps({}))
                return {}
            raise

        parsed = {k: v.get("data", []) for k, v in data.items()}
        cache_f.write_text(json.dumps(parsed))
        return parsed