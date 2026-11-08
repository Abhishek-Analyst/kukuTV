"""
Promo Finder - finds the best 20-60 second promo cuts in a microdrama's screenplays.

  1. PARSE     (code) screenplays -> numbered, timestamped timeline per episode
  2. PROPOSE   (LLM)  per episode: K promo cuts, referenced by LINE ID only (never timestamps)
  3. ASSEMBLE  (code) line IDs -> the script's own timestamps; snap cut points to camera-shot
                      boundaries; enforce 20-60s; repair over-long cuts by keeping the hook and
                      the cliffhanger and dropping the middle
  4. JUDGE     (LLM)  a separate reviewer call scores hook / cliffhanger / standalone / flow
  5. COLD VIEW (LLM)  finalists only: a "stranger" sees ONLY the clip and reports what they
                      understood and whether they would watch on
  6. STITCH    (LLM)  cross-episode promos built from the best moments, then steps 3-5 again
  7. RANK      (code) weighted score, de-duplication, a varied "post these first" list

Every model reply is saved in <out>/raw/ and reused on the next run, so rate limits or
crashes never lose work: just run the same command again.

Model and provider are swappable with environment variables (see README):
  LLM_PROVIDER=gemini|openai   LLM_MODELS=model1,model2,...   LLM_API_KEY (or GEMINI_API_KEY)
  LLM_BASE_URL (any OpenAI-compatible API)   LLM_THINKING   LLM_TEMPERATURE
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from parser import load_show, seconds_to_ts


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


CONFIG = {
    # --- model / provider: swap via environment variables, no code change ---
    "provider": os.environ.get("LLM_PROVIDER", "gemini").strip().lower(),
    # Free tier (Nov 2026): Flash-Lite models ~500 requests/day and 10/min; Flash models only 20/day.
    "models": _env_list("LLM_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.8-flash"),
    "base_url": os.environ.get("LLM_BASE_URL", "").strip(),
    "api_key": (os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip(),
    "temperature": os.environ.get("LLM_TEMPERATURE"),  # unset = model default
    "thinking": os.environ.get("LLM_THINKING"),        # unset = model default
    "timeout_sec": float(os.environ.get("LLM_TIMEOUT", "180")),
    "min_interval_sec": float(os.environ.get("LLM_MIN_INTERVAL", "7")),  # stay under 10 requests/min
    "attempts_per_model": 3,
    "cooldown_sec": 120,  # a failing model is skipped for 2 min, then tried again
    # --- promo rules from the brief ---
    "min_sec": 20.0,
    "max_sec": 60.0,
    "candidates_per_episode": 3,
    # --- engineering choices (see README) ---
    "snap_max_shift_sec": 2.0,  # move a cut point onto the nearest camera cut if within 2s
    "min_segment_sec": 1.0,
    "finalists": 25,            # only the top N get the per-clip checks (ending, stranger)
    "stitch_pool": 10,
    "stitch_count": 3,
    "post_first": 10,
    # premise = does it sell what the title promises (added after run 3: the top 10 missed the
    # proposal and the big reveal, the two moments the title is about)
    "judge_weights": {"hook": 0.25, "cliffhanger": 0.25, "standalone": 0.20, "flow": 0.10, "premise": 0.20},
    "max_per_beat": 3,              # post-first variety: at most 3 promos of one story beat
    "final_weights": {"judge": 0.5, "intrigue": 0.3, "comprehension": 0.2},
    "min_comprehension_to_post": 3,
    "min_cliffhanger_to_post": 3.5, # the brief's must-haves gate the post-first list, not just the
    "min_hook_to_post": 3,          # blended score (judge scores are averages of several votes)
    "min_intrigue_to_post": 3,      # run 4: an Ep 3 clip the stranger would scroll past (2/5) made the list
    # --- added after run 4 (see README, "What changed after run 4") ---
    "repair_rounds": 2,             # a rejected cut goes back to the model up to twice
    "judge_votes": 3,               # the judge scores each episode's clips N times; scores are averaged
    "end_check": True,              # finalists: the model names the peak line, code ends the clip there
    "open_check": True,             # finalists: the model names the hook line, code starts the clip there
    "min_standalone_to_post": 3,    # a stranger must be able to follow it (judge standalone, averaged)
}

DEFAULT_BASE_URL = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openai": "https://api.openai.com/v1",
}


# ----------------------------------------------------------------------------
# LLM client: one place to swap models/providers; survives overload and limits
# ----------------------------------------------------------------------------

class LLMError(RuntimeError):
    """The run cannot continue (bad key, or no model left). Finished steps stay saved."""


class BlockedError(ValueError):
    """The model's safety filter refused this one request: skip the step, keep the run going."""


BLOCK_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


def _retry_wait(err: str, retry_after, default: int) -> int:
    m = re.search(r'retryDelay"?\s*:\s*"(\d+(?:\.\d+)?)s', err or "")
    if m:
        return min(int(float(m.group(1))) + 1, 120)
    if retry_after and str(retry_after).strip().isdigit():
        return min(int(str(retry_after).strip()) + 1, 120)
    return min(default, 120)


def _short_error(err: str) -> str:
    m = re.search(r'"message"\s*:\s*"([^"]+)', err or "")
    return (m.group(1) if m else (err or "")).strip()[:160]


class LLM:
    """Native Gemini API or any OpenAI-compatible API (Groq, OpenRouter, OpenAI,
    or an API Kuku provides). If a model is overloaded, missing or out of quota,
    it moves to the next model in LLM_MODELS."""

    def __init__(self, cfg: dict, log_path: Path, fake=None):
        self.cfg = cfg
        self.fake = fake
        self.log_path = log_path
        self.dead = set()     # missing for this key, or out of today's quota
        self.cooldown = {}    # model -> time until which it is skipped
        self.blocked = 0
        self.last_start = 0.0
        self.json_ok = {}
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.models_used = Counter()
        self.latencies = []

    def _log(self, rec: dict):
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _pace(self):
        wait = self.cfg["min_interval_sec"] - (time.time() - self.last_start)
        if wait > 0:
            time.sleep(wait)
        self.last_start = time.time()

    def _request(self, model: str, prompt: str, json_mode: bool):
        import requests
        cfg = self.cfg
        if cfg["provider"] == "gemini":
            base = (cfg["base_url"] or DEFAULT_BASE_URL["gemini"]).rstrip("/")
            url = f"{base}/models/{model}:generateContent"
            gen = {}
            if json_mode:
                gen["responseMimeType"] = "application/json"
            if cfg["temperature"]:
                gen["temperature"] = float(cfg["temperature"])
            if cfg["thinking"]:
                gen["thinkingConfig"] = {"thinkingLevel": cfg["thinking"]}
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            if gen:
                body["generationConfig"] = gen
            headers = {"x-goog-api-key": cfg["api_key"], "Content-Type": "application/json"}
        else:
            base = (cfg["base_url"] or DEFAULT_BASE_URL["openai"]).rstrip("/")
            url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
            body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            if cfg["temperature"]:
                body["temperature"] = float(cfg["temperature"])
            if cfg["thinking"]:
                body["reasoning_effort"] = cfg["thinking"]
            headers = {"Authorization": f"Bearer {cfg['api_key']}",
                       "Content-Type": "application/json"}

        r = requests.post(url, json=body, headers=headers, timeout=cfg["timeout_sec"])
        text, usage, blocked = "", {}, None
        if r.status_code == 200:
            data = r.json()
            if cfg["provider"] == "gemini":
                blocked = (data.get("promptFeedback") or {}).get("blockReason")
                cands = data.get("candidates") or []
                if cands:
                    parts = (cands[0].get("content") or {}).get("parts") or []
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    if not text.strip() and cands[0].get("finishReason") in BLOCK_REASONS:
                        blocked = cands[0]["finishReason"]
                u = data.get("usageMetadata") or {}
                usage = {"in": u.get("promptTokenCount", 0) or 0,
                         "out": (u.get("candidatesTokenCount", 0) or 0)
                         + (u.get("thoughtsTokenCount", 0) or 0)}
            else:
                choices = data.get("choices") or []
                if choices:
                    text = (choices[0].get("message") or {}).get("content") or ""
                    if not text.strip() and choices[0].get("finish_reason") == "content_filter":
                        blocked = "content_filter"
                u = data.get("usage") or {}
                usage = {"in": u.get("prompt_tokens", 0) or 0,
                         "out": u.get("completion_tokens", 0) or 0}
        return r.status_code, text, usage, r.text[:5000], r.headers.get("Retry-After"), blocked

    def complete(self, prompt: str, purpose: str) -> tuple[str, str]:
        """Best available model first. A model that keeps failing is skipped for a cooldown,
        then tried again; one that is missing or out of today's quota is dropped for the run."""
        self.calls += 1
        if self.fake is not None:
            self.models_used["dry-run"] += 1
            self._log({"purpose": purpose, "model": "dry-run", "status": 200})
            return self.fake(prompt), "dry-run"
        if not self.cfg["api_key"]:
            raise LLMError("No API key found. In this terminal run:  "
                           "export GEMINI_API_KEY=\"your-key\"   (Windows: set GEMINI_API_KEY=...)")
        for _ in range(3):
            alive = [m for m in self.cfg["models"] if m not in self.dead]
            if not alive:
                raise LLMError("No model left to try: every model in LLM_MODELS is missing or out of "
                               "today's free quota. Finished steps are saved. Run the same command after "
                               "the quota resets (midnight Pacific = 1:30 pm IST), or add other models "
                               "to LLM_MODELS (see --list-models).")
            ready = [m for m in alive if self.cooldown.get(m, 0) <= time.time()]
            if not ready:
                soonest = min(alive, key=lambda m: self.cooldown.get(m, 0))
                wait = max(0.0, self.cooldown[soonest] - time.time())
                print(f"    [llm] every model is struggling; trying again in {wait:.0f}s")
                time.sleep(wait)
                ready = [m for m in alive if self.cooldown.get(m, 0) <= time.time()] or [soonest]
            for model in ready:
                text = self._try_model(model, prompt, purpose)
                if text is not None:
                    return text, model
                if model not in self.dead:
                    self.cooldown[model] = time.time() + self.cfg["cooldown_sec"]
                    print(f"    [llm] {model} keeps failing; skipping it for "
                          f"{self.cfg['cooldown_sec']}s")
        raise LLMError("The models kept failing (overloaded or timing out). Finished steps are "
                       "saved; run the same command again in a few minutes.")

    def _try_model(self, model: str, prompt: str, purpose: str):
        """Text on success, None if this model failed for now. Raises LLMError (stop the
        run) or BlockedError (skip this step)."""
        import requests
        tries = self.cfg["attempts_per_model"]
        attempt = 0      # failures: overload, timeouts, empty replies
        rate_waits = 0   # rate limits mean "slow down", not "broken": more patience
        while attempt < tries:
            attempt += 1
            json_mode = self.cfg["provider"] == "gemini" and self.json_ok.get(model, True)
            self._pace()
            t0 = time.time()
            try:
                status, text, usage, err, retry_after, blocked = self._request(model, prompt, json_mode)
            except requests.exceptions.Timeout:
                status, text, usage, err, retry_after, blocked = "timeout", "", {}, "", None, None
            except requests.exceptions.ConnectionError as e:
                status, text, usage, err, retry_after, blocked = (
                    "connection_error", "", {}, str(e)[:300], None, None)
            latency = round(time.time() - t0, 2)
            ok = status == 200 and bool(text.strip())
            self._log({"purpose": purpose, "model": model, "status": status,
                       "latency_s": latency, "tokens_in": usage.get("in"),
                       "tokens_out": usage.get("out"), "attempt": attempt,
                       "error": None if ok else (f"blocked: {blocked}" if blocked
                                                 else _short_error(err) or "empty reply")})

            if ok:
                self.models_used[model] += 1
                self.tokens_in += usage.get("in", 0)
                self.tokens_out += usage.get("out", 0)
                self.latencies.append(latency)
                return text
            if status == 200 and blocked:
                self.blocked += 1
                raise BlockedError(f"the model's safety filter blocked this request ({blocked})")
            if status == 200:
                print(f"    [llm] {model} sent an empty reply, retrying ...")
                continue
            if status == 400 and json_mode:
                self.json_ok[model] = False  # model refuses JSON mode: retry without it
                attempt -= 1
                continue
            if status in (401, 403):
                raise LLMError(f"API key rejected (HTTP {status}): {_short_error(err)}")
            # (Google's free daily-limit message also says "check your plan and billing details",
            #  so only these exact codes mean "out of credit")
            if status == 402 or (status == 429 and re.search(
                    r"insufficient_quota|credit_balance", err or "", re.I)):
                raise LLMError(f"Out of credit or billing not set up (HTTP {status}): "
                               f"{_short_error(err).rstrip('.')}. "
                               "Add credit, then run the same command again; finished steps are saved.")
            if status == 400:
                hint = (" If it mentions reasoning_effort or thinking, run: unset LLM_THINKING"
                        if self.cfg["thinking"] else "")
                raise LLMError(f"{model} rejected the request (HTTP 400): {_short_error(err)}.{hint}")
            if status == 404:
                print(f"    [llm] model '{model}' is not available for this key")
                self.dead.add(model)
                return None
            if status == 429:
                if re.search(r"per ?day|perday|daily|limit:\s*0\b", err or "", re.I):
                    print(f"    [llm] {model}: today's free quota is used up (or it has none)")
                    self.dead.add(model)
                    return None
                rate_waits += 1
                if rate_waits > 6:
                    print(f"    [llm] {model}: still rate limited after 6 waits")
                    return None
                attempt -= 1
                wait = _retry_wait(err, retry_after, default=15 * rate_waits)
                print(f"    [llm] {model}: rate limited, waiting {wait}s ...")
                time.sleep(wait)
                continue
            if status in (500, 502, 503, 504, "timeout", "connection_error"):
                if attempt < tries:
                    wait = 8 * attempt
                    print(f"    [llm] {model}: {status} (busy), retrying in {wait}s ({attempt}/{tries})")
                    time.sleep(wait)
                else:
                    print(f"    [llm] {model}: {status} (busy) ({attempt}/{tries})")
                continue
            raise LLMError(f"Unexpected reply from {model} (HTTP {status}): {_short_error(err)}")
        return None


def ping(cfg: dict):
    import requests
    print(f"Provider: {cfg['provider']}   API key: {'set' if cfg['api_key'] else 'MISSING'}")
    client = LLM(dict(cfg, timeout_sec=60), Path(os.devnull))
    for m in cfg["models"]:
        t0 = time.time()
        try:
            status, text, _, err, _, _ = client._request(m, "Reply with exactly: OK", json_mode=False)
        except requests.exceptions.RequestException as e:
            status, text, err = type(e).__name__, "", str(e)
        ok = status == 200 and text.strip()
        detail = text.strip()[:20] if ok else _short_error(err)
        print(f"  {'OK  ' if ok else 'FAIL'} {m:<30} {status!s:<6} {time.time() - t0:5.1f}s  {detail}")


def list_models(cfg: dict):
    import requests
    if cfg["provider"] == "gemini":
        base = (cfg["base_url"] or DEFAULT_BASE_URL["gemini"]).rstrip("/")
        names, token = [], None
        while True:
            params = {"pageSize": 1000}
            if token:
                params["pageToken"] = token
            r = requests.get(f"{base}/models", params=params, timeout=60,
                             headers={"x-goog-api-key": cfg["api_key"]})
            if r.status_code != 200:
                print(f"HTTP {r.status_code}: {_short_error(r.text)}")
                return
            data = r.json()
            names += [m["name"].split("/", 1)[-1] for m in data.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", [])]
            token = data.get("nextPageToken")
            if not token:
                break
    else:
        base = (cfg["base_url"] or DEFAULT_BASE_URL["openai"]).rstrip("/")
        r = requests.get(f"{base}/models", timeout=60,
                         headers={"Authorization": f"Bearer {cfg['api_key']}"})
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {_short_error(r.text)}")
            return
        names = [m.get("id", "") for m in r.json().get("data", [])]
    print("Models this key can use for text generation:")
    for n in sorted(names):
        print("  ", n)


# ----------------------------------------------------------------------------
# JSON replies, cached on disk (the cache doubles as the raw-outputs folder)
# ----------------------------------------------------------------------------

JSON_NUDGE = "\n\nIMPORTANT: reply with valid JSON only. No prose, no markdown."


def extract_json(text: str):
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    starts = [i for i in (t.find("["), t.find("{")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in the reply")
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[min(starts):])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON ({e})") from None
    return obj


def coerce(obj, expect: str):
    if expect == "list":
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
            return [obj]
    else:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
    raise ValueError(f"expected a JSON {expect}")


def ask_json(llm: LLM, raw_dir: Path, name: str, prompt: str, purpose: str, expect: str):
    """Ask once, keep the raw reply as raw/<name>.txt and reuse it on later runs -
    but only if the prompt that produced it is unchanged (tracked in _prompt_index.json)."""
    path = raw_dir / f"{name}.txt"
    index_path = raw_dir / "_prompt_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:12]
    if path.exists():
        if index.get(name) == h:
            try:
                return coerce(extract_json(path.read_text(encoding="utf-8")), expect), True
            except ValueError:
                pass
        path.replace(raw_dir / f"{name}__stale.txt")  # different prompt or unusable: redo
    last_err = None
    for i in range(2):
        text, _ = llm.complete(prompt if i == 0 else prompt + JSON_NUDGE, purpose)
        try:
            obj = coerce(extract_json(text), expect)
        except ValueError as e:
            (raw_dir / f"{name}__unparseable_{i + 1}.txt").write_text(text, encoding="utf-8")
            last_err = e
            continue
        path.write_text(text, encoding="utf-8")
        index[name] = h
        index_path.write_text(json.dumps(index, indent=0), encoding="utf-8")
        return obj, False
    raise ValueError(f"{name}: the model's reply was not usable JSON ({last_err})")


# ----------------------------------------------------------------------------
# Show data: timelines with line IDs, camera shots
# ----------------------------------------------------------------------------

TYPE_ORDER = {"CAMERA": 0, "ACTION": 1, "SOUND": 2, "DIALOGUE": 3}
LABEL = {"CAMERA": "CAM", "ACTION": "ACTION", "SOUND": "SOUND"}


def line_text(r: dict) -> str:
    if r["type"] == "DIALOGUE":
        tone = f" ({r['tone']})" if r["tone"] else ""
        return f'{r["speaker"]}{tone}: "{r["content"]}"'
    return f"{LABEL.get(r['type'], r['type'])}: {r['content']}"


class Show:
    def __init__(self, title: str, episodes: list[dict]):
        self.title = title
        self.episodes = {e["episode_number"]: e for e in episodes}
        self.rows, self.shots, self.duration = {}, {}, {}
        for n, ep in self.episodes.items():
            rows = []
            for sc in ep["scenes"]:
                for ev in sc["events"]:
                    rows.append({"scene": sc["scene_number"], "heading": sc["heading"],
                                 "type": ev["type"], "speaker": ev["speaker"], "tone": ev["tone"],
                                 "start": ev["start"], "end": ev["end"], "content": ev["content"]})
            rows.sort(key=lambda r: (r["start"], TYPE_ORDER.get(r["type"], 9), r["end"]))
            for i, r in enumerate(rows, 1):
                r["id"] = i
            self.rows[n] = rows
            self.shots[n] = sorted((r["start"], r["end"]) for r in rows if r["type"] == "CAMERA")
            self.duration[n] = ep["duration_sec"]

    def render_timeline(self, n: int) -> str:
        out, last = [], None
        for r in self.rows[n]:
            if r["scene"] != last:
                out.append(f"=== Scene {r['scene']}: {r['heading']} ===")
                last = r["scene"]
            out.append(f"[L{r['id']:03d}] {seconds_to_ts(r['start'])}-{seconds_to_ts(r['end'])} "
                       f"{line_text(r)}")
        return "\n".join(out)

    def scene_key(self, ep: int, scene: int) -> tuple[int, int]:
        """Microdrama episodes often open on a scene headed '... CONTINUOUS': it picks up the
        previous episode's last scene, so both are the same moment for the viewer."""
        while ep - 1 in self.episodes and self.episodes[ep]["scenes"] and self.episodes[ep - 1]["scenes"]:
            first = self.episodes[ep]["scenes"][0]
            if first["scene_number"] != scene or "CONTINUOUS" not in first["heading"].upper():
                break
            ep, scene = ep - 1, self.episodes[ep - 1]["scenes"][-1]["scene_number"]
        return ep, scene

    def moments(self, segs: list[dict]) -> set:
        return {self.scene_key(s["episode"], r["scene"]) for s in segs
                for r in self.rows[s["episode"]][s["start_line"] - 1:s["end_line"]]}

    def size_hint(self, n: int) -> dict:
        """How many timeline lines make 20/30/60 seconds in this episode. The model counts
        lines well but subtracts timestamps badly (run 1: 28 of 150 cuts were too short)."""
        lps = len(self.rows[n]) / max(self.duration[n], 1.0)
        return {"l20": max(1, round(20 * lps)), "l30": max(1, round(30 * lps)),
                "l60": max(1, round(60 * lps))}

    def render_clip(self, segs: list[dict], rel_times: bool = True) -> str:
        """What plays on screen, in order. rel_times adds time from the clip's start."""
        out, offset = [], 0.0
        for k, s in enumerate(segs):
            if k > 0:
                out.append("[CUT]")
            for r in self.rows[s["episode"]][s["start_line"] - 1:s["end_line"]]:
                rel = offset + max(0.0, r["start"] - s["start_sec"])
                out.append((f"+{rel:04.1f}s " if rel_times else "") + line_text(r))
            offset += s["end_sec"] - s["start_sec"]
        return "\n".join(out)


# ----------------------------------------------------------------------------
# ASSEMBLE (code): line IDs -> timestamps, shot snapping, 20-60s rule, repair
# ----------------------------------------------------------------------------

def seg_times(show: Show, ep: int, a: int, b: int) -> tuple[float, float]:
    rows = show.rows[ep][a - 1:b]
    start = min(r["start"] for r in rows)
    end = max(r["end"] for r in rows)
    lim = CONFIG["snap_max_shift_sec"]
    snap = [s for s, e in show.shots[ep] if s <= start < e and start - s <= lim]
    if snap:
        start = max(snap)          # start on the camera cut this moment begins in
    snap = [e for s, e in show.shots[ep] if s < end <= e and e - end <= lim]
    if snap:
        end = min(snap)            # end where that camera shot ends
    if show.duration[ep]:
        end = min(end, show.duration[ep])
    return round(start, 2), round(end, 2)


def make_segment(show: Show, ep: int, a: int, b: int) -> dict:
    s, e = seg_times(show, ep, a, b)
    return {"episode": ep, "start_line": a, "end_line": b, "start_sec": s, "end_sec": e}


def total_duration(segs: list[dict]) -> float:
    return round(sum(s["end_sec"] - s["start_sec"] for s in segs), 2)


def fix_adjacent_overlaps(segs: list[dict]):
    for x, y in zip(segs, segs[1:]):
        if (x["episode"] == y["episode"] and x["end_line"] < y["start_line"]
                and x["end_sec"] > y["start_sec"]):
            x["end_sec"] = max(x["start_sec"], y["start_sec"])


def describe_lines(items: list[tuple[int, int]]) -> str:
    by_ep = {}
    for ep, ln in items:
        by_ep.setdefault(ep, set()).add(ln)
    parts = []
    for ep, lines in sorted(by_ep.items()):
        lines = sorted(lines)
        start = prev = lines[0]
        for x in lines[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            parts.append(f"Ep{ep} L{start:03d}" + (f"-L{prev:03d}" if prev != start else ""))
            if x is not None:
                start = prev = x
    return ", ".join(parts)


def trim_to_fit(show: Show, segs: list[dict]):
    """Over-long cut: keep the opening (hook) and the ending (cliffhanger) and drop lines
    from the middle, the way a promo editor would. A single range is split in two first."""
    work = [dict(s) for s in segs]
    if len(work) == 1:
        s = work[0]
        if s["end_line"] - s["start_line"] < 3:
            return None, None
        mid = (s["start_line"] + s["end_line"]) // 2
        work = [make_segment(show, s["episode"], s["start_line"], mid),
                make_segment(show, s["episode"], mid + 1, s["end_line"])]
    target = CONFIG["max_sec"] - 1.0
    removed = []
    for _ in range(2000):
        fix_adjacent_overlaps(work)
        if total_duration(work) <= target:
            break
        last = len(work) - 1
        # the ending gets more room than the opening: it carries the cliffhanger build-up
        weight = [1.0 if i == 0 else 1.5 if i == last else 0.8 for i in range(len(work))]
        order = sorted(range(len(work)),
                       key=lambda i: -(work[i]["end_sec"] - work[i]["start_sec"]) / weight[i])
        for i in order:
            w = work[i]
            if w["end_line"] <= w["start_line"]:
                continue  # keep at least one line per segment
            if i == last:  # trim from the front of the last segment (keeps the cliffhanger)
                removed.append((w["episode"], w["start_line"]))
                work[i] = make_segment(show, w["episode"], w["start_line"] + 1, w["end_line"])
            else:          # trim from the end of other segments (keeps the hook)
                removed.append((w["episode"], w["end_line"]))
                work[i] = make_segment(show, w["episode"], w["start_line"], w["end_line"] - 1)
            break
        else:
            return None, None
    fix_adjacent_overlaps(work)
    if total_duration(work) > CONFIG["max_sec"]:
        return None, None
    return work, describe_lines(removed)


def _as_int(x) -> int:
    """12, 12.0, "12", "L012", "[L012]", "Ep3" -> int"""
    if isinstance(x, bool):
        raise ValueError
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    m = re.search(r"\d+", str(x)) if isinstance(x, str) else None
    if not m:
        raise ValueError
    return int(m.group(0))


def assemble(show: Show, cand, default_ep=None, allowed=None, max_segments: int = 3):
    """Proposal from the model -> checked promo, or (None, reason)."""
    if not isinstance(cand, dict):
        return None, "proposal is not a JSON object"
    raw = cand.get("segments")
    if isinstance(raw, dict):
        raw = [raw]
    if not raw and ("start_line" in cand or "start" in cand):
        raw = [cand]
    if not raw:
        return None, "no segments given"

    ranges = []
    for s in raw:
        if not isinstance(s, dict):
            return None, "segment is not a JSON object"
        try:
            ep = _as_int(s.get("episode", default_ep))
            a = _as_int(s.get("start_line", s.get("start")))
            b = _as_int(s.get("end_line", s.get("end")))
        except (TypeError, ValueError):
            return None, "segment without valid episode/start_line/end_line"
        if ep not in show.rows:
            return None, f"unknown episode {ep}"
        if a > b:
            a, b = b, a
        n = len(show.rows[ep])
        if a < 1 or b > n:
            return None, f"Ep{ep} L{a}-L{b} is outside the script (L1-L{n})"
        if allowed is not None and not any(ep == e and lo <= a and b <= hi for e, lo, hi in allowed):
            return None, f"Ep{ep} L{a}-L{b} is not inside the moments offered for stitching"
        ranges.append((ep, a, b))
    if len(ranges) > max_segments:
        return None, f"more than {max_segments} segments"
    for i, (e1, a1, b1) in enumerate(ranges):
        for e2, a2, b2 in ranges[i + 1:]:
            if e1 == e2 and not (b1 < a2 or b2 < a1):
                return None, f"segments overlap (Ep{e1} L{a1}-L{b1} and L{a2}-L{b2})"

    segs = [make_segment(show, *r) for r in ranges]
    fix_adjacent_overlaps(segs)
    repair = None
    before = total_duration(segs)
    if before > CONFIG["max_sec"]:
        segs, removed = trim_to_fit(show, segs)
        if segs is None:
            return None, f"too long ({before:.1f}s) and could not be trimmed"
        repair = f"was {before:.0f}s; dropped {removed} from the middle to fit 60s"
    dur = total_duration(segs)
    if dur < CONFIG["min_sec"]:
        return None, f"too short ({dur:.1f}s)"
    if dur > CONFIG["max_sec"]:
        return None, f"too long ({dur:.1f}s)"
    if any(s["end_sec"] - s["start_sec"] < CONFIG["min_segment_sec"] for s in segs):
        return None, "contains a segment shorter than 1s"
    for s in segs:
        s["start"], s["end"] = seconds_to_ts(s["start_sec"]), seconds_to_ts(s["end_sec"])
    return {"segments": segs, "duration_sec": dur,
            "episodes": sorted({s["episode"] for s in segs}),
            "hook": str(cand.get("hook", "")).strip(),
            "cliffhanger": str(cand.get("cliffhanger", "")).strip(),
            "why": str(cand.get("why", "")).strip(),
            "repair": repair, "trimmed": repair is not None}, None


def reason_kind(reason: str) -> str:
    """'Ep6 L95-L127 is outside the script (L1-L94)' -> 'is outside the script'"""
    r = re.sub(r"\(.*?\)|Ep\d+ L\d+-L\d+|L\d+-L\d+", "", reason)
    return re.sub(r"\s+", " ", r).strip(" .,;")


def describe_proposal(cand) -> str:
    """A rejected proposal in words the model can act on, even if it was malformed."""
    if not isinstance(cand, dict):
        return json.dumps(cand)[:120]
    segs = cand.get("segments")
    if isinstance(segs, dict):
        segs = [segs]
    if not segs and ("start_line" in cand or "start" in cand):
        segs = [cand]
    parts = []
    for s in segs or []:
        try:
            a = _as_int(s.get("start_line", s.get("start")))
            b = _as_int(s.get("end_line", s.get("end")))
            parts.append(f"L{a:03d}-L{b:03d}")
        except (AttributeError, TypeError, ValueError):
            parts.append(json.dumps(s)[:80])
    return " + ".join(parts) or "(no segments)"


def overlap_frac(p: dict, q: dict) -> float:
    """Share of the shorter promo's runtime that also appears in the other one."""
    inter = 0.0
    for s in p["segments"]:
        for t in q["segments"]:
            if s["episode"] == t["episode"]:
                inter += max(0.0, min(s["end_sec"], t["end_sec"]) - max(s["start_sec"], t["start_sec"]))
    return inter / max(1e-6, min(p["duration_sec"], q["duration_sec"]))


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

CANDIDATE_PROMPT = """You are an expert promo editor for vertical microdramas (ReelShort, DramaBox, Kuku TV).
A promo is a 20-60 second clip cut from the show and posted on Instagram, TikTok or YouTube to pull in NEW viewers.

A great promo:
1. OPENS ON A HOOK: in the first 3 seconds something makes a scrolling viewer stop - a slap, a shocking line,
   a reveal, a public humiliation, a threat. Start inside the conflict, never on an establishing shot or a walk-in.
2. ENDS ON A CLIFFHANGER: cut at the unresolved peak - right before the answer, the blow or the reveal.
   Never after the tension is released.
3. MAKES SENSE ON ITS OWN: a stranger must grasp who is wronged, by whom, and what is at stake from the clip alone.
4. SELLS THE SHOW'S PROMISE: the title tells viewers what this show promises. Moments that deliver it - the
   central relationship, the big secret or reveal the title hints at - make the strongest promos, as long as
   they open on their sharpest line, not on the slow build-up.

Show: "{title}"
Episode {ep}. Summary: {logline}

Timeline. Every line has an ID. CAM = camera shot, ACTION / SOUND = what we see and hear, NAME = dialogue.
{timeline}

Propose the {k} best promo cuts from this episode, best first, each a genuinely different moment where possible.
If this episode has a moment that delivers the title's promise, make one of your cuts about it.
A cut is usually ONE continuous range of lines; you may join up to 3 ranges from this episode if that makes a
stronger clip. A strong pattern: open on the episode's sharpest line, then cut to its final cliffhanger.
Refer to lines ONLY by their IDs; never write timestamps. Each cut must total 20-60 seconds; 30-45 seconds is ideal.

SIZE CHECK: in this episode, 20 seconds is about {l20} lines of the timeline, 30 seconds about {l30} lines and
60 seconds about {l60} lines. Before answering, check every cut: from the start time of its first line to the end
time of its last line must be at least 20 seconds. Cuts under 20 seconds are thrown away.

Reply ONLY with a JSON array:
[{{"segments": [{{"start_line": <int>, "end_line": <int>}}],
   "hook": "<what happens in the first 3 seconds>",
   "cliffhanger": "<the unresolved moment it ends on>",
   "why": "<one line: why a stranger would stop scrolling and then need to watch the show>"}}]
"""

REPAIR_PROMPT = """You are an expert promo editor for vertical microdramas (ReelShort, DramaBox, Kuku TV).
You proposed promo cuts from episode {ep} of "{title}", but an automatic check rejected the ones listed at the end.
Fix each one, keeping its idea where possible.

Rules: a promo runs 20-60 seconds (30-45 is ideal), opens on a hook in its first 3 seconds, ends on an unresolved
moment, and makes sense to a stranger. A cut is one continuous range of lines, or at most 3 ranges from this episode.
Refer to lines ONLY by their IDs. SIZE CHECK: in this episode, 20 seconds is about {l20} lines of the timeline and
30 seconds about {l30} lines.

Episode summary: {logline}

TIMELINE:
{timeline}

REJECTED CUTS (with the exact problem):
{rejected}

Reply ONLY with a JSON array, one fixed cut per rejected cut, in the same order:
[{{"fixes": "R1", "segments": [{{"start_line": <int>, "end_line": <int>}}],
   "hook": "<what happens in the first 3 seconds of the fixed cut>",
   "cliffhanger": "<the unresolved moment the fixed cut ends on>",
   "why": "<one line: why a stranger would stop scrolling and then need to watch the show. Describe the promo, not your fix.>"}}]
"""

JUDGE_PROMPT = """You are a harsh, experienced reviewer of promo clips for vertical microdramas (ReelShort, DramaBox, Kuku TV).
A promo is a 20-60 second clip posted on Instagram, TikTok or YouTube to pull in NEW viewers. Score each clip below.

Background for you only (the viewer will not know it): {context}

Each clip is listed as what plays on screen, in order, with the time from the clip's start (+0.0s).
[CUT] marks a jump to a different moment.

{clips}

Score every clip from 0 to 5 on each criterion. Be harsh: 3 is average, 5 is rare.
- hook: does something in the first 3 seconds (+0.0s to +3.0s) stop a scrolling thumb? A slap, a shocking line,
  a reveal or visible humiliation scores high; an establishing shot, walking or calm talk scores low.
- cliffhanger: does it END on an unresolved moment that makes the viewer need the next scene? Cutting right
  before an answer, a blow or a reveal scores high; ending calm or resolved scores low.
- standalone: would someone who has NEVER seen the show understand who is wronged, by whom, and what is at stake?
- flow: does it play as one coherent clip, without confusing jumps or dead air?
- premise: does it sell what the show's title promises ("{title}")? 5 = it shows the central relationship or the
  big secret/reveal the title hints at; 1 = a side plot that could belong to any show.
Also name each clip's main story beat, exactly one of: betrayal, romance, reveal, danger, revenge, other.

Reply ONLY with a JSON array, one object per clip:
[{{"clip": "C1", "hook": <0-5>, "cliffhanger": <0-5>, "standalone": <0-5>, "flow": <0-5>, "premise": <0-5>,
   "beat": "<betrayal|romance|reveal|danger|revenge|other>",
   "note": "<one line: the biggest strength or weakness>"}}]
"""

BEATS = ("betrayal", "romance", "reveal", "danger", "revenge", "other")
# Bumped whenever scoring changes; scores are only comparable between runs with the same version.
SCORING_VERSION = 8
CRITERIA = ("hook", "cliffhanger", "standalone", "flow", "premise")

START_PROMPT = """You are a promo editor for vertical microdramas. A promo must OPEN on its hook: in the first 3 seconds
something must stop a scrolling viewer - a shocking line, an accusation, a threat, a reveal, a slap. Opening on an
establishing shot, someone walking in, or calm set-up loses viewers before the story starts.

This promo plays as follows. Each line shows its ID and its time from the promo's start ([CUT] = a jump):

{clip}

On which line should this promo START so its first 3 seconds hook a stranger, while a stranger can still follow what is
at stake? Keep the build-up only if the clip makes no sense without it. If it already opens on its hook, give its first line.
Reply ONLY with JSON:
{{"start_line": "<a line ID from the list, e.g. L042>",
  "hook": "<what the viewer sees and hears in the first 3 seconds from that line>",
  "reason": "<one short line>"}}
"""

END_PROMPT = """You are a promo editor for vertical microdramas. A promo must END on its peak: the unresolved moment
right before an answer, a blow or a reveal. Anything after the tension is released - a calm exit, walking away,
a hug, a slow fade - makes viewers scroll on instead of tapping to watch the show.

This promo plays as follows. Each line shows its ID and its time from the promo's start ([CUT] = a jump):

{clip}

On which line should this promo END so it stops at its peak, before the tension is released?
If it already ends at its peak, give its last line.
Reply ONLY with JSON:
{{"end_line": "<a line ID from the list, e.g. L042>",
  "cliffhanger": "<the unresolved moment the promo ends on at that line>",
  "reason": "<one short line>"}}
"""

COLD_PROMPT = """You are a random person scrolling Instagram Reels. You have NEVER seen this show and know nothing about it.
You just watched a clip. This is exactly what you saw and heard, in order ([CUT] = the clip jumps to another moment):

{clip}

Answer honestly as that viewer; most clips are easy to scroll past. Reply ONLY with JSON:
{{"what_i_think_is_happening": "<1-2 sentences, your best guess>",
  "comprehension": <1-5, 1 = totally lost, 5 = instantly clear>,
  "intrigue": <1-5, 1 = keep scrolling, 5 = I must tap to watch the show>,
  "question_left_with": "<the question stuck in your head, or 'none'>"}}
"""

STITCH_PROMPT = """You are a promo editor for vertical microdramas. Below are the strongest moments found in this show.
You may JOIN moments from DIFFERENT episodes into one promo, like a trailer: open on the most shocking line from
one episode, then cut to a later confrontation and end on its unresolved peak. The whole promo must be 20-60 seconds.

Show: {premise}

MOMENTS (every line shows its episode and line ID):
{moments}

Propose up to {k} stitched promos. Each must use moments from AT LEAST TWO different episodes. You may use part of a
moment (any sub-range of its line IDs). List segments in the order they should PLAY. Keep the total 20-60 seconds
(estimate from the timestamps shown).

Reply ONLY with a JSON array:
[{{"segments": [{{"episode": <int>, "start_line": <int>, "end_line": <int>}}],
   "hook": "...", "cliffhanger": "...", "why": "<one line>"}}]
"""


# ----------------------------------------------------------------------------
# JUDGE and COLD VIEWER
# ----------------------------------------------------------------------------

def _num(x, lo: float, hi: float):
    try:
        return round(max(float(lo), min(float(hi), float(x))), 2)
    except (TypeError, ValueError):
        return None


def judge_score(j) -> float | None:
    if not j:
        return None
    w = CONFIG["judge_weights"]
    have = {k: j[k] for k in w if j.get(k) is not None}
    if not have:
        return None
    return round(sum(w[k] * v for k, v in have.items()) / sum(w[k] for k in have), 3)


def run_judge(llm, show, raw, name, promos, context, votes=None, title=None):
    """Score clips. The judge is asked `votes` times (one call per vote, all clips of an episode
    in each call) and each score is the average: a single vote is noisy (run 4: the same clip
    got hook 4 in one run and 2 in the next)."""
    votes = votes or CONFIG["judge_votes"]
    clips = []
    for k, p in enumerate(promos, 1):
        clips.append(f"--- CLIP C{k} ({p['duration_sec']:.0f}s) ---\n{show.render_clip(p['segments'])}")
    prompt = JUDGE_PROMPT.format(title=title or show.title, context=context, clips="\n\n".join(clips))
    got = [[] for _ in promos]
    cached_all, errors = True, []
    for v in range(1, votes + 1):
        try:
            scores, cached = ask_json(llm, raw, name if v == 1 else f"{name}_v{v}", prompt, "judge", "list")
        except ValueError as e:
            errors.append(str(e))
            continue
        cached_all = cached_all and cached
        by_id = {str(x.get("clip", "")).strip().upper(): x for x in scores if isinstance(x, dict)}
        for k in range(1, len(promos) + 1):
            x = by_id.get(f"C{k}")
            if x is None and len(scores) == len(promos) and isinstance(scores[k - 1], dict):
                x = scores[k - 1]
            if isinstance(x, dict):
                got[k - 1].append(x)
    if errors and not any(got):
        raise ValueError(errors[0])
    for p, vs in zip(promos, got):
        if not vs:
            p["judge"], p["judge_score"] = None, None
            continue
        j, spread = {}, {}
        for c in CRITERIA:
            vals = [x for x in (_num(v.get(c), 0, 5) for v in vs) if x is not None]
            j[c] = round(sum(vals) / len(vals), 2) if vals else None
            spread[c] = round(max(vals) - min(vals), 2) if vals else None
        beats = []
        for v in vs:
            b = str(v.get("beat", "")).strip().lower()
            beats.append(b if b in BEATS else "other")
        j["beat"] = Counter(beats).most_common(1)[0][0]
        j["note"] = str(vs[0].get("note", "")).strip()
        j["votes"], j["spread"] = len(vs), spread
        p["judge"] = j
        p["judge_score"] = judge_score(j)
    return cached_all


def render_clip_with_ids(show, segs) -> str:
    out, offset = [], 0.0
    for k, sg in enumerate(segs):
        if k > 0:
            out.append("[CUT]")
        for r in show.rows[sg["episode"]][sg["start_line"] - 1:sg["end_line"]]:
            rel = offset + max(0.0, r["start"] - sg["start_sec"])
            out.append(f"[L{r['id']:03d}] +{rel:04.1f}s {line_text(r)}")
        offset += sg["end_sec"] - sg["start_sec"]
    return "\n".join(out)


def end_at_peak(show, p, end_id: int):
    """Cut a single-episode clip so it ends on line end_id. Returns (segments, None) or (None, why not)."""
    segs = p["segments"]
    for si, sg in enumerate(segs):
        if sg["start_line"] <= end_id <= sg["end_line"]:
            break
    else:
        return None, "the named line is not in the clip"
    if si == len(segs) - 1 and end_id == sg["end_line"]:
        return None, "already ends at its peak"
    new = [dict(x) for x in segs[:si]] + [make_segment(show, sg["episode"], sg["start_line"], end_id)]
    fix_adjacent_overlaps(new)
    dur = total_duration(new)
    if dur < CONFIG["min_sec"]:
        return None, f"ending there would leave {dur:.1f}s (under {CONFIG['min_sec']:.0f}s)"
    if p["duration_sec"] - dur < 1.5:
        return None, "under 1.5s would go"
    if dur < 0.6 * p["duration_sec"]:
        return None, "it would drop more than 40% of the clip"
    for x in new:
        x["start"], x["end"] = seconds_to_ts(x["start_sec"]), seconds_to_ts(x["end_sec"])
    return new, None


def start_at_hook(show, p, start_id: int):
    """Cut a single-episode clip so it starts on line start_id. Returns (segments, None) or (None, why not)."""
    segs = p["segments"]
    for si, sg in enumerate(segs):
        if sg["start_line"] <= start_id <= sg["end_line"]:
            break
    else:
        return None, "the named line is not in the clip"
    if si == 0 and start_id == sg["start_line"]:
        return None, "already opens on its hook"
    new = [make_segment(show, sg["episode"], start_id, sg["end_line"])] + [dict(x) for x in segs[si + 1:]]
    fix_adjacent_overlaps(new)
    dur = total_duration(new)
    if dur < CONFIG["min_sec"]:
        return None, f"starting there would leave {dur:.1f}s (under {CONFIG['min_sec']:.0f}s)"
    if p["duration_sec"] - dur < 1.5:
        return None, "under 1.5s would go"
    if dur < 0.55 * p["duration_sec"]:
        return None, "it would drop more than 45% of the clip"
    for x in new:
        x["start"], x["end"] = seconds_to_ts(x["start_sec"]), seconds_to_ts(x["end_sec"])
    return new, None


def run_open_check(llm, show, raw, p):
    """Ask where the hook is; code starts the clip there (The Last Signature test: the best cut opened on
    a calm shot of a pen, so the judge gave it hook 2 and the tool dropped it)."""
    ans, cached = ask_json(llm, raw, f"{p['id']}_opening", START_PROMPT.format(
        clip=render_clip_with_ids(show, p["segments"])), "opening", "dict")
    try:
        start_id = _as_int(ans.get("start_line"))
    except (TypeError, ValueError):
        return False, "no usable line ID in the reply"
    new, why_not = start_at_hook(show, p, start_id)
    if new is None:
        return False, why_not
    before = p["duration_sec"]
    p["segments"], p["duration_sec"] = new, total_duration(new)
    if str(ans.get("hook", "")).strip():
        p["hook"] = str(ans["hook"]).strip()
    p["opening"] = (f"opened before its hook; code started it at L{start_id:03d} ({before:.0f}s -> "
                    f"{p['duration_sec']:.0f}s): {str(ans.get('reason', '')).strip()}")
    return True, None


def run_end_check(llm, show, raw, p):
    """Ask where the clip peaks; code ends it there (run 4: the #2 pick ran on past its peak into
    a calm walk-off that the judge still scored 4 for cliffhanger)."""
    ans, cached = ask_json(llm, raw, f"{p['id']}_ending", END_PROMPT.format(
        clip=render_clip_with_ids(show, p["segments"])), "ending", "dict")
    try:
        end_id = _as_int(ans.get("end_line"))
    except (TypeError, ValueError):
        return False, "no usable line ID in the reply"
    new, why_not = end_at_peak(show, p, end_id)
    if new is None:
        return False, why_not
    before = p["duration_sec"]
    p["segments"], p["duration_sec"] = new, total_duration(new)
    if str(ans.get("cliffhanger", "")).strip():
        p["cliffhanger"] = str(ans["cliffhanger"]).strip()
    p["ending"] = (f"ran past its peak; code ended it at L{end_id:03d} ({before:.0f}s -> "
                   f"{p['duration_sec']:.0f}s): {str(ans.get('reason', '')).strip()}")
    return True, None


def run_cold_viewer(llm, show, raw, p):
    cv, cached = ask_json(llm, raw, f"{p['id']}_coldviewer", COLD_PROMPT.format(
        clip=show.render_clip(p["segments"], rel_times=False)), "cold_viewer", "dict")
    p["cold_viewer"] = {
        "comprehension": _num(cv.get("comprehension"), 1, 5),
        "intrigue": _num(cv.get("intrigue"), 1, 5),
        "what_i_think_is_happening": str(cv.get("what_i_think_is_happening", "")).strip(),
        "question_left_with": str(cv.get("question_left_with", "")).strip(),
    }
    c = p["cold_viewer"]
    if p.get("judge_score") is not None and c["comprehension"] and c["intrigue"]:
        w = CONFIG["final_weights"]
        p["final_score"] = round(w["judge"] * p["judge_score"] + w["intrigue"] * c["intrigue"]
                                 + w["comprehension"] * c["comprehension"], 3)
    return cached


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def quality_failures(p: dict) -> list[str]:
    """Every quality bar a clip misses (empty list = good enough to post)."""
    c, j, out = p.get("cold_viewer") or {}, p.get("judge") or {}, []
    if (c.get("comprehension") or 0) < CONFIG["min_comprehension_to_post"]:
        out.append(f"a stranger was lost (comprehension {c.get('comprehension')})")
    if (c.get("intrigue") or 0) < CONFIG["min_intrigue_to_post"]:
        out.append(f"a stranger would scroll past (intrigue {c.get('intrigue')})")
    if (j.get("cliffhanger") or 0) < CONFIG["min_cliffhanger_to_post"]:
        out.append(f"weak cliffhanger (judge {j.get('cliffhanger')}/5)")
    if (j.get("hook") or 0) < CONFIG["min_hook_to_post"]:
        out.append(f"weak hook (judge {j.get('hook')}/5)")
    if (j.get("standalone") or 0) < CONFIG["min_standalone_to_post"]:
        out.append(f"hard to follow on its own (judge standalone {j.get('standalone')}/5)")
    return out


def pick_post_first(show: Show, checked: list[dict]):
    """Walk the stranger-checked promos best first. Returns (picked, skipped with reasons)."""
    post, skipped, used_eps, used_moments, beats = [], [], {}, {}, Counter()
    for p in checked:
        if len(post) >= CONFIG["post_first"]:
            break
        j = p.get("judge") or {}
        beat = j.get("beat") or "other"
        fails = quality_failures(p)
        same = [used_moments[m] for m in show.moments(p["segments"]) if m in used_moments]
        if fails:
            why_not = fails[0]
        elif p["type"] == "single-episode" and p["episodes"][0] in used_eps:
            why_not = f"episode already used by {used_eps[p['episodes'][0]]}"
        elif any(overlap_frac(p, q) > 0 for q in post):
            why_not = "repeats footage already picked"
        elif beat != "other" and beats[beat] >= CONFIG["max_per_beat"]:
            why_not = f"already {beats[beat]} '{beat}' promos in the list (variety)"
        elif same:
            why_not = f"same scene as {same[0]} (the episode picks up where it ends)"
        else:
            why_not = None
        if why_not:
            skipped.append((p, why_not))
            continue
        post.append(p)
        beats[beat] += 1
        if p["type"] == "single-episode":
            used_eps[p["episodes"][0]] = p["id"]
        for m in show.moments(p["segments"]):
            used_moments.setdefault(m, p["id"])
    return post, skipped


def ep_meta(show, n, title, premise):
    """(title, premise) for episode n. A pack of separate stories gives each file its own TITLE."""
    ep = show.episodes[n]
    t = ep.get("title") or title
    t = t.title() if t.isupper() else t
    return t, (premise if t == title else f'"{t}". {ep["logline"]}')


def run(args):
    t0 = time.time()
    title, episodes = load_show(args.input)
    empty = [e for e in episodes if e["n_events"] < 3]
    episodes = [e for e in episodes if e["n_events"] >= 3]
    for e in empty:
        print(f"Skipping {e['file']}: could not read timed lines from it.")
    if not episodes:
        sys.exit(
            "\nNo usable episodes: no file had lines with start/end times, so there is nothing to cut.\n"
            "The tool expects each line of a script to carry its time, for example:\n"
            "    **CAMERA:**\n    Start: (00:01.00) | End: (00:04.80)\n    Close-up on the heroine...\n"
            "Also accepted: no ** stars, no brackets, '00:01.00 - 00:04.80' or '00:01.00 --> 00:04.80',\n"
            "and the header and the time on one line. No API calls were made.")
    if args.limit:
        episodes = episodes[:args.limit]
    show = Show(title, episodes)
    out = Path(args.out)
    raw = out / "raw"
    if args.fresh and raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True, exist_ok=True)
    llm = LLM(CONFIG, out / "calls_log.jsonl", fake=fake_llm if args.dry_run else None)
    first_ep = next(iter(show.episodes.values()))
    premise = f'"{title}". It begins: {first_ep["logline"]}'

    print(f"Show: {title} ({len(show.episodes)} episodes)")
    print(f"Models: {'dry-run (fake replies)' if args.dry_run else ', '.join(CONFIG['models'])}"
          f"  [{CONFIG['provider']}]")
    if not args.dry_run:
        fin = min(args.finalists, 2 * len(show.episodes))
        est = ((1 + CONFIG["judge_votes"]) * len(show.episodes) + 3 * fin
               + 6 + CONFIG["judge_votes"] + args.stability)
        mins = max(1, round(est * (CONFIG["min_interval_sec"] + 3) / 60))
        print(f"Expect about {est} model calls, roughly {mins} min "
              "(finished steps are skipped). You can leave it running.")
    print(f"Outputs: {out}/\n")

    stats = Counter()
    first_reasons = Counter()
    rejected, singles, stitched = [], [], []
    stopped = None
    stability = None

    # ---- 2-4: propose, assemble, judge (two model calls per episode) ----
    try:
        for i, (n, ep) in enumerate(show.episodes.items(), 1):
            tag = f"ep{n:03d}"
            ep_title, ep_premise = ep_meta(show, n, title, premise)
            try:
                cands, cached = ask_json(llm, raw, f"{tag}_candidates", CANDIDATE_PROMPT.format(
                    title=ep_title, ep=n, logline=ep["logline"], timeline=show.render_timeline(n),
                    k=CONFIG["candidates_per_episode"], **show.size_hint(n)), "candidates", "list")
            except ValueError as e:
                rejected.append({"source": tag, "reason": str(e)})
                print(f"Ep {n:>3} ({i}/{len(show.episodes)}): no usable proposals ({e})")
                continue
            promos, to_fix = [], []
            for j, c in enumerate(cands[:CONFIG["candidates_per_episode"]], 1):
                stats["proposed"] += 1
                p, why_not = assemble(show, c, default_ep=n)
                if p is None:
                    to_fix.append((j, c, why_not))
                    first_reasons[reason_kind(why_not)] += 1
                    continue
                stats["repaired"] += p["trimmed"]
                p.update({"id": f"{tag}_c{j}", "type": "single-episode"})
                promos.append(p)

            # Validator feedback: a rejected cut goes back to the model with the exact problem,
            # up to repair_rounds times
            fixed_now = 0
            if to_fix:
                stats["sent_back"] += len(to_fix)
            problems = {j: [why] for j, c, why in to_fix}
            pending = [(j, c, why, c) for j, c, why in to_fix]  # slot, original, problem, latest version
            for rnd in range(1, CONFIG["repair_rounds"] + 1):
                if not pending:
                    break
                listing = []
                for k, (j, c, why, latest) in enumerate(pending, 1):
                    idea = c if isinstance(c, dict) else {}
                    item = (f"R{k}: lines {describe_proposal(latest)}. PROBLEM: {why}; promos must run "
                            f"20-60 seconds. Its idea - hook: \"{str(idea.get('hook', ''))[:200]}\"; "
                            f"cliffhanger: \"{str(idea.get('cliffhanger', ''))[:200]}\"")
                    m = re.search(r"too short \(([\d.]+)s\)", why)
                    if rnd > 1 and m:
                        lps = show.size_hint(n)["l20"] / 20
                        more = max(2, round((CONFIG["min_sec"] + 5 - float(m.group(1))) * lps))
                        item += (f". This is the second try: it is still {m.group(1)}s, so extend it by "
                                 f"at least {more} lines")
                    listing.append(item)
                name = f"{tag}_repair" if rnd == 1 else f"{tag}_repair{rnd}"
                try:
                    fixes, _ = ask_json(llm, raw, name, REPAIR_PROMPT.format(
                        ep=n, title=ep_title, logline=ep["logline"], timeline=show.render_timeline(n),
                        rejected="\n".join(listing), **show.size_hint(n)), "repair", "list")
                except ValueError as e:
                    fixes = []
                    rejected.append({"source": name, "reason": str(e)})
                by_id = {str(f.get("fixes", "")).strip().upper(): f for f in fixes if isinstance(f, dict)}
                still = []
                for k, (j, c, why, latest) in enumerate(pending, 1):
                    f = by_id.get(f"R{k}")
                    if f is None and len(fixes) == len(pending) and isinstance(fixes[k - 1], dict):
                        f = fixes[k - 1]
                    p2, why2 = assemble(show, f, default_ep=n) if f else (None, "no fix was returned")
                    if p2 is None:
                        problems[j].append(why2)
                        still.append((j, c, why2, f or latest))
                        continue
                    stats["fixed_by_model"] += 1
                    stats[f"fixed_round{rnd}"] += 1
                    stats["repaired"] += p2["trimmed"]
                    fixed_now += 1
                    p2["repair"] = (f"code rejected the first version ({problems[j][0]}); the model fixed it"
                                    + (" on its second try" if rnd > 1 else "")
                                    + (f"; then {p2['repair']}" if p2["repair"] else ""))
                    p2.update({"id": f"{tag}_c{j}", "type": "single-episode"})
                    promos.append(p2)
                pending = still
            for j, c, why, latest in pending:
                stats["rejected"] += 1
                tries = len(problems[j]) - 1
                rejected.append({"source": f"{tag}_c{j}", "proposal": c, "fix": latest,
                                 "reason": f"{problems[j][0]}; still rejected after "
                                           f"{'one fix' if tries == 1 else f'{tries} fixes'}: "
                                           + "; ".join(problems[j][1:])})

            if promos:
                try:
                    run_judge(llm, show, raw, f"{tag}_judge", promos,
                              context=f"{ep_premise} This is episode {n}: {ep['logline']}", title=ep_title)
                except ValueError as e:
                    rejected.append({"source": f"{tag}_judge", "reason": str(e)})
            singles.extend(promos)
            best = max((p["judge_score"] for p in promos if p.get("judge_score") is not None),
                       default=None)
            print(f"Ep {n:>3} ({i}/{len(show.episodes)}): {len(cands)} proposed -> {len(promos)} valid"
                  f" ({fixed_now} fixed after feedback, {sum(p['trimmed'] for p in promos)} trimmed)"
                  f"{f', best judge score {best:.2f}' if best is not None else ''}"
                  f"{'  [cached]' if cached else ''}")

        # ---- 5: cold viewer on de-duplicated finalists ----
        scored = sorted((p for p in singles if p.get("judge_score") is not None),
                        key=lambda p: -p["judge_score"])
        finalists = []
        for p in scored:
            if len(finalists) >= args.finalists:
                break
            dup = next((f for f in finalists if overlap_frac(p, f) >= 0.5), None)
            if dup:
                p["note"] = f"near-duplicate of {dup['id']}"
                stats["near_duplicates"] += 1
                continue
            finalists.append(p)
        changed = []
        if CONFIG["open_check"]:
            print(f"\nOpening check on {len(finalists)} finalists ...")
            for p in finalists:
                stats["open_checked"] += 1
                try:
                    moved, _ = run_open_check(llm, show, raw, p)
                except ValueError as e:
                    rejected.append({"source": f"{p['id']}_opening", "reason": str(e)})
                    continue
                if moved:
                    stats["open_trimmed"] += 1
                    changed.append(p)
        if CONFIG["end_check"]:
            print(f"{'' if CONFIG['open_check'] else chr(10)}Ending check on {len(finalists)} finalists ...")
            for p in finalists:
                stats["end_checked"] += 1
                try:
                    moved, _ = run_end_check(llm, show, raw, p)
                except ValueError as e:
                    rejected.append({"source": f"{p['id']}_ending", "reason": str(e)})
                    continue
                if moved:
                    stats["end_trimmed"] += 1
                    if p not in changed:
                        changed.append(p)
        if changed:  # the clip changed, so its old scores no longer apply: judge it again, per story
            groups = {}
            for p in changed:
                t, prem = ep_meta(show, p["episodes"][0], title, premise)
                groups.setdefault((t, prem), []).append(p)
            for k, ((t, prem), ps) in enumerate(groups.items()):
                name = "ending_rejudge" if len(groups) == 1 else f"rejudge_{k + 1:02d}"
                try:
                    run_judge(llm, show, raw, name, ps, context=prem, title=t)
                except ValueError as e:
                    rejected.append({"source": name, "reason": str(e)})
        print(f"Cold-viewer check on {len(finalists)} finalists ...")
        for p in finalists:
            try:
                run_cold_viewer(llm, show, raw, p)
            except ValueError as e:
                rejected.append({"source": f"{p['id']}_coldviewer", "reason": str(e)})

        # ---- 6: stitched cross-episode promos ----
        pool = sorted((p for p in finalists if p.get("final_score") is not None),
                      key=lambda p: -p["final_score"])[:CONFIG["stitch_pool"]]
        separate = len({(e.get("title") or "").strip().lower() for e in show.episodes.values()
                        if e.get("title")}) > 1
        if separate:
            print("Stitching skipped: these episodes are separate stories.")
        elif len({p["episodes"][0] for p in pool}) >= 2:
            print("Stitching cross-episode promos ...")
            moments, allowed = [], []
            for k, p in enumerate(pool, 1):
                lines = []
                for e in p["episodes"]:  # any range inside a moment's span is allowed
                    own = [s for s in p["segments"] if s["episode"] == e]
                    allowed.append((e, min(s["start_line"] for s in own),
                                    max(s["end_line"] for s in own)))
                for si, s in enumerate(p["segments"]):
                    if si > 0:
                        lines.append("[... lines skipped ...]")
                    for r in show.rows[s["episode"]][s["start_line"] - 1:s["end_line"]]:
                        lines.append(f"[Ep{s['episode']} L{r['id']:03d}] {seconds_to_ts(r['start'])}-"
                                     f"{seconds_to_ts(r['end'])} {line_text(r)}")
                moments.append(f"--- MOMENT {k} (episode {p['episodes'][0]}) ---\n" + "\n".join(lines))
            try:
                cands, _ = ask_json(llm, raw, "stitched_candidates", STITCH_PROMPT.format(
                    premise=premise, moments="\n\n".join(moments), k=CONFIG["stitch_count"]),
                    "stitch", "list")
            except ValueError as e:
                cands = []
                rejected.append({"source": "stitched", "reason": str(e)})
            for j, c in enumerate(cands[:CONFIG["stitch_count"]], 1):
                stats["stitch_proposed"] += 1
                p, why_not = assemble(show, c, None, allowed, max_segments=4)
                if p is not None and len(p["episodes"]) < 2:
                    p, why_not = None, "uses only one episode"
                if p is None:
                    rejected.append({"source": f"stitched_c{j}", "reason": why_not, "proposal": c})
                    continue
                p.update({"id": f"stitched_c{j}", "type": "stitched"})
                stitched.append(p)
            if stitched:
                try:
                    run_judge(llm, show, raw, "stitched_judge", stitched, context=premise)
                except ValueError as e:
                    rejected.append({"source": "stitched_judge", "reason": str(e)})
                for p in stitched:
                    if p.get("judge_score") is not None:
                        try:
                            run_cold_viewer(llm, show, raw, p)
                        except ValueError as e:
                            rejected.append({"source": f"{p['id']}_coldviewer", "reason": str(e)})

        # ---- stability: propose again for the first N episodes, compare picks ----
        if args.stability:
            print(f"Stability check on the first {args.stability} episodes ...")
            stability = {"episodes": [], "picks": 0, "reproduced": 0}
            for n in list(show.episodes)[:args.stability]:
                ep = show.episodes[n]
                ep_title = ep.get("title") or title
                ep_title = ep_title.title() if ep_title.isupper() else ep_title
                try:
                    again, _ = ask_json(llm, raw, f"ep{n:03d}_candidates_rerun", CANDIDATE_PROMPT.format(
                        title=ep_title, ep=n, logline=ep["logline"], timeline=show.render_timeline(n),
                        k=CONFIG["candidates_per_episode"], **show.size_hint(n)), "stability", "list")
                except ValueError:
                    continue
                second = [q for q in (assemble(show, c, n)[0] for c in again[:CONFIG["candidates_per_episode"]]) if q]
                first = [p for p in singles if p["episodes"] == [n]]
                hits = sum(any(overlap_frac(p, q) >= 0.5 for q in second) for p in first)
                stability["episodes"].append({"episode": n, "first_run_picks": len(first),
                                              "reproduced": hits})
                stability["picks"] += len(first)
                stability["reproduced"] += hits
    except (LLMError, KeyboardInterrupt) as e:
        stopped = "stopped by user" if isinstance(e, KeyboardInterrupt) else str(e)
        print(f"\nSTOPPED: {stopped}")

    # ---- 7: rank ----
    everything = singles + stitched
    checked = sorted((p for p in everything if p.get("final_score") is not None),
                     key=lambda p: -p["final_score"])
    unchecked = sorted((p for p in everything if p.get("final_score") is None
                        and p.get("judge_score") is not None), key=lambda p: -p["judge_score"])
    unscored = [p for p in everything if p.get("judge_score") is None]
    ranked = checked + unchecked + unscored
    for r_i, p in enumerate(ranked, 1):
        p["rank"] = r_i
    post, skipped = pick_post_first(show, checked)
    for k, p in enumerate(post, 1):
        p["post_order"] = k
    for p, why in skipped:
        p["post_skip"] = why

    summary = {
        "show": title, "episodes": len(show.episodes),
        "provider": "dry-run" if args.dry_run else CONFIG["provider"],
        "models_used": dict(llm.models_used), "model_calls_made": sum(llm.models_used.values()),
        "tokens_in": llm.tokens_in, "tokens_out": llm.tokens_out,
        "avg_latency_s": round(sum(llm.latencies) / len(llm.latencies), 2) if llm.latencies else None,
        "wall_time_min": round((time.time() - t0) / 60, 1),
        "proposed": stats["proposed"], "first_pass_rejected": stats["sent_back"],
        "first_pass_reasons": dict(first_reasons.most_common()),
        "fixed_by_model": stats["fixed_by_model"],
        "fixed_by_round": {r: stats[f"fixed_round{r}"] for r in range(1, CONFIG["repair_rounds"] + 1)},
        "valid": len(singles), "trimmed": stats["repaired"],
        "judge_votes": CONFIG["judge_votes"],
        "judge_vote_spread_avg": {c: (round(sum(v) / len(v), 2) if v else None) for c, v in (
            (c, [p["judge"]["spread"][c] for p in singles if p.get("judge") and p["judge"].get("votes", 1) > 1
                 and p["judge"]["spread"].get(c) is not None]) for c in CRITERIA)},
        "end_checked": stats["end_checked"], "end_trimmed": stats["end_trimmed"],
        "open_checked": stats["open_checked"], "open_trimmed": stats["open_trimmed"],
        "scoring_version": SCORING_VERSION, "input": input_ref(args.input),
        "episode_numbers": list(show.episodes),
        "rejected": stats["rejected"], "near_duplicates": stats["near_duplicates"],
        "cold_checked": len(checked), "stitched_valid": len(stitched),
        "blocked_by_safety_filter": llm.blocked,
        "stability": stability, "stopped": stopped,
    }
    write_outputs(out, show, ranked, post, rejected, summary, skipped)
    print(f"\nDone{' (PARTIAL)' if stopped else ''}. {len(ranked)} promos ranked, "
          f"{len(post)} picked to post first. Model calls this run: {summary['model_calls_made']}.")
    print(f"See {out}/report.md and {out}/promos_ranked.csv")
    if stopped:
        print("Run the same command again to continue; finished steps are reused.")
        sys.exit(1)


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

def seg_str(p: dict) -> str:
    return " | ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])


def write_outputs(out: Path, show: Show, ranked, post, rejected, summary, skipped=()):
    keep = ("rank", "post_order", "id", "type", "episodes", "segments", "duration_sec", "hook",
            "cliffhanger", "why", "repair", "opening", "ending", "judge", "judge_score", "cold_viewer", "final_score",
            "post_skip", "note")
    (out / "promos_ranked.json").write_text(json.dumps(
        [{k: p.get(k) for k in keep} for p in ranked], indent=1), encoding="utf-8")
    (out / "rejected_proposals.json").write_text(json.dumps(rejected, indent=1, default=str),
                                                 encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    with open(out / "promos_ranked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "post_first", "promo_id", "type", "episodes", "segments_in_play_order",
                    "duration_s", "hook", "cliffhanger", "why_it_works", "final_score", "judge_score",
                    "hook_0to5", "cliffhanger_0to5", "standalone_0to5", "flow_0to5", "premise_0to5",
                    "story_beat", "judge_note",
                    "stranger_comprehension_1to5", "stranger_intrigue_1to5", "stranger_thinks",
                    "stranger_wonders", "judge_votes", "opening_ending", "repair", "note",
                    "not_posted_because"])
        for p in ranked:
            j, c = p.get("judge") or {}, p.get("cold_viewer") or {}
            w.writerow([p["rank"], p.get("post_order", ""), p["id"], p["type"],
                        ",".join(map(str, p["episodes"])), seg_str(p), p["duration_sec"],
                        p["hook"], p["cliffhanger"], p["why"], p.get("final_score", ""),
                        p.get("judge_score", ""), j.get("hook", ""), j.get("cliffhanger", ""),
                        j.get("standalone", ""), j.get("flow", ""), j.get("premise", ""),
                        j.get("beat", ""), j.get("note", ""),
                        c.get("comprehension", ""), c.get("intrigue", ""),
                        c.get("what_i_think_is_happening", ""), c.get("question_left_with", ""),
                        j.get("votes", ""), "; ".join(x for x in (p.get("opening"), p.get("ending")) if x),
                        p.get("repair") or "", p.get("note", ""), p.get("post_skip", "")])

    s = summary
    L = [f"# Promo Finder results: {s['show']}", ""]
    if s["stopped"]:
        L += [f"> **PARTIAL RUN** - stopped early: {s['stopped']}", ""]
    L += [f"{s['episodes']} episodes | models: {', '.join(f'{m} x{c}' for m, c in s['models_used'].items()) or 'none (all cached)'} | "
          f"{s['model_calls_made']} model calls this run | {s['tokens_in'] + s['tokens_out']:,} tokens | "
          f"{s['wall_time_min']} min", "", "## Post these first", "",
          "Ranked by final score (judge 50%, stranger's intrigue 30%, stranger's comprehension 20%). "
          f"Judge scores are the average of {CONFIG['judge_votes']} votes. "
          f"A clip must meet the brief's must-haves: judge cliffhanger {CONFIG['min_cliffhanger_to_post']}+ "
          f"and hook {CONFIG['min_hook_to_post']}+, clear on its own (judge standalone "
          f"{CONFIG['min_standalone_to_post']}+), and a stranger understood it (comprehension "
          f"{CONFIG['min_comprehension_to_post']}+) and would stop for it (intrigue "
          f"{CONFIG['min_intrigue_to_post']}+). At most one per episode, at most "
          f"{CONFIG['max_per_beat']} of one story beat, no repeated footage, and never the "
          "same scene twice (an episode whose first scene is marked CONTINUOUS picks up the previous "
          "episode's last scene).", ""]
    if not post:
        L += ["**No clip passed every bar in this run**, so none is marked to post first. The closest ones, "
              "and the bar each one missed, are listed below.", ""]

    def n(x):
        return "?" if x is None else (f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}")
    for p in post:
        j, c = p["judge"], p["cold_viewer"]
        L += [f"### {p['post_order']}. {seg_str(p)} ({p['duration_sec']:.0f}s, {p['type']}) - score {p['final_score']:.2f}",
              f"- **Hook:** {p['hook']}",
              f"- **Cliffhanger:** {p['cliffhanger']}",
              f"- **Why it works:** {p['why']}",
              f"- **Judge:** hook {n(j.get('hook'))}, cliffhanger {n(j.get('cliffhanger'))}, "
              f"standalone {n(j.get('standalone'))}, flow {n(j.get('flow'))}, premise {n(j.get('premise'))} "
              f"(beat: {j.get('beat', 'other')}) - {j.get('note', '')}",
              f"- **Stranger** (comprehension {c['comprehension']:.0f}/5, intrigue {c['intrigue']:.0f}/5): "
              f"\"{c['what_i_think_is_happening']}\" Left wondering: \"{c['question_left_with']}\""]
        if p.get("repair"):
            L.append(f"- **Repaired:** {p['repair']}")
        if p.get("opening"):
            L.append(f"- **Opening:** {p['opening']}")
        if p.get("ending"):
            L.append(f"- **Ending:** {p['ending']}")
        L += ["", "<details><summary>Clip script</summary>", "", "```",
              show.render_clip(p["segments"]), "```", "</details>", ""]
    if skipped:
        L += ["### Skipped on the way down the ranking", ""]
        L += [f"- {p['id']} ({seg_str(p)}, score {p['final_score']:.2f}): {why}" for p, why in skipped]
        L.append("")

    final_reasons = Counter(reason_kind(r["reason"].split(";")[-1]) for r in rejected if "proposal" in r)
    first = s["first_pass_reasons"]
    L += ["## How the funnel went", "",
          f"- Proposals: {s['proposed']}. Passed the code's checks first time: "
          f"{s['proposed'] - s['first_pass_rejected']}. Sent back to the model with the exact problem: "
          f"{s['first_pass_rejected']}"
          + (f" ({', '.join(f'{k} x{v}' for k, v in first.items())})" if first else "")
          + f"; fixed by the model: {s['fixed_by_model']}"
          + (f" ({', '.join(f'{v} on try {r}' for r, v in s['fixed_by_round'].items())})"
             if len(s['fixed_by_round']) > 1 else "") + ".",
          f"- Final: {s['valid']} valid single-episode promos ({s['trimmed']} trimmed by the code to fit "
          f"60s), {s['rejected']} rejected"
          + (f" ({', '.join(f'{k} x{v}' for k, v in final_reasons.most_common())})" if final_reasons else "")
          + ".",
          f"- Near-duplicates skipped before the stranger check: {s['near_duplicates']}",
          f"- Judge: {s['judge_votes']} votes per clip. Average gap between a clip's highest and lowest vote: "
          + ", ".join(f"{c} {v}" for c, v in s["judge_vote_spread_avg"].items() if v is not None)
          + " (on a 0-5 scale; this is why one vote is not enough).",
          f"- Opening check: {s['open_checked']} finalists; {s['open_trimmed']} opened before their hook and "
          "were started there by the code (then judged again).",
          f"- Ending check: {s['end_checked']} finalists; {s['end_trimmed']} ran past their peak and "
          "were cut there by the code (then judged again).",
          f"- Stranger-checked: {s['cold_checked']} | stitched cross-episode promos: {s['stitched_valid']}",
          f"- Requests refused by the model's safety filter this run: {s['blocked_by_safety_filter']}"
          + (" (those steps are listed under Rejected proposals)" if s['blocked_by_safety_filter'] else ""),
          ""]

    checked = [p for p in ranked if p.get("final_score") is not None]

    def cv(p, k):
        return (p.get("cold_viewer") or {}).get(k)

    def flag(p):
        return (f"{p['id']} ({seg_str(p)}; judge {p['judge_score']:.2f}, stranger comprehension "
                f"{cv(p, 'comprehension') or '?'}, intrigue {cv(p, 'intrigue') or '?'})")
    lost = [p for p in checked if (cv(p, "comprehension") or 5) <= 2]
    flat = [p for p in checked if p["judge_score"] >= 4.0 and (cv(p, "intrigue") or 5) <= 2]
    missed = [p for p in checked if p["judge_score"] < 3.0 and (cv(p, "intrigue") or 0) >= 4]
    weak_fix = [p for p in ranked if p.get("repair") and p.get("judge") and (p["judge"].get("flow") or 5) <= 2]
    L += ["## Where it picks badly (auto-flagged)", ""]
    for label, items in (("The stranger was lost (comprehension 2 or lower)", lost),
                         ("The judge loved it (4+), but the stranger would scroll past (intrigue 2 or lower)", flat),
                         ("The stranger was hooked (intrigue 4+), but the judge scored it under 3", missed),
                         ("Trimmed or fixed cuts that lost their flow (flow 2 or lower)", weak_fix)):
        L.append(f"**{label}:** " + ("; ".join(flag(p) for p in items) if items else "none"))
        L.append("")
    if s["stability"]:
        st = s["stability"]
        L += ["## Stability", "",
              f"Proposals for the first {len(st['episodes'])} episodes were generated twice. "
              f"{st['reproduced']} of {st['picks']} first-run picks reappeared in the second run "
              "(50%+ footage overlap).", ""]
    L += ["## Rejected proposals", "", "| source | reason |", "|---|---|"]
    L += [f"| {r['source']} | {r['reason']} |" for r in rejected] or ["| - | none |"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Test your own cut (--evaluate): score a hand-picked cut exactly like the tool's own picks
# ----------------------------------------------------------------------------

def input_ref(path) -> str:
    """Where the scripts came from, relative to this folder when possible (the UI reuses it)."""
    p = Path(path).expanduser().resolve()
    try:
        return os.path.relpath(p, Path(__file__).resolve().parent)
    except ValueError:            # another drive on Windows
        return str(p)


def parse_time(t: str) -> float:
    """'01:37', '1:37.5', '97' or '00:01:37' -> seconds."""
    t = t.strip().replace(",", ".")
    if not re.fullmatch(r"\d+(?:\.\d+)?(?::\d+(?:\.\d+)?){0,2}", t):
        raise ValueError(f"'{t}' is not a time (use mm:ss, like 01:37)")
    secs = 0.0
    for x in t.split(":"):
        secs = secs * 60 + float(x)
    return secs


def parse_cut(spec: str, episodes) -> tuple[int, list[tuple[float, float]]]:
    """'101 01:37-02:18' (episode, then start-end; join pieces with +) -> (101, [(97.0, 138.0)])."""
    m = re.match(r"^\s*(?:ep(?:isode)?\s*)?(\d+)\s*[:|]?\s+(.*\S)\s*$", spec, re.I)
    if m and not re.match(r"^\s*\d+(?:[.:]\d+)*\s*(?:-|–|—|to)", spec):
        ep, rest = int(m.group(1)), m.group(2)
    elif len(episodes) == 1:
        ep, rest = next(iter(episodes)), spec
    else:
        raise ValueError(f'write a cut as episode, then start-end, like "101 01:37-02:18" (got "{spec}")')
    pieces = []
    for piece in rest.split("+"):
        r = re.fullmatch(r"\s*([\d:.,]+)\s*(?:-|–|—|to)\s*([\d:.,]+)\s*", piece)
        if not r:
            raise ValueError(f'"{piece.strip()}" is not a start-end range (like 01:37-02:18)')
        pieces.append((parse_time(r.group(1)), parse_time(r.group(2))))
    return ep, pieces


def cut_segment(show: Show, ep: int, t0: float, t1: float) -> dict:
    """The script lines that start inside [t0, t1], as a segment. Long camera or sound lines may run
    on past t1; the cut still ends at t1 (or at the end of a line of dialogue still being spoken)."""
    if ep not in show.episodes:
        raise ValueError(f"episode {ep} is not in these scripts (they have {', '.join(map(str, show.episodes))})")
    if t1 <= t0:
        raise ValueError(f"the end ({seconds_to_ts(t1)}) must come after the start ({seconds_to_ts(t0)})")
    if show.duration[ep] and t0 >= show.duration[ep]:
        raise ValueError(f"episode {ep} is only {seconds_to_ts(show.duration[ep])} long")
    rows = [r for r in show.rows[ep] if t0 - 0.3 <= r["start"] < t1 - 0.3]
    if not rows:
        raise ValueError(f"no script line starts between {seconds_to_ts(t0)} and {seconds_to_ts(t1)} "
                         f"in episode {ep}")
    sg = make_segment(show, ep, rows[0]["id"], rows[-1]["id"])
    spoken = max((r["end"] for r in rows if r["type"] == "DIALOGUE"), default=t1)
    sg["end_sec"] = round(max(sg["start_sec"] + 1, min(sg["end_sec"], max(t1, spoken))), 2)
    return sg


def pick_for_episode(out: Path, n: int):
    """The tool's own pick for episode n in this run (post-first if there is one), to compare against."""
    path = out / "promos_ranked.json"
    if not path.exists():
        return None
    try:
        ranked = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    mine = [p for p in ranked if p.get("type") == "single-episode" and p.get("episodes") == [n]
            and p.get("segments") and p.get("judge_score") is not None]
    mine.sort(key=lambda p: (p.get("post_order") is None, p.get("rank") or 999))
    if not mine:
        return None
    ref = {k: mine[0].get(k) for k in ("id", "type", "episodes", "segments", "duration_sec", "hook",
                                       "cliffhanger", "why", "post_order", "final_score")}
    ref["run_final_score"] = ref.pop("final_score")
    return ref


def evaluate_cuts(args):
    """Score the cuts given with --evaluate: same judge (same votes), same stranger, same quality bars.
    Each episode's cuts are judged in one call together with the tool's own pick for that episode,
    so the scores sit on the same scale and can be compared directly."""
    title, episodes = load_show(args.input)
    episodes = [e for e in episodes if e["n_events"] >= 3]
    if not episodes:
        sys.exit("No usable episodes in those scripts, so there is nothing to cut.")
    show = Show(title, episodes)
    out = Path(args.out)
    raw = out / ("raw_demo" if args.dry_run else "raw")   # never mix fake replies into the real cache
    raw.mkdir(parents=True, exist_ok=True)
    llm = LLM(CONFIG, out / "calls_log_evaluate.jsonl", fake=fake_llm if args.dry_run else None)
    first_ep = next(iter(show.episodes.values()))
    premise = f'"{title}". It begins: {first_ep["logline"]}'

    cuts, problems = [], []
    for spec in args.evaluate:
        try:
            ep, pieces = parse_cut(spec, show.episodes)
            segs = [cut_segment(show, ep, a, b) for a, b in pieces]
        except ValueError as e:
            problems.append(f"{spec}: {e}")
            continue
        fix_adjacent_overlaps(segs)
        for x in segs:
            x["start"], x["end"] = seconds_to_ts(x["start_sec"]), seconds_to_ts(x["end_sec"])
        cid = f"cut_ep{ep:03d}_" + "_".join(f"L{x['start_line']:03d}-L{x['end_line']:03d}" for x in segs)
        if any(c["id"] == cid for c in cuts):
            continue
        cuts.append({"id": cid, "type": "your cut", "episodes": [ep], "segments": segs,
                     "duration_sec": total_duration(segs), "requested": spec.strip()})
    for msg in problems:
        print(f"Can't use {msg}")
    if not cuts:
        sys.exit("No cut to test. Write each one as episode, then start-end, like: 101 01:37-02:18")

    print(f"Show: {title}. Testing {len(cuts)} cut(s): judge x{CONFIG['judge_votes']} votes, then a stranger.")
    refs = []
    try:
        for n in dict.fromkeys(c["episodes"][0] for c in cuts):
            mine = [c for c in cuts if c["episodes"][0] == n]
            ref = pick_for_episode(out, n)
            if ref and any(overlap_frac(ref, c) >= 0.9 for c in mine):
                ref = None                      # one of the cuts is the tool's pick already
            group = mine + ([ref] if ref else [])
            ep, (ep_title, ep_premise) = show.episodes[n], ep_meta(show, n, title, premise)
            key = hashlib.md5("|".join(c["id"] for c in group).encode()).hexdigest()[:8]
            print(f"Judging episode {n}: {len(mine)} cut(s)" + (f" next to the tool's pick {ref['id']}" if ref else ""))
            run_judge(llm, show, raw, f"eval_ep{n:03d}_{key}_judge", group,
                      context=f"{ep_premise} This is episode {n}: {ep['logline']}", title=ep_title)
            if ref:
                refs.append(ref)
        print("Stranger check ...")
        for c in cuts + refs:
            if c.get("judge_score") is not None:
                run_cold_viewer(llm, show, raw, c)
    except (LLMError, KeyboardInterrupt, ValueError) as e:
        print(f"\nSTOPPED: {'stopped by user' if isinstance(e, KeyboardInterrupt) else e}")
        sys.exit(1)

    when = time.strftime("%Y-%m-%d %H:%M")
    new = []
    for c in cuts:
        ref = next((r for r in refs if r["episodes"] == c["episodes"]), None)
        bars = quality_failures(c) if c.get("final_score") is not None else ["not scored"]
        if not CONFIG["min_sec"] <= c["duration_sec"] <= CONFIG["max_sec"]:
            bars.insert(0, f"{c['duration_sec']:.0f}s long (promos run {CONFIG['min_sec']:.0f}-"
                           f"{CONFIG['max_sec']:.0f}s)")
        new.append({k: c.get(k) for k in ("id", "requested", "episodes", "segments", "duration_sec", "judge",
                                          "judge_score", "cold_viewer", "final_score")} | {
            "good_enough_to_post": not bars, "misses": bars, "clip": show.render_clip(c["segments"]),
            "compared_with": ({k: ref.get(k) for k in ("id", "segments", "duration_sec", "judge", "judge_score",
                                                       "cold_viewer", "final_score", "run_final_score",
                                                       "post_order", "hook")} if ref else None),
            "demo": bool(args.dry_run), "scoring_version": SCORING_VERSION, "when": when})

    path = out / "evaluated_cuts.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except ValueError:
        old = []
    done = {(e["id"], e["demo"]) for e in new}
    keep = new + [e for e in old if isinstance(e, dict) and (e.get("id"), e.get("demo")) not in done]
    path.write_text(json.dumps(keep, indent=1), encoding="utf-8")

    L = [f"# Your cuts: {title}", "",
         "Scored exactly like the tool's own picks: the same judge (votes averaged), the same stranger, "
         "the same quality bars. Each episode's cuts were judged in one call next to the tool's own pick, "
         "so those scores are directly comparable.", "",
         "| Cut | Plays | Length | Final | Judge | Hook | Cliff | Standalone | Stranger (understood / hooked) | Post it? |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    def row(label, x, verdict):
        j, cv = x.get("judge") or {}, x.get("cold_viewer") or {}
        plays = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in x.get("segments") or [])
        return (f"| {label} | {plays} | {x.get('duration_sec') or 0:.0f}s | {x.get('final_score')} | "
                f"{x.get('judge_score')} | {j.get('hook')} | {j.get('cliffhanger')} | {j.get('standalone')} | "
                f"{cv.get('comprehension')} / {cv.get('intrigue')} | {verdict} |")
    for k, e in enumerate(keep):
        L.append(row(e.get("requested", e["id"]) + (" (demo)" if e.get("demo") else ""), e,
                     "yes" if e.get("good_enough_to_post") else "no: " + "; ".join(e.get("misses") or [])))
        r = e.get("compared_with")
        nxt = (keep[k + 1].get("compared_with") or {}) if k + 1 < len(keep) else {}
        if r and (nxt.get("id"), nxt.get("judge_score")) != (r.get("id"), r.get("judge_score")):
            L.append(row(f"*tool's pick {r['id']}*", r, f"*post #{r['post_order']} in this run*"
                         if r.get("post_order") else "*its best cut in this run*"))
    (out / "evaluated_cuts.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print()
    for e in new:
        j, cv = e.get("judge") or {}, e.get("cold_viewer") or {}
        plays = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in e["segments"])
        print(f"{e['requested']}  ->  plays {plays} ({e['duration_sec']:.0f}s)")
        print(f"   final {e.get('final_score')} | judge {e.get('judge_score')} (hook {j.get('hook')}, cliffhanger "
              f"{j.get('cliffhanger')}, standalone {j.get('standalone')}, flow {j.get('flow')}, premise "
              f"{j.get('premise')}) | stranger: understood {cv.get('comprehension')}, hooked {cv.get('intrigue')}")
        print("   good enough to post" if e["good_enough_to_post"] else "   would not be posted: " + "; ".join(e["misses"]))
    for r in refs:
        plays = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in r["segments"])
        print(f"Tool's pick {r['id']} ({plays}), in the same judge call: final {r.get('final_score')} | "
              f"judge {r.get('judge_score')}")
    print(f"\nDone. {len(new)} cut(s) tested. See {out}/evaluated_cuts.md")


# ----------------------------------------------------------------------------
# Fake model for --dry-run (tests every code path without an API key)
# ----------------------------------------------------------------------------

def fake_llm(prompt: str) -> str:
    rng = random.Random(int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16))
    ts = r"(\d+:\d{2}\.\d{2})-(\d+:\d{2}\.\d{2})"

    def secs(t):
        m, s = t.split(":")
        return int(m) * 60 + float(s)

    if "an automatic check rejected" in prompt:
        rows = [(int(a), secs(b), secs(c)) for a, b, c in re.findall(rf"\[L(\d+)\] {ts}", prompt)]
        n = len(rows)
        out = []
        for rid in re.findall(r"^(R\d+):", prompt, re.M):
            i0 = rng.randrange(0, max(1, n // 2))
            target = rng.uniform(25, 40) if rng.random() > 0.15 else 8  # some fixes still fail
            j = i0
            while j + 1 < n and rows[j][2] - rows[i0][1] < target:
                j += 1
            out.append({"fixes": rid, "segments": [{"start_line": f"L{rows[i0][0]:03d}",
                                                    "end_line": f"L{rows[j][0]:03d}"}],
                        "hook": "fake fixed hook", "cliffhanger": "fake", "why": "fake"})
        return json.dumps(out)
    if "best promo cuts from this episode" in prompt:
        if "Episode 7." in prompt and "IMPORTANT: reply with valid JSON" not in prompt:
            return "Sure! Here are the best promo moments from this episode, in my opinion."
        rows = [(int(a), secs(b), secs(c)) for a, b, c in re.findall(rf"\[L(\d+)\] {ts}", prompt)]
        n = len(rows)

        def cut(i0, length):
            j = i0
            while j + 1 < n and rows[j][2] - rows[i0][1] < length:
                j += 1
            return {"segments": [{"start_line": rows[i0][0], "end_line": rows[j][0]}],
                    "hook": "fake hook", "cliffhanger": "fake cliffhanger", "why": "fake reason"}
        out = [cut(rng.randrange(0, max(1, n // 2)), rng.uniform(28, 45)),
               cut(rng.randrange(0, max(1, n // 3)), rng.uniform(70, 95))]
        roll = rng.random()
        if roll < 0.25:
            out.append({"segments": [{"start_line": n - 2, "end_line": n + 30}]})
        elif roll < 0.45:
            out.append(cut(rng.randrange(n // 2, n - 1), 6))
        else:
            out.append(cut(rng.randrange(n // 2, max(n // 2 + 1, n - 10)), rng.uniform(25, 40)))
        txt = json.dumps(out)
        return f"```json\n{txt}\n```" if rng.random() < 0.3 else txt
    if "harsh, experienced reviewer" in prompt:
        arr = [{"clip": c, "hook": rng.randint(1, 5), "cliffhanger": rng.randint(1, 5),
                "standalone": rng.randint(1, 5), "flow": rng.randint(1, 5),
                "premise": rng.randint(1, 5), "beat": rng.choice(BEATS + ("Danger ", "unknown")),
                "note": "fake note"}
               for c in re.findall(r"--- CLIP (C\d+)", prompt)]
        return json.dumps({"scores": arr} if rng.random() < 0.2 else arr)
    if "START so its first 3 seconds hook a stranger" in prompt:
        ids = re.findall(r"^\[L(\d+)\]", prompt, re.M)
        pick = ids[0] if rng.random() < 0.5 or len(ids) < 4 else ids[rng.randint(1, min(3, len(ids) - 1))]
        return json.dumps({"start_line": f"L{pick}", "hook": "fake opening", "reason": "fake"})
    if "stops at its peak, before the tension is released" in prompt:
        ids = re.findall(r"^\[L(\d+)\]", prompt, re.M)
        pick = ids[-1] if rng.random() < 0.5 or len(ids) < 4 else ids[-rng.randint(2, 4)]
        return json.dumps({"end_line": f"L{pick}", "cliffhanger": "fake peak", "reason": "fake"})
    if "scrolling Instagram Reels" in prompt:
        return json.dumps({"what_i_think_is_happening": "fake guess", "comprehension": rng.randint(1, 5),
                           "intrigue": rng.randint(1, 5), "question_left_with": "fake question?"})
    if "JOIN moments from DIFFERENT episodes" in prompt:
        lines = [(int(e), int(l)) for e, l in re.findall(r"\[Ep(\d+) L(\d+)\]", prompt)]
        eps = list(dict.fromkeys(e for e, _ in lines))
        if len(eps) < 2:
            return "[]"
        a = [l for e, l in lines if e == eps[0]]
        b = [l for e, l in lines if e == eps[1]]
        return json.dumps([
            {"segments": [{"episode": eps[0], "start_line": a[0], "end_line": a[min(10, len(a) - 1)]},
                          {"episode": eps[1], "start_line": b[max(0, len(b) - 12)], "end_line": b[-1]}],
             "hook": "fake", "cliffhanger": "fake", "why": "fake"},
            {"segments": [{"episode": eps[0], "start_line": a[0], "end_line": a[-1]},
                          {"episode": eps[1], "start_line": b[0], "end_line": b[-1]}],
             "hook": "fake (too long)", "cliffhanger": "fake", "why": "fake"},
            {"segments": [{"episode": eps[0], "start_line": 999, "end_line": 1005},
                          {"episode": eps[1], "start_line": b[0], "end_line": b[-1]}],
             "hook": "fake (bad lines)", "cliffhanger": "fake", "why": "fake"}])
    return "{}"


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Find the best promo cuts in microdrama screenplays.")
    ap.add_argument("input", nargs="?", help="folder (or .zip) containing the episode .txt files")
    ap.add_argument("--out", help="output folder (default: outputs, or outputs_dryrun with --dry-run)")
    ap.add_argument("--limit", type=int, help="only the first N episodes (quick test)")
    ap.add_argument("--stability", type=int, default=0,
                    help="propose again for the first N episodes and measure how stable picks are")
    ap.add_argument("--finalists", type=int, default=CONFIG["finalists"],
                    help="how many top clips get the stranger (cold-viewer) check")
    ap.add_argument("--judge-votes", type=int, default=CONFIG["judge_votes"],
                    help="how many times the judge scores each clip (scores are averaged)")
    ap.add_argument("--evaluate", action="append", metavar='"EP START-END"',
                    help='score your own cut like the tool\'s picks, e.g. --evaluate "101 01:37-02:18" '
                         "(repeat for more cuts; use the run's --out folder to compare with its picks)")
    ap.add_argument("--dry-run", action="store_true", help="fake model replies: test with no API key")
    ap.add_argument("--fresh", action="store_true", help="ignore saved replies and start over")
    ap.add_argument("--ping", action="store_true", help="check which models in LLM_MODELS answer right now")
    ap.add_argument("--list-models", action="store_true", help="list the models your API key can use")
    a = ap.parse_args()

    if a.ping:
        ping(CONFIG)
        sys.exit(0)
    if a.list_models:
        list_models(CONFIG)
        sys.exit(0)
    if not a.input:
        ap.error("give the folder (or .zip) with the episode scripts")
    a.out = a.out or ("outputs_dryrun" if a.dry_run else "outputs")
    CONFIG["judge_votes"] = max(1, a.judge_votes)
    try:
        if a.evaluate:
            evaluate_cuts(a)
            sys.exit(0)
        run(a)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(2)
