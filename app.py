#!/usr/bin/env python3
"""Promo Finder UI: a local web app around pipeline.py.

    python3 app.py            then open http://127.0.0.1:8000 (it opens by itself)

Standard library only, nothing to install. Each run is pipeline.py in a subprocess, so a failed
run never takes the UI down. An API key typed in the UI goes to that one run as an environment
variable and is never written to disk. The server listens on this computer only (127.0.0.1).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
POSTER_DIR = ROOT / "posters"
RUNS_DIR = ROOT / "outputs_ui"          # matches .gitignore (outputs*/): scripts and results stay local
UPLOAD_DIR = RUNS_DIR / "uploads"
RESULT_FILES = {"promos_ranked.csv", "promos_ranked.json", "report.md", "run_summary.json",
                "evaluated_cuts.md", "evaluated_cuts.json"}
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
               ".gif": "image/gif", ".avif": "image/avif"}
DEFAULT_MODELS = "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.8-flash"

_m = re.search(r"^SCORING_VERSION = (\d+)", (ROOT / "pipeline.py").read_text(encoding="utf-8"), re.M)
SCORING_NOW = int(_m.group(1)) if _m else 0
SCORING_TEXT = {
    8: "5-part judge, votes averaged, opening + ending checks, clear-on-its-own bar",
    7: "5-part judge, votes averaged, ending check",
}

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- saved runs
def _safe_run_dir(rel: str) -> Path | None:
    """Only folders named outputs* inside this project (or outputs_ui/<run>) can be read."""
    if not rel:
        return None
    p = (ROOT / unquote(rel)).resolve()
    try:
        parts = p.relative_to(ROOT).parts
    except ValueError:
        return None
    if not parts or not parts[0].startswith("outputs") or not p.is_dir():
        return None
    return p


def scoring_of(s: dict) -> dict:
    """Which scoring a run used. Scores are only comparable between runs with the same version."""
    v = s.get("scoring_version")
    if v is None:                      # written down from v8 on; v7 was the first with judge votes
        v = 7 if "judge_votes" in s else 6
    return {"version": v, "current": v == SCORING_NOW, "earlier": v < 7, "demo": s.get("provider") == "dry-run",
            "text": SCORING_TEXT.get(v, "an earlier judge: one vote, fewer checks and bars. Its scores are on "
                                        "a different scale, so they can't be compared with v7+ runs")}


def list_runs() -> list[dict]:
    found, seen = [], {}
    candidates = [d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("outputs")]
    if RUNS_DIR.is_dir():
        candidates += [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    for d in candidates:
        summ, ranked = d / "run_summary.json", d / "promos_ranked.json"
        if not (summ.exists() and ranked.exists()):
            continue
        try:
            body = ranked.read_bytes()
            s = json.loads(summ.read_text(encoding="utf-8"))
            promos = json.loads(body)
        except (OSError, ValueError):
            continue
        scores = [p["final_score"] for p in promos if p.get("final_score") is not None]
        first = next((p for p in promos if p.get("post_order") == 1), None)
        found.append({
            "dir": d.relative_to(ROOT).as_posix(), "show": s.get("show", "?"),
            "episodes": s.get("episodes"), "valid": s.get("valid"),
            "post_first": sum(1 for p in promos if p.get("post_order")),
            "top": first.get("final_score") if first else (max(scores) if scores else None),
            "top_is_pick": first is not None,
            "calls": s.get("model_calls_made"), "minutes": s.get("wall_time_min"),
            "provider": s.get("provider"), "modified": int(ranked.stat().st_mtime),
            "scoring": scoring_of(s), "md5": hashlib.md5(body).hexdigest(),
        })
    found.sort(key=lambda r: (r["dir"] != "outputs_final", -r["modified"]))
    for r in found:                    # an exact copy of another run (same results file) is not a new result
        first = seen.setdefault(r.pop("md5"), r["dir"])
        r["copy_of"] = first if first != r["dir"] else None
    return found


def clip_scripts(report: Path) -> dict:
    """Pull each post-first pick's clip script out of report.md, keyed by post order."""
    if not report.exists():
        return {}
    text = report.read_text(encoding="utf-8")
    if "## Post these first" not in text:
        return {}
    block = text.split("## Post these first", 1)[1].split("### Skipped", 1)[0].split("\n## ", 1)[0]
    out = {}
    for part in re.split(r"\n### ", block)[1:]:
        m = re.match(r"(\d+)\.", part)
        code = re.search(r"```\n(.*?)```", part, re.S)
        if m and code:
            out[int(m.group(1))] = code.group(1).rstrip()
    return out


def load_result(rel: str) -> dict | None:
    d = _safe_run_dir(rel)
    if d is None or not (d / "promos_ranked.json").exists():
        return None
    summary = {}
    if (d / "run_summary.json").exists():
        summary = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
    promos = json.loads((d / "promos_ranked.json").read_text(encoding="utf-8"))
    evaluated = []
    if (d / "evaluated_cuts.json").exists():
        try:
            evaluated = json.loads((d / "evaluated_cuts.json").read_text(encoding="utf-8"))
        except ValueError:
            evaluated = []
    eps = summary.get("episode_numbers") or sorted({e for p in promos for e in p.get("episodes") or []})
    return {"dir": d.relative_to(ROOT).as_posix(), "summary": summary, "promos": promos,
            "clips": clip_scripts(d / "report.md"), "scoring": scoring_of(summary), "episode_numbers": eps,
            "scripts_known": bool(summary.get("input")), "evaluated": evaluated}


# ----------------------------------------------------------------------------- running a job
STAGE_PATTERNS = [
    (re.compile(r"^Opening check on"), "opening", 0.82),
    (re.compile(r"^Ending check on"), "ending", 0.85),
    (re.compile(r"^Judging episode"), "judge", 0.3),
    (re.compile(r"^Stranger check"), "viewer", 0.7),
    (re.compile(r"^Cold-viewer check on"), "viewer", 0.89),
    (re.compile(r"^Stitching cross-episode"), "stitch", 0.94),
    (re.compile(r"^Stability check on"), "stability", 0.97),
]


def _track(job: dict, line: str) -> None:
    job["log"].append(line)
    if len(job["log"]) > 600:
        del job["log"][:100]
    m = re.match(r"^Show: (.*) \((\d+) episodes\)", line)
    if m:
        job["show"], job["total"] = m.group(1), int(m.group(2))
        job["stage"], job["progress"] = "propose", 0.03
        return
    m = re.match(r"^Expect about (\d+) model calls, roughly (\d+) min", line)
    if m:
        job["eta_min"] = int(m.group(2))
        return
    m = re.match(r"^Ep\s+\d+ \((\d+)/(\d+)\)", line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        job["done_eps"], job["total"] = done, total
        job["stage"], job["progress"] = "propose", 0.03 + 0.78 * done / max(total, 1)
        return
    for pat, stage, prog in STAGE_PATTERNS:
        if pat.match(line):
            job["stage"], job["progress"] = stage, prog
            return
    if line.startswith("STOPPED:"):
        job["error"] = line[len("STOPPED:"):].strip()
    if line.startswith("Done"):
        job["stage"], job["progress"] = "done", 1.0
        job["partial"] = "(PARTIAL)" in line


def _worker(job: dict, cmd: list[str], env: dict) -> None:
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
    except OSError as e:
        job.update(state="error", error=str(e), ended=time.time())
        return
    job["proc"] = proc
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if line.strip():
            _track(job, line)
    code = proc.wait()
    job["ended"] = time.time()
    if job.get("stop_requested"):
        job["state"] = "stopped"
    elif code == 0 and job.get("stage") == "done":
        job["state"] = "partial" if job.get("partial") else "done"
    else:
        job["state"] = "error"
        if not job.get("error"):
            tail = [l for l in job["log"][-8:] if l.strip()]
            job["error"] = tail[-1] if tail else f"the run exited with code {code}"


def _busy() -> str | None:
    with JOBS_LOCK:
        if any(j["state"] == "running" for j in JOBS.values()):
            return "A run is already going. Stop it first, or wait for it to finish."
    return None


def _scripts_path(src: str) -> tuple[Path | None, str | None]:
    src = str(src or "").strip().strip('"').strip("'")
    if not src:
        return None, "Add the scripts first: drop a .zip, or paste the path of a folder or .zip."
    p = Path(os.path.expanduser(src))
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return None, f"Can't find {p}. Check the path."
    return p, None


def _env_for(opts: dict) -> tuple[dict | None, str | None]:
    """Environment for one run: provider, models and the key go to that process only."""
    dry = bool(opts.get("dry_run"))
    key = str(opts.get("api_key") or "").strip()
    env = dict(os.environ)
    if not dry and not key and not (env.get("LLM_API_KEY") or env.get("GEMINI_API_KEY")):
        return None, "No API key: paste one in the API key box, or turn on Demo mode."
    provider = str(opts.get("provider") or "gemini").strip().lower()
    env["LLM_PROVIDER"] = provider
    if str(opts.get("models") or "").strip():
        env["LLM_MODELS"] = str(opts["models"]).strip()
    if provider == "openai" and str(opts.get("base_url") or "").strip():
        env["LLM_BASE_URL"] = str(opts["base_url"]).strip()
    if key:
        env["LLM_API_KEY"] = key           # this run only; never written anywhere
    env["LLM_MIN_INTERVAL"] = "7" if opts.get("free_tier", True) else "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env, None


def _launch(cmd: list[str], env: dict, out: Path, dry: bool, kind: str) -> dict:
    job = {"id": uuid.uuid4().hex[:10], "state": "running", "stage": "start", "progress": 0.01,
           "log": [], "out": out.relative_to(ROOT).as_posix(), "started": time.time(), "ended": None,
           "done_eps": 0, "total": None, "show": None, "eta_min": None, "error": None, "dry_run": dry,
           "kind": kind}
    with JOBS_LOCK:
        JOBS[job["id"]] = job
    threading.Thread(target=_worker, args=(job, cmd, env), daemon=True).start()
    return job


def start_job(opts: dict) -> tuple[dict | None, str | None]:
    err = _busy()
    if err:
        return None, err
    src_path, err = _scripts_path(opts.get("path"))
    if err:
        return None, err
    env, err = _env_for(opts)
    if err:
        return None, err
    dry = bool(opts.get("dry_run"))
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"run_{run_id}"
    cmd = [sys.executable, "-u", str(ROOT / "pipeline.py"), str(src_path), "--out", str(out)]
    if opts.get("limit"):
        cmd += ["--limit", str(int(opts["limit"]))]
    cmd += ["--judge-votes", str(max(1, min(5, int(opts.get("judge_votes") or 3))))]
    if opts.get("stability"):
        cmd += ["--stability", "3"]
    if dry:
        cmd.append("--dry-run")
    return _launch(cmd, env, out, dry, "run"), None


def start_evaluate(opts: dict) -> tuple[dict | None, str | None]:
    """Score the user's own cuts inside a saved run (pipeline.py --evaluate), next to its picks."""
    err = _busy()
    if err:
        return None, err
    d = _safe_run_dir(str(opts.get("dir") or ""))
    if d is None:
        return None, "Open a saved run first."
    summary = {}
    if (d / "run_summary.json").exists():
        summary = json.loads((d / "run_summary.json").read_text(encoding="utf-8"))
    src = str(opts.get("path") or "").strip() or summary.get("input") or ""
    if not src:
        return None, "This run doesn't say where its scripts are. Paste the scripts folder or .zip path."
    src_path, err = _scripts_path(src)
    if err:
        return None, err
    cuts = [str(c).strip() for c in (opts.get("cuts") or []) if str(c).strip()][:6]
    if not cuts:
        return None, "Type a cut first: episode, start and end."
    env, err = _env_for(opts)
    if err:
        return None, err
    dry = bool(opts.get("dry_run"))
    cmd = [sys.executable, "-u", str(ROOT / "pipeline.py"), str(src_path), "--out", str(d),
           "--judge-votes", str(max(1, min(5, int(opts.get("judge_votes") or 3))))]
    for c in cuts:
        cmd += ["--evaluate", c]
    if dry:
        cmd.append("--dry-run")
    return _launch(cmd, env, d, dry, "evaluate"), None


def job_view(job: dict) -> dict:
    end = job["ended"] or time.time()
    return {k: job.get(k) for k in ("id", "state", "stage", "progress", "out", "done_eps", "total", "show",
                                     "eta_min", "error", "dry_run", "kind")} | {
        "elapsed": round(end - job["started"], 1), "log": job["log"][-250:]}


# ----------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "PromoFinder/1.0"

    def log_message(self, fmt, *args):   # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, path: Path, ctype: str, download: str | None = None):
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        extra = {"Content-Disposition": f'attachment; filename="{download}"'} if download else None
        self._send(200, path.read_bytes(), ctype, extra)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/", "/index.html"):
            return self._file(UI_DIR / "index.html", "text/html; charset=utf-8")
        if url.path == "/api/info":
            return self._json({
                "key_set": bool(os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")),
                "provider": os.environ.get("LLM_PROVIDER", "gemini"),
                "models": os.environ.get("LLM_MODELS", DEFAULT_MODELS),
                "runs": list_runs(),
                "running": next((j["id"] for j in JOBS.values() if j["state"] == "running"), None),
            })
        if url.path == "/api/result":
            res = load_result(q.get("dir", ""))
            return self._json(res) if res else self._json({"error": "run not found"}, 404)
        if url.path == "/api/status":
            job = JOBS.get(q.get("job", ""))
            return self._json(job_view(job)) if job else self._json({"error": "no such run"}, 404)
        if url.path == "/api/file":
            d, name = _safe_run_dir(q.get("dir", "")), q.get("name", "")
            if d is None or name not in RESULT_FILES:
                return self._json({"error": "not allowed"}, 403)
            ctype = {"csv": "text/csv", "json": "application/json", "md": "text/markdown"}[name.rsplit(".", 1)[1]]
            return self._file(d / name, ctype + "; charset=utf-8", download=f"{d.name}_{name}")
        if url.path == "/api/posters":
            files = sorted(p.name for p in POSTER_DIR.glob("*") if p.suffix.lower() in IMAGE_TYPES) \
                if POSTER_DIR.is_dir() else []
            return self._json({"posters": files})
        if url.path.startswith("/posters/"):
            p = (POSTER_DIR / unquote(url.path[len("/posters/"):])).resolve()
            if p.parent != POSTER_DIR.resolve() or p.suffix.lower() not in IMAGE_TYPES:
                return self._json({"error": "not allowed"}, 403)
            return self._file(p, IMAGE_TYPES[p.suffix.lower()])
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        length = int(self.headers.get("Content-Length") or 0)
        if url.path == "/api/upload":
            if length <= 0 or length > 300 * 1024 * 1024:
                return self._json({"error": "send a .zip or .txt file under 300 MB"}, 400)
            raw_name = Path(unquote(self.headers.get("X-Filename", "scripts.zip"))).name
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name) or "file"
            remaining, chunks = length, []
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            if name.lower().endswith(".zip"):
                dest = UPLOAD_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
                dest.write_bytes(data)
                return self._json({"path": str(dest), "kind": "zip", "name": name, "bytes": len(data)})
            if name.lower().endswith(".txt"):
                batch = self.headers.get("X-Batch", "")
                if not re.fullmatch(r"[A-Za-z0-9_-]{6,40}", batch):
                    return self._json({"error": "missing upload batch id"}, 400)
                folder = UPLOAD_DIR / f"scripts_{batch}"
                folder.mkdir(parents=True, exist_ok=True)
                dest, k = folder / name, 1
                while dest.exists():                      # same name from two folders: keep both
                    dest, k = folder / f"{Path(name).stem}_{k}{Path(name).suffix}", k + 1
                dest.write_bytes(data)
                return self._json({"path": str(folder), "kind": "folder", "name": name, "bytes": len(data),
                                   "count": len(list(folder.glob("*.txt")))})
            return self._json({"error": "only .zip or .txt script files"}, 400)
        body = self.rfile.read(length) if length else b"{}"
        try:
            opts = json.loads(body or b"{}")
        except ValueError:
            return self._json({"error": "bad request"}, 400)
        if url.path in ("/api/run", "/api/evaluate"):
            job, err = (start_job if url.path == "/api/run" else start_evaluate)(opts)
            return self._json({"error": err}, 400) if err else self._json(job_view(job))
        if url.path == "/api/stop":
            job = JOBS.get(q.get("job", "") or opts.get("job", ""))
            if not job:
                return self._json({"error": "no such run"}, 404)
            if job["state"] == "running" and job.get("proc"):
                job["stop_requested"] = True
                job["proc"].terminate()
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = None
    for p in range(port, port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        sys.exit("No free port between 8000 and 8009. Close the other app and try again.")
    url = f"http://127.0.0.1:{port}"
    print(f"Promo Finder UI is running at {url}  (press Control + C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        for j in JOBS.values():
            if j["state"] == "running" and j.get("proc"):
                j["proc"].terminate()


if __name__ == "__main__":
    main()
