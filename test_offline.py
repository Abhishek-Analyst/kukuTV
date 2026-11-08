"""
Offline test of the model client against a local fake API server.
Checks the real HTTP code paths (native Gemini format and OpenAI-compatible format)
and the failure handling: overload (503), missing model (404), daily quota (429),
JSON mode refused (400), rate limit with retryDelay, empty replies.

Run:  python3 test_offline.py "<folder with episode scripts>"
"""
import json
import sys
import tempfile
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pipeline

REQUESTS = []
FLAKY = {"n": 0}


def gemini_ok(text):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 200,
                              "thoughtsTokenCount": 50, "totalTokenCount": 1250}}


def openai_ok(text):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1beta/models"):
            assert self.headers.get("x-goog-api-key") == "test-key"
            self._send(200, {"models": [
                {"name": "models/good-model", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/embed-model", "supportedGenerationMethods": ["embedContent"]}]})
        elif self.path.startswith("/v1/models"):
            assert self.headers.get("Authorization") == "Bearer test-key"
            self._send(200, {"data": [{"id": "good-model"}, {"id": "flaky-model"}]})
        else:
            self._send(404, {"error": {"message": "no route"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/v1beta/models/"):
            model = self.path.split("/v1beta/models/")[1].split(":")[0]
            assert self.headers.get("x-goog-api-key") == "test-key", "missing Gemini key header"
            prompt = body["contents"][0]["parts"][0]["text"]
            json_mode = body.get("generationConfig", {}).get("responseMimeType") == "application/json"
            REQUESTS.append(("gemini", model, json_mode))
            fmt = gemini_ok
        elif self.path == "/v1/chat/completions":
            model = body["model"]
            assert self.headers.get("Authorization") == "Bearer test-key", "missing Bearer key"
            prompt = body["messages"][0]["content"]
            json_mode = False
            REQUESTS.append(("openai", model, False))
            fmt = openai_ok
        else:
            return self._send(404, {"error": {"message": "no route"}})

        if model == "busy-model":
            return self._send(503, {"error": {"code": 503, "message": "The model is overloaded."}})
        if model == "missing-model":
            return self._send(404, {"error": {"code": 404, "message": f"models/{model} is not found"}})
        if model == "daily-quota-model":  # realistic body: the "PerDay" marker sits deep inside
            return self._send(429, {"error": {
                "code": 429, "status": "RESOURCE_EXHAUSTED", "message":
                "You exceeded your current quota, please check your plan and billing details. For more "
                "information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. "
                "To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit. \n* Quota "
                "exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                "limit: 500, model: daily-quota-model",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.Help", "links": [
                        {"description": "Learn more about Gemini API quotas",
                         "url": "https://ai.google.dev/gemini-api/docs/rate-limits"}]},
                    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                        {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                         "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                         "quotaDimensions": {"location": "global", "model": "daily-quota-model"},
                         "quotaValue": "500"}]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "43200s"}]}})
        if model == "broke-model":  # OpenAI's out-of-credit reply
            return self._send(429, {"error": {"message": "You exceeded your current quota, please check your "
                              "plan and billing details.", "type": "insufficient_quota",
                              "code": "insufficient_quota"}})
        if model == "nojson-model" and json_mode:
            return self._send(400, {"error": {"code": 400, "message": "JSON mode is not supported"}})
        if model == "flaky-model":
            FLAKY["n"] += 1
            if FLAKY["n"] == 1:
                return self._send(503, {"error": {"code": 503, "message": "overloaded"}})
            if FLAKY["n"] in (2, 3, 4):
                return self._send(429, {"error": {"code": 429, "message": "slow down", "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"}]}})
            if FLAKY["n"] == 5:
                return self._send(200, fmt(""))  # empty reply
        if "best promo cuts" in prompt and "Episode 2." in prompt and fmt is gemini_ok:
            return self._send(200, {"candidates": [{"index": 0, "finishReason": "SAFETY"}],
                                    "usageMetadata": {"promptTokenCount": 1000}})
        if "best promo cuts" in prompt and "Episode 3." in prompt and fmt is openai_ok:
            return self._send(200, {"choices": [{"index": 0, "finish_reason": "content_filter",
                                                 "message": {"role": "assistant", "content": None}}]})
        return self._send(200, fmt(pipeline.fake_llm(prompt)))


def main(pack):
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    pipeline.time.sleep = lambda s: None  # skip back-off waits in the test

    base = dict(pipeline.CONFIG, api_key="test-key", min_interval_sec=0)
    out_root = Path(tempfile.mkdtemp(prefix="promo_test_"))

    # 1) native Gemini format: every failure mode, ending on a model that refuses JSON mode
    pipeline.CONFIG.update(base, provider="gemini", base_url=f"http://127.0.0.1:{port}/v1beta",
                           models=["busy-model", "missing-model", "daily-quota-model",
                                   "nojson-model", "good-model"])
    args = Namespace(input=pack, out=str(out_root / "gemini"), limit=4, stability=1,
                     finalists=6, dry_run=False, fresh=False)
    try:
        pipeline.run(args)
    except SystemExit as e:
        assert e.code in (None, 0), f"run exited with {e.code}"
    s = json.loads((out_root / "gemini" / "run_summary.json").read_text())
    print("\n[native gemini] models used:", s["models_used"], "| tokens:", s["tokens_in"], s["tokens_out"])
    assert list(s["models_used"]) == ["nojson-model"], "fallback chain did not land on nojson-model"
    assert ("gemini", "nojson-model", False) in REQUESTS, "JSON mode was not switched off"
    assert s["tokens_in"] > 0 and s["valid"] > 0 and s["cold_checked"] > 0
    counts = {m: sum(1 for r in REQUESTS if r[1] == m) for m in
              ["busy-model", "missing-model", "daily-quota-model", "nojson-model"]}
    print("Calls per model:", counts)
    assert counts["missing-model"] == 1, "a missing model should be dropped after one try"
    assert counts["daily-quota-model"] == 1, "daily quota (marker deep in the body) not recognised"
    assert counts["busy-model"] == 3, "an overloaded model should cool down, not be retried every call"
    rej = json.loads((out_root / "gemini" / "rejected_proposals.json").read_text())
    assert any(r["source"] == "ep002" and "safety filter" in r["reason"] for r in rej), \
        "safety-blocked episode should be skipped and recorded"
    assert s["blocked_by_safety_filter"] == 1

    # 2) OpenAI-compatible format: overload, rate limit with retryDelay, empty reply, recovery
    REQUESTS.clear()
    pipeline.CONFIG.update(base, provider="openai", base_url=f"http://127.0.0.1:{port}/v1",
                           models=["flaky-model", "good-model"])
    args.out = str(out_root / "openai")
    try:
        pipeline.run(args)
    except SystemExit as e:
        assert e.code in (None, 0), f"run exited with {e.code}"
    s = json.loads((out_root / "openai" / "run_summary.json").read_text())
    print("\n[openai-compatible] models used:", s["models_used"])
    assert list(s["models_used"]) == ["flaky-model"], "flaky model should recover, not be dropped"
    rej = json.loads((out_root / "openai" / "rejected_proposals.json").read_text())
    assert any(r["source"] == "ep003" and "safety filter" in r["reason"] for r in rej), \
        "content_filter reply should be skipped and recorded"

    # 3) all models down -> partial outputs written, clear message, exit code 1
    pipeline.CONFIG.update(base, provider="gemini", base_url=f"http://127.0.0.1:{port}/v1beta",
                           models=["busy-model"])
    args.out = str(out_root / "down")
    code = 0
    try:
        pipeline.run(args)
    except SystemExit as e:
        code = e.code
    s = json.loads((out_root / "down" / "run_summary.json").read_text())
    assert code == 1 and s["stopped"], "should stop cleanly with exit code 1"
    assert (out_root / "down" / "report.md").read_text().count("PARTIAL RUN") == 1

    # 3b) out of credit (OpenAI insufficient_quota) -> stop at once with a clear message, no retry loop
    REQUESTS.clear()
    pipeline.CONFIG.update(base, provider="openai", base_url=f"http://127.0.0.1:{port}/v1",
                           models=["broke-model"])
    args.out = str(out_root / "broke")
    code = 0
    try:
        pipeline.run(args)
    except SystemExit as e:
        code = e.code
    s = json.loads((out_root / "broke" / "run_summary.json").read_text())
    assert code == 1 and "Out of credit" in (s["stopped"] or ""), s["stopped"]
    assert len(REQUESTS) == 1, f"should stop after one request, made {len(REQUESTS)}"

    # 4) --ping and --list-models
    pipeline.CONFIG.update(base, provider="gemini", base_url=f"http://127.0.0.1:{port}/v1beta",
                           models=["good-model", "busy-model", "missing-model"])
    print()
    pipeline.ping(pipeline.CONFIG)
    pipeline.list_models(pipeline.CONFIG)
    pipeline.CONFIG.update(provider="openai", base_url=f"http://127.0.0.1:{port}/v1")
    pipeline.list_models(pipeline.CONFIG)
    server.shutdown()

    # 5) script format variants must parse to the same timed lines as the original
    check_formats(pack)
    # 6) opening check cuts, "test your own cut" parsing, the quality bars
    check_cut_tools()
    print("\nALL OFFLINE TESTS PASSED")


def check_cut_tools():
    import tempfile
    from parser import parse_episode
    lines = ["TITLE: CUT TEST", "EPISODE: 3", "LOGLINE: A test.", "RUNTIME: 01:10", "",
             "[00:00.0] SCENE 1 - INT. ROOM - DAY", "[00:00.0] CAM: A pen over a contract."]
    for k in range(1, 14):        # a line every 5 seconds: 00:05 ... 01:05
        lines.append(f"[{k * 5 // 60:02d}:{k * 5 % 60:02d}.0] ANNA: Line {k}.")
    f = Path(tempfile.mkdtemp(prefix="cut_")) / "ep3.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ep = parse_episode(f)
    ep["duration_sec"] = 70.0
    show = pipeline.Show("Cut Test", [ep])
    rows = show.rows[3]
    clip = {"segments": [pipeline.make_segment(show, 3, 1, 10)], "duration_sec": None}
    clip["duration_sec"] = pipeline.total_duration(clip["segments"])          # 00:00-00:50
    new, why = pipeline.start_at_hook(show, clip, 3)                           # start on Line 2 (00:10)
    assert new and new[0]["start_sec"] == 10.0 and pipeline.total_duration(new) == 40.0, (new, why)
    assert pipeline.start_at_hook(show, clip, 1)[1] == "already opens on its hook"
    assert "45%" in pipeline.start_at_hook(show, clip, 7)[1] or "under" in pipeline.start_at_hook(show, clip, 7)[1]
    assert pipeline.start_at_hook(show, clip, 12)[1] == "the named line is not in the clip"
    print("  opening check ok: starts on the hook line, refuses to drop most of the clip")

    eps = show.episodes
    assert pipeline.parse_cut("3 00:10-00:45", eps) == (3, [(10.0, 45.0)])
    assert pipeline.parse_cut("ep3 0:10 - 0:45 + 1:00-1:05", eps) == (3, [(10.0, 45.0), (60.0, 65.0)])
    assert pipeline.parse_cut("00:10-00:45", eps) == (3, [(10.0, 45.0)])        # one episode: no number needed
    assert pipeline.parse_cut("3 10-45", eps) == (3, [(10.0, 45.0)])
    for bad in ("3 01:10", "3 ab-cd", "", "3 1:2:3:4-5"):
        try:
            pipeline.parse_cut(bad, eps)
            raise AssertionError(f"accepted a bad cut: {bad!r}")
        except ValueError:
            pass
    sg = pipeline.cut_segment(show, 3, 10.0, 45.0)
    assert (rows[sg["start_line"] - 1]["start"], sg["start_sec"], sg["end_sec"]) == (10.0, 10.0, 45.0), sg
    for t0, t1, msg in ((40, 30, "must come after"), (80, 90, "only"), (11, 13, "no script line")):
        try:
            pipeline.cut_segment(show, 3, t0, t1)
            raise AssertionError("accepted a bad range")
        except ValueError as e:
            assert msg in str(e), e
    try:
        pipeline.cut_segment(show, 9, 0, 30)
        raise AssertionError("accepted a missing episode")
    except ValueError as e:
        assert "not in these scripts" in str(e)
    print("  your-cut parsing ok: episode + mm:ss ranges, joined pieces, clear errors")

    good = {"judge": {"hook": 4, "cliffhanger": 4, "standalone": 3}, "cold_viewer": {"comprehension": 4, "intrigue": 4}}
    assert pipeline.quality_failures(good) == []
    unclear = {"judge": dict(good["judge"], standalone=2.3), "cold_viewer": good["cold_viewer"]}
    assert pipeline.quality_failures(unclear) == ["hard to follow on its own (judge standalone 2.3/5)"]
    print("  quality bars ok: a clip a stranger can't follow on its own is not posted")


def check_formats(pack):
    import re
    import tempfile
    from parser import find_episode_files, parse_episode
    files, _ = find_episode_files(pack)
    src_file = sorted(files)[0]
    src = src_file.read_text(encoding="utf-8")

    def sig(ep):
        return [(e["type"], e["speaker"], round(e["start"], 2), round(e["end"], 2), e["content"])
                for sc in ep["scenes"] for e in sc["events"]]

    base = sig(parse_episode(src_file))
    times = r"Start: \((\S+)\) \| End: \((\S+)\)"
    variants = {
        "no ** stars": src.replace("**", ""),
        "no brackets": re.sub(times, r"Start: \1 | End: \2", src),
        "'a - b' times": re.sub(times, r"\1 - \2", src),
        "'a --> b' times": re.sub(times, r"\1 --> \2", src),
        "header and time on one line": re.sub(r"\*\*(.+?):\*\*\n" + times, r"**\1:** Start: (\2) | End: (\3)", src),
        "BOM and Windows line ends": "\ufeff" + src.replace("\n", "\r\n"),
        "no SCENE markers": re.sub(r"SCENE \d+\n", "", src).replace("Scene heading: ", ""),
    }
    tmp = Path(tempfile.mkdtemp(prefix="fmt_"))
    print()
    for name, text in variants.items():
        f = tmp / "episode-001.txt"
        f.write_text(text, encoding="utf-8", newline="")
        got = sig(parse_episode(f))
        assert got == base, f"format variant '{name}' parsed differently ({len(got)} vs {len(base)} lines)"
        print(f"  format ok: {name} ({len(got)} timed lines)")

    # second layout: one timestamp at the start of each line, a line ends where the next begins
    stamped = (
        "TITLE: TEST STORY\nEPISODE: 7\nLOGLINE: A test.\nRUNTIME: 00:20\n\n"
        "[00:00.0] SCENE 1 - INT. ROOM - DAY\n"
        "[00:00.0] CAM: A door.\n"
        "[00:03.0] DIALOGUE - ANNA: Who is there?\n"
        "[00:06.5] ACTION: The door opens.\n"
        "[00:09.0] BEN (whispering): It's me.\n"
        "[00:12.0] SOUND: Thunder.\n"
        "[00:15.0] CUT TO BLACK.\n")
    f = tmp / "story.txt"
    f.write_text(stamped, encoding="utf-8")
    ep = parse_episode(f)
    evs = [e for sc in ep["scenes"] for e in sc["events"]]
    assert ep["episode_number"] == 7 and ep.get("title") == "TEST STORY", ep
    assert [e["type"] for e in evs] == ["CAMERA", "DIALOGUE", "ACTION", "DIALOGUE", "SOUND", "ACTION"], evs
    assert (evs[1]["speaker"], evs[3]["speaker"], evs[3]["tone"]) == ("ANNA", "BEN", "whispering"), evs
    assert [e["end"] for e in evs] == [3.0, 6.5, 9.0, 12.0, 15.0, 20.0], [e["end"] for e in evs]
    print(f"  format ok: one timestamp per line ({len(evs)} timed lines)")


if __name__ == "__main__":
    main(sys.argv[1])
