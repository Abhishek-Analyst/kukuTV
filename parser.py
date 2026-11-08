"""
Screenplay parser for Kuku microdrama scripts.

Turns the structured .txt screenplay format into clean JSON: episode metadata,
scenes, and timestamped events (CAMERA, ACTION, SOUND, DIALOGUE).
Pure Python, no AI: deterministic and checkable.

Accepts a folder of episode files (searched recursively) or a .zip of them.
"""
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

TS_PAT = r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)"
# Block header: "**CAMERA:**" (the pack's format). Also accepted: "CAMERA:" without the stars,
# but only when a time line follows, so a dialogue line ending in ":" is not mistaken for one.
BLOCK_HEADER = re.compile(r"^\*\*(.+?):\*\*\s*$")
LOOSE_HEADER = re.compile(r"^\*{0,2}\s*([A-Za-z][^:*\"“]{0,79}?)\s*(?::\s*\*{0,2}|\*{0,2}\s*:)\s*$")
# Time line: "Start: (00:01.00) | End: (00:04.80)". Also accepted: no brackets, and
# "00:01.00 - 00:04.80", "00:01.00 --> 00:04.80", "00:01.00 to 00:04.80", "[00:01.00 | 00:04.80]".
_SEP = r"(?:\||-->|->|–|—|-|\bto\b|,)"
TIME_PART = (rf"[\(\[]?\s*(?:start(?:\s*time)?\s*[:=]?\s*)?[\(\[]?{TS_PAT}[\)\]]?\s*{_SEP}\s*"
             rf"(?:end(?:\s*time)?\s*[:=]?\s*)?[\(\[]?{TS_PAT}[\)\]]?\s*[\)\]]?")
TIME_LINE = re.compile(rf"^{TIME_PART}$", re.IGNORECASE)
# Header and time on one line: "**CAMERA:** Start: (00:00.00) | End: (00:04.80)" or "CAMERA (00:00.00 - 00:04.80):"
INLINE_HEADER = re.compile(rf"^\*{{0,2}}\s*([A-Za-z][^:*\"“\[\(]{{0,79}}?(?:\s*\([^)]*\))*)\s*:?\s*\*{{0,2}}\s*:?\s*{TIME_PART}\s*:?$",
                           re.IGNORECASE)
SCENE_HEADER = re.compile(r"^\*{0,2}\s*SCENE\s+(\d+)\b", re.IGNORECASE)
SLUGLINE = re.compile(r"^\*{0,2}\s*(INT\.|EXT\.|INT/EXT|I/E\b)", re.IGNORECASE)

# Second layout: every line starts with its own time, e.g.
#   [00:04.0] DIALOGUE - KAVITA: Sign the transfer, Meera.
#   [00:43.0] SCENE 2 - INT. BANK RECORDS ROOM - DAY
# A line ends where the next one starts (the last one at RUNTIME). A range is used when given:
#   00:04.0 - 00:08.0 KAVITA: Sign the transfer.
STAMP_LINE = re.compile(rf"^[\[\(]?\s*{TS_PAT}\s*(?:(?:-->|->|–|—|-|\bto\b)\s*{TS_PAT}\s*)?[\]\)]?\s*[:|–—-]?\s*(\S.*)$",
                        re.IGNORECASE)
STAMP_SCENE = re.compile(r"^SCENE\s+(\d+)\b\s*[-–—:.]?\s*(.*)$", re.IGNORECASE)
STAMP_KIND = re.compile(r"^(CAM|CAMERA|SHOT|ANGLE|ACTION|SOUND FX|SOUND|SFX|MUSIC|AUDIO)\b\s*[:\-–—]\s*(.*)$", re.IGNORECASE)
STAMP_DIALOGUE = re.compile(r"^(?:DIALOGUE|DIALOG|LINE|V\.?O\.?)\s*(?:[-–—:]\s*|\()\s*([^:()]+?)\s*(\([^)]*\))?\s*\)?\s*:\s*(.+)$",
                            re.IGNORECASE)
STAMP_SPEAKER = re.compile(r"^([A-Z][A-Z0-9 .'\-]{0,38}[A-Z0-9.])\s*((?:\([^)]*\)\s*)*):\s*(.+)$")
STAMP_TYPES = {"CAM": "CAMERA", "CAMERA": "CAMERA", "SHOT": "CAMERA", "ANGLE": "CAMERA", "ACTION": "ACTION",
               "SOUND FX": "SOUND", "SOUND": "SOUND", "SFX": "SOUND", "MUSIC": "SOUND", "AUDIO": "SOUND"}

# Block headers that are not dialogue, mapped to our event types
NON_DIALOGUE = {
    "CAMERA": "CAMERA",
    "ACTION": "ACTION",
    "SOUND FX": "SOUND",
    "SFX": "SOUND",
    "SOUND": "SOUND",
    "MUSIC": "SOUND",
}


def ts_to_seconds(ts: str) -> float:
    """'01:23.45' -> 83.45 ; '1:02:03.5' -> 3723.5"""
    parts = ts.strip().replace(",", ".").split(":")
    sec = float(parts[-1])
    mins = int(parts[-2]) if len(parts) >= 2 else 0
    hrs = int(parts[-3]) if len(parts) >= 3 else 0
    return hrs * 3600 + mins * 60 + sec


def seconds_to_ts(sec: float) -> str:
    """83.45 -> '01:23.45' (same format as the scripts)"""
    total = round(max(sec, 0.0), 2)
    m = int(total // 60)
    s = total - m * 60
    return f"{m:02d}:{s:05.2f}"


def _make_event(header: str, start: str, end: str) -> dict:
    ev = {"type": None, "speaker": None, "tone": None,
          "start": ts_to_seconds(start), "end": ts_to_seconds(end), "content": ""}
    key = header.strip().upper()
    if key in NON_DIALOGUE:
        ev["type"] = NON_DIALOGUE[key]
    else:
        # e.g. "ANNA (V.O.) (Reflective, Vulnerable)" -> speaker ANNA,
        #      tone "V.O., Reflective, Vulnerable"
        ev["type"] = "DIALOGUE"
        ev["speaker"] = header.split("(", 1)[0].strip()
        mods = [m.strip() for m in re.findall(r"\(([^)]*)\)", header) if m.strip()]
        ev["tone"] = ", ".join(mods) if mods else None
    if ev["end"] < ev["start"]:
        ev["start"], ev["end"] = ev["end"], ev["start"]
    return ev


def parse_episode(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    ep = {"file": path.name, "episode_number": None, "logline": "",
          "continuity_notes": "", "scenes": []}

    runtime = None
    for ln in lines:
        s = ln.strip()
        up = s.upper()
        if s.startswith("EPISODE NUMBER:"):
            m = re.search(r"(\d+)", s)
            ep["episode_number"] = int(m.group(1)) if m else None
        elif s.startswith("EPISODE LOGLINE:"):
            ep["logline"] = s.split(":", 1)[1].strip()
        elif s.startswith("CONTINUITY NOTES:"):
            ep["continuity_notes"] = s.split(":", 1)[1].strip()
        elif s.startswith("--- SCRIPT START"):
            break
        elif up.startswith(("TITLE:", "SHOW TITLE:", "SERIES:", "SHOW:")) and not ep.get("title"):
            ep["title"] = s.split(":", 1)[1].strip()
        elif up.startswith(("EPISODE:", "EPISODE NO:", "EP:")) and ep["episode_number"] is None:
            m = re.search(r"(\d+)", s.split(":", 1)[1])
            ep["episode_number"] = int(m.group(1)) if m else None
        elif up.startswith(("LOGLINE:", "SYNOPSIS:", "SUMMARY:")) and not ep["logline"]:
            ep["logline"] = s.split(":", 1)[1].strip()
        elif up.startswith(("RUNTIME:", "DURATION:", "LENGTH:")) and runtime is None:
            m = re.search(TS_PAT, s)
            runtime = ts_to_seconds(m.group(1)) if m else None

    if ep["episode_number"] is None:  # fall back to the first number in the file name
        m = re.search(r"(\d+)", path.stem)
        ep["episode_number"] = int(m.group(1)) if m else 0

    stripped_all = [raw.strip() for raw in lines]
    n_stamped = sum(1 for ln in stripped_all if STAMP_LINE.match(ln))
    n_blocks = sum(1 for ln in stripped_all if TIME_LINE.match(ln) or INLINE_HEADER.match(ln))
    if n_stamped >= 3 and n_stamped > n_blocks:
        return _finish(_parse_stamped(stripped_all, ep, runtime))

    cur_scene = None
    pending_header = None
    cur_event = None

    def flush():
        nonlocal cur_event, pending_header
        if cur_event is not None and cur_event["content"] and cur_scene is not None:
            c = cur_event["content"].strip()
            if (cur_event["type"] == "DIALOGUE" and len(c) >= 2
                    and c[0] in "\"“" and c[-1] in "\"”"):
                cur_event["content"] = c[1:-1].strip()  # scripts wrap dialogue in quotes
            cur_scene["events"].append(cur_event)
        cur_event = None
        pending_header = None

    stripped = [raw.strip() for raw in lines]
    has_scene_markers = any(SCENE_HEADER.match(ln) for ln in stripped)

    def next_line(i):
        for ln2 in stripped[i + 1:]:
            if ln2:
                return ln2
        return ""

    def new_scene(number, heading=""):
        nonlocal cur_scene
        flush()
        cur_scene = {"scene_number": number, "heading": heading,
                     "start": None, "end": None, "events": []}
        ep["scenes"].append(cur_scene)

    for i, ln in enumerate(stripped):
        if ln.startswith("--- SCRIPT END"):
            flush()
            break

        sm = SCENE_HEADER.match(ln)
        if sm:
            new_scene(int(sm.group(1)))
            continue
        if not has_scene_markers and SLUGLINE.match(ln):
            # scripts without "SCENE 01" markers: each INT./EXT. slugline starts a scene
            new_scene(len(ep["scenes"]) + 1, ln.strip("* "))
            continue
        if cur_scene is None:
            if not (BLOCK_HEADER.match(ln) or INLINE_HEADER.match(ln)
                    or (LOOSE_HEADER.match(ln) and TIME_LINE.match(next_line(i)))):
                continue
            new_scene(1)  # events before any scene marker go into scene 1

        if ln.startswith("Scene heading:"):
            cur_scene["heading"] = ln.split(":", 1)[1].strip()
            continue
        if ln.startswith("Scene start:"):
            cur_scene["start"] = ts_to_seconds(ln.split(":", 1)[1].strip())
            continue
        if ln.startswith("Scene end:"):
            cur_scene["end"] = ts_to_seconds(ln.split(":", 1)[1].strip())
            continue
        if ln.startswith("Actions and dialogue:"):
            continue

        hm = BLOCK_HEADER.match(ln)
        if not hm and not TIME_LINE.match(ln) and TIME_LINE.match(next_line(i)):
            hm = LOOSE_HEADER.match(ln)
        if hm:
            flush()  # a new block always closes the previous one
            pending_header = hm.group(1).strip()
            continue

        im = None if TIME_LINE.match(ln) else INLINE_HEADER.match(ln)
        if im:
            flush()
            cur_event = _make_event(im.group(1).strip(), im.group(2), im.group(3))
            continue

        tm = TIME_LINE.match(ln)
        if tm and pending_header is not None and cur_event is None:
            cur_event = _make_event(pending_header, tm.group(1), tm.group(2))
            continue

        if ln == "":
            if cur_event is not None and cur_event["content"]:
                flush()
            continue

        if cur_event is not None:
            cur_event["content"] = (cur_event["content"] + " " + ln).strip()

    flush()
    return _finish(ep)


def _finish(ep: dict) -> dict:
    ends = [s["end"] for s in ep["scenes"] if s["end"]] + [
        e["end"] for s in ep["scenes"] for e in s["events"]]
    ep["duration_sec"] = max(ends) if ends else 0.0
    ep["n_events"] = sum(len(s["events"]) for s in ep["scenes"])
    ep["n_dialogue"] = sum(1 for s in ep["scenes"] for e in s["events"]
                           if e["type"] == "DIALOGUE")
    return ep


def _parse_stamped(stripped: list[str], ep: dict, runtime: float | None) -> dict:
    """Layout where each line carries its own start time (see STAMP_LINE)."""
    items = []   # (start, explicit end or None, kind, payload)
    for ln in stripped:
        m = STAMP_LINE.match(ln)
        if not m:
            continue
        start = ts_to_seconds(m.group(1))
        end = ts_to_seconds(m.group(2)) if m.group(2) else None
        body = m.group(3).strip()
        sc = STAMP_SCENE.match(body)
        if sc:
            items.append((start, end, "SCENE", (int(sc.group(1)), sc.group(2).strip())))
            continue
        if SLUGLINE.match(body):
            items.append((start, end, "SCENE", (None, body)))
            continue
        ev = {"type": "ACTION", "speaker": None, "tone": None, "content": body}
        k = STAMP_KIND.match(body)
        d = None if k else STAMP_DIALOGUE.match(body)
        sp = None if (k or d) else STAMP_SPEAKER.match(body)
        if k:
            ev["type"], ev["content"] = STAMP_TYPES[k.group(1).upper()], k.group(2).strip()
        elif d:
            mods = [x.strip() for x in re.findall(r"\(([^)]*)\)", d.group(2) or "") if x.strip()]
            ev.update(type="DIALOGUE", speaker=d.group(1).strip().upper(), tone=", ".join(mods) or None,
                      content=d.group(3).strip())
        elif sp and sp.group(1).strip().upper() not in {"INT.", "EXT.", "NOTE", "NOTES"}:
            mods = [x.strip() for x in re.findall(r"\(([^)]*)\)", sp.group(2) or "") if x.strip()]
            ev.update(type="DIALOGUE", speaker=sp.group(1).strip(), tone=", ".join(mods) or None,
                      content=sp.group(3).strip())
        c = ev["content"]
        if ev["type"] == "DIALOGUE" and len(c) >= 2 and c[0] in "\"“" and c[-1] in "\"”":
            ev["content"] = c[1:-1].strip()
        if ev["content"]:
            items.append((start, end, "EVENT", ev))

    starts = [it[0] for it in items]
    cur_scene = None
    for idx, (start, end, kind, payload) in enumerate(items):
        later = [t for t in starts[idx + 1:] if t > start]
        if end is None or end <= start:
            if later:
                end = later[0]
            elif runtime and runtime > start:
                end = runtime
            else:
                end = start + 2.0
        if kind == "SCENE":
            number, heading = payload
            cur_scene = {"scene_number": number or len(ep["scenes"]) + 1, "heading": heading,
                         "start": start, "end": None, "events": []}
            ep["scenes"].append(cur_scene)
            continue
        if cur_scene is None:
            cur_scene = {"scene_number": 1, "heading": "", "start": start, "end": None, "events": []}
            ep["scenes"].append(cur_scene)
        payload.update(start=start, end=end)
        cur_scene["events"].append(payload)
    for sc in ep["scenes"]:
        if sc["events"]:
            sc["end"] = max(e["end"] for e in sc["events"])
    ep["scenes"] = [sc for sc in ep["scenes"] if sc["events"]]
    return ep


def find_episode_files(src: str) -> tuple[list[Path], str]:
    """Returns (episode files, show title). Accepts a folder or a .zip."""
    p = Path(src).expanduser()
    if p.is_file() and p.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="promo_pack_"))
        with zipfile.ZipFile(p) as z:
            z.extractall(tmp)  # zipfile strips absolute paths and '..'
        p = tmp
    if not p.exists():
        raise FileNotFoundError(
            f"Input not found: {src}\n"
            "Pass the folder that contains the episode .txt files (or the .zip itself).")

    files = [f for f in p.rglob("*.txt")
             if "__MACOSX" not in f.parts and not f.name.startswith("._")
             and not re.match(r"(readme|license|notes?)\b", f.stem, re.IGNORECASE)]
    episodes = [f for f in files
                if f.read_text(encoding="utf-8-sig", errors="replace").lstrip()
                .startswith("EPISODE NUMBER")]
    files = episodes or files
    if not files:
        raise FileNotFoundError(
            f"No episode .txt files found under: {p}\n"
            "Pass the folder that contains files like 'episode-001-....txt' (or the .zip).")
    parents = {f.parent for f in files}
    title = files[0].parent.name if len(parents) == 1 else p.name
    if title.startswith("promo_pack_"):
        title = "Untitled show"
    return files, title


def load_show(src: str) -> tuple[str, list[dict]]:
    files, title = find_episode_files(src)
    eps = sorted((parse_episode(f) for f in files), key=lambda e: e["episode_number"])
    titles = {e["title"].strip() for e in eps if e.get("title")}
    if len(titles) == 1 and len(eps) == sum(1 for e in eps if e.get("title")):
        t = titles.pop()
        title = t.title() if t.isupper() else t   # "THE LAST SIGNATURE" -> "The Last Signature"
    return title, eps


if __name__ == "__main__":
    title, eps = load_show(sys.argv[1])
    out = Path("parsed_episodes.json")
    out.write_text(json.dumps(eps, indent=1), encoding="utf-8")
    print(f"Show: {title}")
    print(f"Parsed {len(eps)} episodes -> {out}")
    for e in eps:
        flag = "  <-- CHECK" if e["n_dialogue"] == 0 or e["duration_sec"] == 0 else ""
        print(f"Ep {e['episode_number']:>3} | {seconds_to_ts(e['duration_sec'])} "
              f"| scenes {len(e['scenes'])} | events {e['n_events']:>3} "
              f"| dialogue {e['n_dialogue']:>3}{flag}")
