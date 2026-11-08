# Promo Finder (Kuku AI Generalist Assignment, Problem 1)

Reads a microdrama's screenplays and finds the best 20–60 second promo cuts: the episode and the
exact start/end timestamps of every segment (in play order), the hook, the cliffhanger, one line
on why it works, and a ranked "post these first" list.

## Quick start (Mac / Linux, Python 3.10+)

```bash
pip3 install requests
export GEMINI_API_KEY="your-key"
python3 pipeline.py --ping                                         # which models answer right now?
python3 pipeline.py "path/to/scripts-folder-or.zip" --stability 3  # full run (~45 min on free tier)
```

Add `--limit 3` for a quick test on 3 episodes. On Windows, use `set GEMINI_API_KEY=your-key` and
`python` instead of `python3`. With no key at all, `--dry-run` runs the whole pipeline on fake replies.

If a run stops (overloaded model, rate limit, closed laptop), run the same command again:
finished steps are reused, so nothing is paid for twice.

## Web app (optional)

A local web page around the same pipeline: drop a show's scripts (episode .txt files, a folder, or a
.zip), pick a model, watch the run, and browse
the promos (timestamps, hook, cliffhanger, scores, what a stranger thought, the clip script).

```bash
python3 app.py          # opens http://127.0.0.1:8000
```

- Nothing to install: standard library only. Each run is `pipeline.py` in a subprocess.
- Saved runs (any `outputs*` folder) open instantly, without new model calls.
- An API key typed in the page is passed to that run only and never written to disk; a key already
  exported in the terminal is used if the box is empty.
- **Test your own cut**: under a run's picks, type an episode and a start/end time. The cut is scored
  exactly like the tool's picks, next to the tool's own pick for that episode (see below).
- Every saved run shows the scoring version it used. Earlier experiments, made while the tool was being
  built, are folded away: their scores sit on a different scale and are not comparable.
- Demo mode runs with fake model replies, to try the app without a key.
- UI runs and uploaded zips go to `outputs_ui/`, which git ignores.
- The poster shelf at the bottom uses generated art. Put images you have the rights to in `posters/`
  and the shelf shows them instead.

## Outputs (`outputs/`)

| File | What it is |
|---|---|
| `report.md` | Post-first list with hook, cliffhanger, scores, the stranger's reaction and the clip's script; funnel stats; auto-flagged weak picks; stability check |
| `promos_ranked.csv` | Every valid promo, ranked: segments, duration, hook, cliffhanger, why, all scores |
| `promos_ranked.json` | Same, as JSON |
| `rejected_proposals.json` | Every proposal or step that was rejected, and why |
| `run_summary.json` | Model calls, tokens, latency, wall time, funnel counts, scoring version |
| `evaluated_cuts.md` / `.json` | Cuts you tested with `--evaluate` (or in the web app), with their scores |
| `calls_log.jsonl` | One line per model call, including failures and retries |
| `raw/` | Every model reply, unedited (the required raw outputs) |

## How it works

1. **Parse** (code). Screenplays become a numbered, timestamped timeline per episode.
2. **Propose** (model, 1 call per episode). The model proposes 3 cuts and refers to lines **by ID only**.
   The prompt also says how many lines make 20/30/60 seconds in that episode. (Measured, one run
   each: too-short first proposals fell from 28 to 18, so the hint helps a little; the feedback
   loop in step 3 is what recovers the rest.)
3. **Assemble** (code). IDs map back to the script's own timestamps. Cut points snap to the nearest
   camera cut (if within 2s). The 20–60s rule is enforced. Over-long cuts are repaired, not
   discarded: the code keeps the opening (hook) and the ending (cliffhanger) and drops lines from
   the middle, as an editor would. Any cut the code rejects goes back to the model with the
   exact problem ("runs 11.5s, needs 20–60s") and is checked again, up to twice; the second try
   also says how many lines to add. In run 1, before this loop existed, 28 of 150 cuts were lost
   as too short.
4. **Judge** (model, 3 calls per episode). A separate reviewer scores each cut 0–5 on hook (first
   3 seconds), cliffhanger, standalone, flow and premise (does it sell what the title promises?),
   and labels its story beat (betrayal, romance, reveal, danger, revenge, other). It votes 3 times
   and the scores are averaged, because one vote is noisy: in run 4 the same clip got hook 4 in one
   run and 2 in the next. The report shows the average gap between a clip's votes. `--judge-votes 1`
   is cheaper. The premise score was added after run 3, whose top 10 missed the two moments the
   title is about.
5. **Opening and ending checks** (model, top 25 only). The model is shown the clip with line IDs and
   names the line where the hook lands; if the clip opens before it (an establishing shot, someone
   walking in), the code starts it there. Then it names the line where the tension peaks; if the
   clip runs on past it (a calm exit, a slow fade), the code ends it there. Changed clips are judged
   again. Each trim keeps at least 20 seconds and most of the clip. Why: on a new practice script the
   best cut opened on a calm shot of a pen, so the judge gave it hook 2 and the tool dropped it; the
   accusation 11 seconds later was the real hook. And run 4's #2 pick ran 8 seconds past its peak
   into a walk-off, yet the judge still gave its cliffhanger a 4.
6. **Stranger check** (model, top 25 only). A call that sees **only the clip** (no show context)
   reports what it thinks is happening, comprehension 1–5 and intrigue 1–5. This tests "makes sense
   on its own" directly.
7. **Stitch** (model). Cross-episode promos are built from the best moments, then judged and
   stranger-checked.
8. **Rank** (code). Final score = 50% judge + 30% stranger intrigue + 20% stranger comprehension.
   The post-first list also enforces the brief's must-haves as hard rules: judge cliffhanger 3.5+
   (average of the votes), hook 3+, clear on its own (judge standalone 3+), a stranger understood
   the clip (comprehension 3+) and would stop for it (intrigue 3+). It takes at most one promo per
   episode and at most 3 of one story beat, never repeats footage, and never repeats a scene: an
   episode whose first scene is
   headed CONTINUOUS picks up the previous episode's last scene, so the two count as one moment.
   The report lists every promo skipped on the way down the ranking, and why. If no clip passes
   every bar, none is marked to post first and the report and app show the closest ones and the
   bar each missed.

About 330 model calls per 50-episode show from scratch (50 proposals, 150 judge votes, ~30 fixes,
25 opening and 25 ending checks, 25 stranger checks, stitching). Spending the per-clip checks only on finalists
cuts those calls roughly 3x versus checking everything.

## Test your own cut

If you think another moment makes a better promo, score it the same way:

```bash
python3 pipeline.py "path/to/scripts" --out outputs --evaluate "101 01:37-02:18" --evaluate "101 00:04-00:36"
```

Each cut is written as the episode, then start-end (`mm:ss`); join pieces with `+`
(`"12 00:20-00:55 + 01:10-01:20"`). The cut goes through the same judge (same number of votes), the
same stranger and the same bars as the tool's picks. Each episode's cuts are judged in one call
together with the tool's own pick for that episode from the `--out` run, so the two scores sit on the
same scale. About 6 model calls for 3 cuts. Results go to `evaluated_cuts.md` and `.json` in that
folder.

## Comparing runs

`run_summary.json` records `scoring_version` (this code: 8). Scores are only comparable between runs
with the same scoring. Earlier versions used one judge vote and fewer checks and bars (the first
run also had no premise score). One vote is noisier, and the highest of many noisy scores runs high,
so an older run's top score can be higher without its pick being better.

## Models, limits and swapping

Defaults try `gemini-3.5-flash-lite`, then `gemini-3.1-flash-lite`, then `gemini-3.8-flash`, pacing
one request every 7 seconds. On the free tier at the time of writing, the Flash-Lite models allow
about 500 requests/day and 10/minute, while the Flash models allow only about 20/day, so the run
leads with Flash-Lite. Check your own limits at aistudio.google.com (Rate limit page).

How failures are handled: a model that keeps failing (overloaded, timing out) is skipped for 2
minutes and then tried again. A model that is missing or out of today's quota is dropped for the
run. Rate limits are waited out. If the model's safety filter refuses one request, only that step
is skipped and it is listed in the report.

Everything is set by environment variables; no code changes.

```bash
export LLM_MODELS="gemini-3.8-flash"     # models to try, in order
export LLM_THINKING=low                  # optional thinking level
export LLM_TEMPERATURE=0.7               # optional (default: the model's own)
export LLM_MIN_INTERVAL=7                # seconds between requests (paid tiers can use 0)
```

Example with OpenAI (paid, about $1 per full show with a mini model):

```bash
export LLM_PROVIDER=openai
export LLM_API_KEY="sk-..."
export LLM_MODELS="gpt-5.4-mini"
export LLM_MIN_INTERVAL=2
export LLM_THINKING=low
python3 pipeline.py "path/to/scripts" --stability 3 --out outputs_openai
```

Any OpenAI-compatible API (OpenAI, Groq, OpenRouter, or an API Kuku provides):

```bash
export LLM_PROVIDER=openai
export LLM_BASE_URL="https://<provider>/v1"
export LLM_API_KEY="<key>"
export LLM_MODELS="<model name>"
python3 pipeline.py --list-models        # see what that key can use
```

`calls_log.jsonl` records which model answered each call.

## Known limits

- Two script layouts are read: Start/End time blocks (the pack's format), and one timestamp at the
  start of each line (`[00:04.0] DIALOGUE - KAVITA: ...`), where a line ends when the next begins
  (the last at `RUNTIME`). Files that are separate stories (each with its own `TITLE:`) keep their
  own title in the prompts and are never stitched together.
- For the first layout, the parser takes the pack's format and small
  variations of it: no `**` stars, no brackets, `00:01.00 - 00:04.80` or `-->` times, the header and
  time on one line, no `SCENE 01` markers, Windows line endings. A file with no timed lines is skipped,
  and if none have them the run stops with a clear message before any API call.
- It reads scripts, not video. Acting, music and visual punch are judged from the written descriptions.
- Model judges are noisy. Averaging 3 votes per clip narrows it but does not remove it: the order
  below the top few picks is partly luck. `report.md` shows the vote gap and a stability check
  (how many picks reappear when proposals are generated again).
- The "stranger" is a model, not a person. The real test is posting the clips and comparing
  view-through and install rates per promo.
- Trimmed cuts can feel jumpy. The report flags any whose flow score is 2 or lower.

## Confidential inputs

The screenplays are Kuku's content. `.gitignore` keeps the scripts (`episode-*.txt`, zips) and the
`outputs/` folder (which quotes the scripts) out of git. Share outputs through a private link instead.

## Tests

```bash
python3 pipeline.py "path/to/scripts" --dry-run       # whole pipeline on fake replies
python3 test_offline.py "path/to/scripts"             # model client vs a local fake API: overload,
                                                      # missing model, daily quota, rate limits,
                                                      # safety blocks, JSON mode refused,
                                                      # both API formats,
                                                      # script format variants
```
