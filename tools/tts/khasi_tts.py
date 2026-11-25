"""Read Khasi Bible chapters aloud with the Khasi-trained OmniVoice model, cloning Marcel's voice.

Two commands, both run by .github/workflows/generate-ai-audio.yml:

  plan  <books> <chapters> <release> <chars_per_job>
        Print a GitHub Actions matrix (JSON) of jobs. Each job gets whole chapters, about
        chars_per_job characters of text. Chapters already in the release are left out, so a
        human recording is never replaced and a re-run only does what is still missing.
  speak <items> <release> <deadline_minutes>
        Generate each chapter in <items> ("gen:2,gen:3"), save <book>_<chapter>.mp3 and upload it
        to the release straight away. Stops before the deadline so a job never times out mid-chapter.

books: "gen" / "gen,exo" / "all". chapters: "" for all, or "2" / "2-5".
Settings (env): TTS_MODEL, NUM_STEP (default 32), BATCH (verses per model call, default 8).
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATE = 24000


def gh_assets(release):
    out = subprocess.run(["gh", "release", "view", release, "--json", "assets", "--jq", ".assets[].name"],
                         capture_output=True, text=True)
    return set(out.stdout.split()) if out.returncode == 0 else set()


def load_book(book_id):
    return json.loads((HERE / "text" / f"{book_id}.json").read_text(encoding="utf-8"))


def chapter_range(spec):
    if not spec:
        return None
    a, _, b = spec.partition("-")
    return set(range(int(a), int(b or a) + 1))


def plan(books, chapters, release, chars_per_job):
    ids = sorted(p.stem for p in (HERE / "text").glob("*.json")) if books == "all" else books.split(",")
    wanted = chapter_range(chapters)
    have = gh_assets(release)
    todo = []
    for book_id in ids:
        for ch in load_book(book_id.strip())["chapters"]:
            n = ch["number"]
            if (wanted and n not in wanted) or f"{book_id}_{n}.mp3" in have:
                continue
            todo.append((f"{book_id}:{n}", sum(len(v["text"]) for v in ch["verses"])))
    jobs, cur, size = [], [], 0
    for item, chars in todo:
        if cur and size + chars > chars_per_job:
            jobs.append(cur)
            cur, size = [], 0
        cur.append(item)
        size += chars
    if cur:
        jobs.append(cur)
    if len(jobs) > 256:
        sys.exit(f"{len(jobs)} jobs is over GitHub's limit of 256 per run: pick fewer books.")
    print(f"{len(todo)} chapters to read in {len(jobs)} jobs", file=sys.stderr)
    print(json.dumps({"items": [",".join(j) for j in jobs] or ["none"]}))


def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"[“”\"*]", "", text)).strip()


def speak(items, release, deadline_min):
    import numpy as np
    import torch
    from omnivoice import OmniVoice

    start = time.time()
    torch.set_num_threads(os.cpu_count() or 4)
    names = json.loads((HERE / "book_names.json").read_text(encoding="utf-8"))
    model_id = os.environ.get("TTS_MODEL", "toiar/Khasi-OmniVoice-TTS-2")
    num_step, batch = int(os.environ.get("NUM_STEP", "32")), int(os.environ.get("BATCH", "8"))
    print(f"Model: {model_id}, num_step={num_step}, batch={batch}", flush=True)
    model = OmniVoice.from_pretrained(model_id, device_map="cpu", dtype=torch.float32)
    prompt = model.create_voice_clone_prompt(ref_audio=str(HERE / "ref.wav"),
                                             ref_text=(HERE / "ref.txt").read_text(encoding="utf-8").strip())
    pause = np.zeros(int(RATE * 0.6), dtype=np.float32)
    rate = None  # seconds of compute per character, measured as we go
    for item in items.split(","):
        book_id, n = item.split(":")
        name = f"{book_id}_{n}.mp3"
        ch = next(c for c in load_book(book_id)["chapters"] if c["number"] == int(n))
        lines = [f"{names.get(book_id, book_id)}, lynnong {n}."] + [clean(v["text"]) for v in ch["verses"]]
        chars = sum(map(len, lines))
        left = deadline_min * 60 - (time.time() - start)
        if rate and rate * chars > left:
            print(f"Stopping before {item}: about {rate * chars / 60:.0f} min needed, {left / 60:.0f} left. "
                  "Run the workflow again to finish.", flush=True)
            break
        if name in gh_assets(release):
            print(f"{name} already exists, skipping", flush=True)
            continue
        t = time.time()
        torch.manual_seed(1)
        pieces = []
        for i in range(0, len(lines), batch):
            part = lines[i:i + batch]
            audios = model.generate(text=part, voice_clone_prompt=[prompt] * len(part), num_step=num_step)
            for a in audios:
                pieces += [a.astype(np.float32), pause]
        took = time.time() - t
        rate = took / chars if rate is None else 0.5 * rate + 0.5 * took / chars
        wav = np.concatenate(pieces)
        print(f"{item}: {len(wav) / RATE / 60:.1f} min of audio in {took / 60:.1f} min", flush=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(RATE), "-ac", "1",
                        "-i", "-", "-ac", "1", "-b:a", "64k", name], input=wav.tobytes(), check=True)
        if name in gh_assets(release):  # someone uploaded it meanwhile: never replace it
            print(f"{name} appeared while generating, not uploading", flush=True)
            continue
        subprocess.run(["gh", "release", "upload", release, name], check=True)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "plan":
        plan(args[0], args[1], args[2], int(args[3]))
    elif cmd == "speak":
        speak(args[0], args[1], float(args[2]))
    else:
        sys.exit(__doc__)
