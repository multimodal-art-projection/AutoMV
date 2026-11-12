"""Shared I/O for the Section 3.3 objective metrics."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
SONG_IDS = (
    4,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    23,
    29,
    30,
    31,
    33,
    35,
    37,
    38,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def song_id(path):
    match = re.fullmatch(r"(?:mv_)?(\d+)", Path(path).stem)
    if not match:
        raise ValueError(f"Expected <id> or mv_<id> filename: {path}")
    return int(match[1])


def media_by_id(directory, extensions=VIDEO_EXTENSIONS):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Missing media directory: {directory}")
    found = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        sid = song_id(path)
        if sid in found:
            raise ValueError(f"Duplicate song {sid}: {found[sid]} and {path}")
        found[sid] = path
    if not found:
        raise ValueError(f"No supported media in {directory}")
    return found


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unique_rows(rows, key="id"):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_segments(path):
    segments = json.loads(Path(path).read_text(encoding="utf-8"))["segments"]
    if not segments:
        raise ValueError(f"No lyric-aligned segments: {path}")
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid segment in {path}: {segment}")
    return segments


@contextmanager
def extracted_audio(video, sample_rate, duration=None):
    with tempfile.TemporaryDirectory(prefix="automv-audio-") as folder:
        wav = Path(folder) / "audio.wav"
        command = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
        if duration is not None:
            command += ["-t", str(duration)]
        command += ["-vn", "-ac", "1", "-ar", str(sample_rate), str(wav)]
        subprocess.run(command, check=True, capture_output=True)
        yield wav
