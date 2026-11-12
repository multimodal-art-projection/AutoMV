#!/usr/bin/env python3
"""Generate the video captions used by paper CLAP (an optional Gemini API step)."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from common import append_jsonl, media_by_id, read_jsonl, unique_rows

MODEL = "gemini-3.1-pro-preview"
PROMPT = (
    "You are analyzing a music video. Watch the full video (visuals + audio) and "
    "return ONLY a JSON object with two fields:\n"
    '1. "emotion": an object giving a 0-1 intensity for each of these 8 moods '
    "conveyed by the VISUALS: joyful, sad, tender, angry, tense, energetic, calm, nostalgic.\n"
    '2. "caption": one English sentence describing the video\'s overall mood, imagery '
    "and atmosphere (not the plot).\n"
    "Return only the JSON, e.g. "
    '```json\n{"emotion":{"joyful":0.1,...,"nostalgic":0.6},"caption":"..."}\n```'
)


def upload_path(video, folder):
    # Same 14 MiB threshold and compression ladder as the local research run.
    if video.stat().st_size <= 14 * 1024 * 1024:
        return video
    target = Path(folder) / "upload.mp4"
    for height, fps, crf, kbps in [
        (480, 8, 30, 96),
        (360, 6, 32, 64),
        (288, 4, 34, 48),
        (240, 2, 36, 32),
    ]:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video),
                "-vf",
                f"scale=-2:{height},fps={fps}",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                "-b:a",
                f"{kbps}k",
                "-movflags",
                "+faststart",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        if target.stat().st_size <= 14 * 1024 * 1024:
            return target
    raise ValueError("Video exceeds the inline-upload limit after compression")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        parser.error(
            "Set GEMINI_API_KEY to generate captions, or use archived captions"
        )
    videos = media_by_id(args.dir)
    done = unique_rows(read_jsonl(args.out)) if args.out.exists() else {}
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=300000),
    )
    for sid, video in videos.items():
        if sid in done:
            continue
        with tempfile.TemporaryDirectory(prefix="automv-caption-") as folder:
            path = upload_path(video, folder)
            response = client.models.generate_content(
                model=args.model,
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=path.read_bytes(), mime_type="video/mp4"
                            )
                        ),
                        types.Part(text=PROMPT),
                    ]
                ),
                config=types.GenerateContentConfig(temperature=0.0),
            )
        text = response.text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        result = json.loads(fenced[1] if fenced else text)
        if not isinstance(result.get("caption"), str) or not result["caption"].strip():
            raise ValueError(f"Missing caption in response for {sid}")
        append_jsonl(
            args.out,
            {
                "id": sid,
                "caption": result["caption"],
                "emotion": result.get("emotion"),
                "model": args.model,
            },
        )
        print(f"Caption saved: {sid}", flush=True)


if __name__ == "__main__":
    main()
