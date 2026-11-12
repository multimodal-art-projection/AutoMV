#!/usr/bin/env python3
"""Evaluate paper Beat and CLAP for a batch; resume completed metric/song rows."""

import argparse
import re
import sys
from pathlib import Path

from common import (
    HERE,
    SONG_IDS,
    append_jsonl,
    media_by_id,
    read_jsonl,
    read_segments,
    unique_rows,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, required=True, help="<id>.mp4 or mv_<id>.mp4"
    )
    parser.add_argument("--name", required=True, help="Batch name, e.g. AutoMV")
    parser.add_argument(
        "--metrics", nargs="+", choices=["beat", "clap"], default=["beat", "clap"]
    )
    parser.add_argument("--ids", type=int, nargs="+", default=list(SONG_IDS))
    parser.add_argument("--segments-dir", type=Path, default=HERE / "segments")
    parser.add_argument("--captions", type=Path, help="JSONL rows with id and caption")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="CLAP device; Beat runs on CPU")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name must contain only letters, digits, underscores or hyphens")
    if len(set(args.ids)) != len(args.ids):
        parser.error("--ids contains duplicates")
    if "clap" in args.metrics and args.captions is None:
        parser.error("CLAP requires --captions; generate them with caption.py")
    try:
        videos = media_by_id(args.dir)
        missing = set(args.ids) - videos.keys()
        if missing:
            raise ValueError(f"Missing requested videos: {sorted(missing)}")
        captions = (
            unique_rows(read_jsonl(args.captions)) if "clap" in args.metrics else {}
        )
        outputs = {
            metric: args.output_dir / f"{metric}_{args.name}.jsonl"
            for metric in args.metrics
        }
        done = {
            metric: unique_rows(read_jsonl(path)) if path.exists() else {}
            for metric, path in outputs.items()
        }
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    clap = None
    failures = 0
    for sid in args.ids:
        for metric in dict.fromkeys(args.metrics):
            if sid in done[metric]:
                continue
            try:
                if metric == "beat":
                    from beat import score

                    result = score(
                        videos[sid], read_segments(args.segments_dir / f"{sid}.json")
                    )
                else:
                    from clap_score import ClapScorer

                    caption = captions.get(sid, {}).get("caption")
                    if not caption:
                        raise ValueError(f"Missing caption for song {sid}")
                    if clap is None:
                        clap = ClapScorer(args.device)
                    result = clap.score(videos[sid], caption)
                append_jsonl(
                    outputs[metric], {"id": sid, "method": args.name, **result}
                )
                print(f"{args.name}/{sid}: {metric}={result[metric]}", flush=True)
            except Exception as error:
                failures += 1
                record = {
                    "id": sid,
                    "metric": metric,
                    "method": args.name,
                    "error": f"{type(error).__name__}: {error}",
                }
                append_jsonl(args.output_dir / f"errors_{args.name}.jsonl", record)
                print(f"ERROR {record}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
