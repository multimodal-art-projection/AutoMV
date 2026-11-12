#!/usr/bin/env python3
"""Rebuild main-paper objective tables from archived scores; no model dependencies."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from common import HERE, SONG_IDS, read_jsonl, song_id, unique_rows

METHODS = [
    ("groundtruth", "GroundTruth", "Human"),
    ("music2video", "music2video", "Music2Video"),
    ("baseline2", "baseline2", "Revid.ai-base"),
    ("baseline1", "baseline1", "OpenArt-story"),
    ("lip", "AutoMV", "AutoMV"),
]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_archive(base=HERE):
    manifest = json.loads((base / "results/provenance.json").read_text())
    for record in manifest["files"]:
        data = (base / record["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {record['path']}")


def finite_map(rows, field):
    scores = unique_rows(rows)
    result = {}
    for sid, row in scores.items():
        value = row[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {field} score for song {sid}")
        result[sid] = value
    return result


def build_tables(base=HERE):
    raw = base / "results/raw"
    fvd = json.loads((raw / "fvd_results.json").read_text())["groups"]
    summaries, per_song = [], []
    for method, group, display in METHODS:
        ib = json.loads((raw / "imagebind" / group / "metrics.json").read_text())
        pairs = (raw / "imagebind" / group / "pairs.txt").read_text().splitlines()
        similarities = ib["paired_similarities"]
        if len(pairs) != len(similarities):
            raise ValueError(f"ImageBind pair/score length mismatch: {group}")
        ib_rows = []
        for pair, value in zip(pairs, similarities):
            audio, video = pair.split()
            if song_id(audio) != song_id(video):
                raise ValueError(f"Mismatched ImageBind pair: {pair}")
            ib_rows.append({"id": song_id(video), "ib": value})
        scores = {
            "IB": finite_map(ib_rows, "ib"),
            "CLAP": finite_map(
                read_jsonl(raw / f"metric3c_clapgen_cap_{method}.jsonl"), "clap"
            ),
            "Beat": finite_map(read_jsonl(raw / f"metric4c_{method}.jsonl"), "beat"),
        }
        summary = {"method": display, "song_set": "shared M2VBench"}
        for metric, values in scores.items():
            expected = set(SONG_IDS) - (
                {38} if method == "music2video" and metric == "CLAP" else set()
            )
            if set(values) != expected:
                raise ValueError(
                    f"Unexpected {display}/{metric} song coverage: {sorted(values)}"
                )
            summary.update(
                {
                    f"{metric}_n": len(values),
                    f"{metric}_percent": statistics.mean(values.values()) * 100,
                    f"{metric}_sd": statistics.stdev(values.values()) * 100,
                }
            )
        if not math.isclose(
            statistics.mean(scores["IB"].values()),
            ib["paired_similarity_mean"],
            abs_tol=1e-6,
        ):
            raise ValueError(f"ImageBind mean mismatch: {group}")
        summary["FVD"] = fvd[group].get("fvd_vs_GT")
        summaries.append(summary)
        for sid in SONG_IDS:
            per_song.append(
                {
                    "song_id": sid,
                    "method": display,
                    **{
                        f"{metric}_percent": values.get(sid) * 100
                        if sid in values
                        else None
                        for metric, values in scores.items()
                    },
                    "note": "Original AutoMV stream is truncated near 10.7 s"
                    if method == "lip" and sid == 16
                    else "",
                }
            )
    demos = read_jsonl(raw / "external_demos.jsonl")
    for group, display in [("mvcrafter", "MV-Crafter"), ("charcha", "CHARCHA")]:
        ib_values = json.loads(
            (raw / "imagebind" / group / "metrics.json").read_text()
        )["paired_similarities"]
        clap_values = [
            row["clap"]
            for row in demos
            if row["group"] == group and row.get("clap") is not None
        ]
        summaries.append(
            {
                "method": display,
                "song_set": "external demos (different songs)",
                "IB_n": len(ib_values),
                "IB_percent": statistics.mean(ib_values) * 100,
                "IB_sd": statistics.stdev(ib_values) * 100,
                "CLAP_n": len(clap_values),
                "CLAP_percent": statistics.mean(clap_values) * 100,
                "CLAP_sd": statistics.stdev(clap_values) * 100,
                "Beat_n": 0,
                "Beat_percent": None,
                "Beat_sd": None,
                "FVD": None,
            }
        )
    return per_song, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    verify_archive()
    per_song, summaries = build_tables()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_song_scores.csv", per_song)
    write_csv(args.output_dir / "summary.csv", summaries)
    print("method, IB (%), CLAP (%), FVD, Beat (%)")
    for row in summaries:

        def show(key):
            return f"{row[key]:.1f}" if row[key] is not None else "--"

        print(
            f"{row['method']}, {show('IB_percent')}, {show('CLAP_percent')}, "
            f"{show('FVD')}, {show('Beat_percent')}"
        )
    print(
        "LSE-D/LSE-C: see paper_table2.csv for manuscript values; original per-video records unavailable."
    )


if __name__ == "__main__":
    main()
