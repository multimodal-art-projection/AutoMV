#!/usr/bin/env python3
"""LSE-D/LSE-C adapter for joonson/syncnet_python.

Use a directory of singing excerpts, or --prepared-crops for already selected
face-track AVIs at 25 fps with 16 kHz audio. This is a new upstream adapter;
the original per-video LSE records and face selections were not found locally.
"""

import argparse
import math
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from common import VIDEO_EXTENSIONS, write_json


def distance_and_confidence(distances):
    """SyncNet returns window-by-shift distances; reduce over windows first."""
    import numpy as np

    distances = np.asarray(distances)
    if (
        distances.ndim != 2
        or min(distances.shape) == 0
        or not np.isfinite(distances).all()
    ):
        raise ValueError(
            "Expected a finite, nonempty SyncNet window-by-shift distance matrix"
        )
    mean_distances = distances.mean(axis=0)
    minimum = float(mean_distances.min())
    # torch.median uses the lower middle entry when the number of shifts is even.
    median = float(np.sort(mean_distances)[(len(mean_distances) - 1) // 2])
    return minimum, median - minimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syncnet-root", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, help="Defaults to <syncnet-root>/data/syncnet_v2.model"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Singing excerpts or prepared face-track AVIs",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepared-crops", action="store_true")
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    root = args.syncnet_root.expanduser().resolve()
    folder = args.dir.expanduser().resolve()
    work = args.work_dir.expanduser().resolve()
    if not (root / "SyncNetInstance.py").is_file():
        parser.error("--syncnet-root must be a checkout of joonson/syncnet_python")
    videos = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos or args.vshift < 0 or args.batch_size < 1:
        parser.error(
            "Provide videos, a nonnegative --vshift and a positive --batch-size"
        )
    model_path = (args.model or root / "data/syncnet_v2.model").expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"Missing SyncNet checkpoint: {model_path}")
    sys.path.insert(0, str(root))
    from SyncNetInstance import SyncNetInstance

    model = SyncNetInstance()
    model.loadParameters(str(model_path))
    records, errors = [], []
    for index, video in enumerate(videos):
        # Separate directories prevent face-track selection from leaking between inputs.
        reference = f"video_{index:05d}"
        local_work = work / reference
        local_work.mkdir(parents=True, exist_ok=True)
        try:
            if args.prepared_crops:
                crops = [video]
            else:
                subprocess.run(
                    [
                        sys.executable,
                        str(root / "run_pipeline.py"),
                        "--videofile",
                        str(video),
                        "--reference",
                        reference,
                        "--data_dir",
                        str(local_work),
                        "--frame_rate",
                        "25",
                        "--overwrite",
                    ],
                    cwd=root,
                    check=True,
                )
                crops = sorted((local_work / "pycrop" / reference).glob("0*.avi"))
            if not crops:
                raise ValueError(
                    "No valid face tracks; LSE is unavailable for this input"
                )
            options = SimpleNamespace(
                batch_size=args.batch_size,
                vshift=args.vshift,
                tmp_dir=str(local_work / "pytmp"),
                reference=reference,
            )
            tracks = []
            for crop in crops:
                offset, confidence, distances = model.evaluate(
                    options, videofile=str(crop)
                )
                distance, reduced_confidence = distance_and_confidence(distances)
                if not math.isclose(
                    float(confidence), reduced_confidence, abs_tol=1e-4
                ):
                    raise ValueError(
                        "SyncNet confidence disagrees with the returned distance matrix"
                    )
                tracks.append(
                    {
                        "track": str(crop),
                        "offset_frames": int(offset),
                        "lse_d": distance,
                        "lse_c": float(confidence),
                    }
                )
            records.append(
                {
                    "video": video.name,
                    "n_tracks": len(tracks),
                    "tracks": tracks,
                    "lse_d": statistics.mean(t["lse_d"] for t in tracks),
                    "lse_c": statistics.mean(t["lse_c"] for t in tracks),
                }
            )
        except Exception as error:
            errors.append(
                {"video": video.name, "error": f"{type(error).__name__}: {error}"}
            )
        write_json(
            args.out,
            {
                "implementation": "joonson/syncnet_python adapter",
                "aggregation": "equal weight per track within input, then per input",
                "n_requested": len(videos),
                "n_valid": len(records),
                "lse_d": statistics.mean(r["lse_d"] for r in records)
                if records
                else None,
                "lse_c": statistics.mean(r["lse_c"] for r in records)
                if records
                else None,
                "records": records,
                "errors": errors,
            },
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
