#!/usr/bin/env python3
"""Paper FVD: I3D Kinetics-400, four uniform 16-frame clips, 224x224.

Sampling/readers adapted from vbench_eval/server_artifacts/fvd_compute.py.
Only compare systems producing videos for the same songs as the reference.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from common import SONG_IDS, VIDEO_EXTENSIONS, media_by_id, write_json

CLIPS_PER_VIDEO, CLIP_LEN, RES = 4, 16, 224


def clip_indices(n_frames, fps):
    """Return list of index-arrays, one per clip."""
    if n_frames < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video must contain frames at a positive frame rate")
    stride = max(1, int(round(fps / 16.0)))
    span = CLIP_LEN * stride
    if n_frames < span:
        stride = max(1, n_frames // CLIP_LEN)
        span = CLIP_LEN * stride
    if n_frames >= span + CLIPS_PER_VIDEO - 1:
        starts = [
            int(round(i * (n_frames - span) / (CLIPS_PER_VIDEO - 1)))
            for i in range(CLIPS_PER_VIDEO)
        ]
    else:
        starts = [0]
    starts = sorted(set(starts))
    out = []
    for s in starts:
        idx = s + np.arange(CLIP_LEN) * stride
        idx = np.clip(idx, 0, n_frames - 1)
        out.append(idx)
    return out, stride


def read_clips_decord(path):
    import decord

    vr = decord.VideoReader(path, num_threads=2)
    n, fps = len(vr), float(vr.get_avg_fps()) or 30.0
    idx_list, stride = clip_indices(n, fps)
    clips = []
    for idx in idx_list:
        frames = vr.get_batch(idx.tolist()).asnumpy()  # (T,H,W,3) uint8
        clip = np.stack(
            [cv2.resize(f, (RES, RES), interpolation=cv2.INTER_AREA) for f in frames]
        )
        clips.append(clip)
    del vr
    return clips, n, fps, stride


def read_clips_cv2(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cv2 cannot open " + path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx_list, stride = clip_indices(n, fps)
    wanted = sorted(set(int(i) for idx in idx_list for i in idx))
    frames = {}
    cur = -1
    for w in wanted:
        if w - cur > 300:
            cap.set(cv2.CAP_PROP_POS_FRAMES, w)
            cur = w - 1
        ok, fr = False, None
        while cur < w:
            ok, fr = cap.read()
            if not ok:
                break
            cur += 1
        if ok and fr is not None:
            frames[w] = cv2.resize(
                cv2.cvtColor(fr, cv2.COLOR_BGR2RGB),
                (RES, RES),
                interpolation=cv2.INTER_AREA,
            )
    cap.release()
    if not frames:
        raise RuntimeError("cv2 decoded no frames for " + path)
    last = frames[max(frames)]
    clips = []
    for idx in idx_list:
        clip = np.stack([frames.get(int(i), last) for i in idx])
        clips.append(clip)
    return clips, n, fps, stride


def extract_group(name, folder, fallback):
    files = sorted(
        str(path)
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    all_clips, meta, failures = [], [], []
    for f in files:
        base = os.path.basename(f)
        got = None
        for cand, how in (
            (f, "decord"),
            (os.path.join(fallback, base) if fallback else None, "decord-fix"),
            (f, "cv2"),
            (os.path.join(fallback, base) if fallback else None, "cv2-fix"),
        ):
            if cand is None or not os.path.exists(cand):
                continue
            try:
                if how.startswith("decord"):
                    got = read_clips_decord(cand)
                else:
                    got = read_clips_cv2(cand)
                src = how
                break
            except Exception as e:
                print(
                    "  [warn] %s via %s failed: %s" % (base, how, repr(e)[:200]),
                    flush=True,
                )
        if got is None:
            failures.append(base)
            print("  [FAIL] %s: all readers failed" % base, flush=True)
            continue
        clips, n, fps, stride = got
        all_clips.extend(clips)
        meta.append(
            {
                "file": base,
                "frames": n,
                "fps": round(fps, 2),
                "stride": stride,
                "clips": len(clips),
                "reader": src,
            }
        )
        print(
            "  %s: %d frames @ %.2ffps stride=%d -> %d clips [%s]"
            % (base, n, fps, stride, len(clips), src),
            flush=True,
        )
    arr = (
        np.stack(all_clips)
        if all_clips
        else np.zeros((0, CLIP_LEN, RES, RES, 3), np.uint8)
    )
    return arr, meta, failures


def named_paths(values):
    result = {}
    for value in values:
        name, separator, folder = value.partition("=")
        if (
            not separator
            or not name
            or not folder
            or name in result
            or name == "GroundTruth"
        ):
            raise ValueError("Use unique NAME=/path entries (GroundTruth is reserved)")
        result[name] = Path(folder).expanduser().resolve()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--generated", nargs="+", required=True, help="NAME=/path ...")
    parser.add_argument(
        "--fallback", nargs="*", default=[], help="NAME=/reencoded/path ..."
    )
    parser.add_argument("--ids", type=int, nargs="+", default=list(SONG_IDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        generated, fallbacks = named_paths(args.generated), named_paths(args.fallback)
        folders = {
            "GroundTruth": args.reference_dir.expanduser().resolve(),
            **generated,
        }
        for name, folder in folders.items():
            actual = set(media_by_id(folder))
            if actual != set(args.ids):
                raise ValueError(
                    f"{name}: FVD requires exactly the reference song IDs; "
                    f"missing={sorted(set(args.ids) - actual)}, "
                    f"extra={sorted(actual - set(args.ids))}"
                )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    from cdfvd import fvd as cdfvd_mod
    from cdfvd.utils.metric_utils import FeatureStats

    evaluator = cdfvd_mod.cdfvd(
        model="i3d", n_real="full", n_fake="full", device=args.device
    )

    def features(clips):
        if len(clips) < 2:
            raise ValueError("FVD requires at least two clips")
        stats = FeatureStats(capture_mean_cov=True)
        return evaluator.feature_fn(
            stats, evaluator.model, clips, batchsize=8, device=args.device
        )

    results = {
        "protocol": {
            "backbone": "I3D Kinetics-400 logits (cd-fvd 0.1.1)",
            "clips_per_video": 4,
            "clip_len": 16,
            "resolution": "224x224 stretch",
            "frame_stride": "round(fps/16)",
            "reference": "GroundTruth",
        },
        "groups": {},
    }
    reference_stats = None
    for name, folder in folders.items():
        clips, metadata, failures = extract_group(
            name, str(folder), str(fallbacks[name]) if name in fallbacks else None
        )
        results["groups"][name] = {
            "n_videos": len(metadata),
            "n_clips": len(clips),
            "failures": failures,
            "videos": metadata,
        }
        write_json(args.out, results)
        if failures:
            raise RuntimeError(
                f"{name}: failed to decode {failures}; see output metadata"
            )
        stats = features(clips)
        del clips
        if name == "GroundTruth":
            reference_stats = stats
        else:
            value = evaluator.compute_fvd_from_stats(
                fake_stats=stats, real_stats=reference_stats
            )
            results["groups"][name]["fvd_vs_GT"] = round(float(value), 3)
            print(f"{name}: FVD={float(value):.3f}", flush=True)
        write_json(args.out, results)


if __name__ == "__main__":
    main()
