#!/usr/bin/env python3
"""Paired ImageBind cosine, using the transforms in evaluate/IB/inference_batch.py.

The wrapper pairs numeric song IDs, including <id> / mv_<id> name differences,
and reports failed pairs explicitly instead of falling back to retrieval mode.
"""

import argparse
import sys
from pathlib import Path

from common import SONG_IDS, media_by_id, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="+", default=list(SONG_IDS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        videos = media_by_id(args.video_dir)
        audios = media_by_id(args.audio_dir, {".wav", ".mp3", ".flac", ".m4a", ".ogg"})
        if len(set(args.ids)) != len(args.ids):
            raise ValueError("Duplicate requested IDs")
        missing = set(args.ids) - (videos.keys() & audios.keys())
        if missing:
            raise ValueError(f"Unmatched audio/video song IDs: {sorted(missing)}")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    model = imagebind_model.imagebind_huge(pretrained=True).eval().to(args.device)
    records, errors = [], []
    for sid in args.ids:
        try:
            with torch.no_grad():
                embeddings = model(
                    {
                        ModalityType.AUDIO: data.load_and_transform_audio_data(
                            [str(audios[sid])], args.device
                        ),
                        ModalityType.VISION: data.load_and_transform_video_data(
                            [str(videos[sid])], args.device
                        ),
                    }
                )
                audio = functional.normalize(embeddings[ModalityType.AUDIO], dim=-1)
                video = functional.normalize(embeddings[ModalityType.VISION], dim=-1)
                value = float((audio * video).sum().item())
            records.append(
                {
                    "id": sid,
                    "audio": str(audios[sid]),
                    "video": str(videos[sid]),
                    "ib": value,
                }
            )
            print(f"{sid}: IB={value:.6f}", flush=True)
        except Exception as error:
            errors.append({"id": sid, "error": f"{type(error).__name__}: {error}"})
    values = [record["ib"] for record in records]
    write_json(
        args.output_dir / "metrics.json",
        {
            "paired_similarity_mean": float(np.mean(values)) if values else None,
            "paired_similarity_std": float(np.std(values)) if values else None,
            "paired_similarities": values,
            "n_requested": len(args.ids),
            "n_valid": len(values),
            "records": records,
            "errors": errors,
        },
    )
    (args.output_dir / "pairs.txt").write_text(
        "".join(f"{record['audio']} {record['video']}\n" for record in records),
        encoding="utf-8",
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
