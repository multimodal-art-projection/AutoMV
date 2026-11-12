import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beat import xcorr_max  # noqa: E402 - import the standalone CLI modules
from common import media_by_id  # noqa: E402 - import the standalone CLI modules
from fvd_score import clip_indices  # noqa: E402 - import the standalone CLI modules
from lipsync_score import distance_and_confidence  # noqa: E402 - import the standalone CLI modules
from summarize import build_tables, verify_archive  # noqa: E402 - import the standalone CLI modules


class ArchiveTests(unittest.TestCase):
    def test_table2_reproduces_archived_metrics(self):
        verify_archive()
        per_song, summaries = build_tables()
        self.assertEqual(len(per_song), 150)
        by_method = {row["method"]: row for row in summaries}
        with (ROOT / "results/paper_table2.csv").open() as stream:
            paper = list(csv.DictReader(stream))
        for row in paper:
            actual = by_method[row["method"]]
            for metric in ["IB_percent", "CLAP_percent", "Beat_percent", "FVD"]:
                if row[metric]:
                    digits = 0 if metric == "FVD" else 1
                    self.assertEqual(round(actual[metric], digits), float(row[metric]))

    def test_missing_score_stays_missing(self):
        per_song, summaries = build_tables()
        missing = [row for row in per_song if row["CLAP_percent"] is None]
        self.assertEqual(
            [(r["method"], r["song_id"]) for r in missing], [("Music2Video", 38)]
        )
        music2video = next(r for r in summaries if r["method"] == "Music2Video")
        self.assertEqual(music2video["CLAP_n"], 29)


class ProtocolTests(unittest.TestCase):
    def test_beat_accepts_lagged_motion_and_rejects_constant_motion(self):
        beats = np.zeros(80)
        beats[[15, 35, 55, 75]] = 1
        lagged = np.roll(beats, 3)
        unrelated = np.random.default_rng(123).normal(size=80)
        self.assertGreater(xcorr_max(beats, lagged), 0.9)
        self.assertGreater(xcorr_max(beats, lagged), xcorr_max(beats, unrelated) + 0.5)
        self.assertIsNone(xcorr_max(beats, np.ones(80)))
        self.assertIsNone(xcorr_max(beats[:5], lagged[:5]))

    def test_fvd_sampling_and_short_video_bounds(self):
        clips, stride = clip_indices(300, 30)
        self.assertEqual(stride, 2)
        self.assertEqual(len(clips), 4)
        self.assertTrue(all(len(clip) == 16 for clip in clips))
        self.assertEqual(int(clips[0][0]), 0)
        self.assertEqual(int(clips[-1][-1]), 298)
        clips, _ = clip_indices(5, 25)
        self.assertTrue(all(np.all((clip >= 0) & (clip < 5)) for clip in clips))
        with self.assertRaises(ValueError):
            clip_indices(0, 30)

    def test_syncnet_reduces_windows_before_shift_selection(self):
        distance, confidence = distance_and_confidence([[1, 8, 4], [9, 2, 4]])
        self.assertEqual(distance, 4)
        self.assertEqual(confidence, 1)

    def test_duplicate_song_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "4.mp4").touch()
            (Path(folder) / "mv_4.mp4").touch()
            with self.assertRaises(ValueError):
                media_by_id(folder)

    def test_beat_only_cli_requires_no_api_or_model_packages(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg is required for the media integration test")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = folder / "4.mp4"
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=128x96:rate=8:duration=6",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=22050:duration=6,"
                    "volume='if(lt(mod(t,0.5),0.08),1,0)':eval=frame",
                    "-c:v",
                    "mpeg4",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(video),
                ],
                check=True,
            )
            segments = folder / "segments"
            segments.mkdir()
            (segments / "4.json").write_text('{"segments":[{"start":0,"end":6}]}')
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in {"GEMINI_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"}
            }
            command = [
                sys.executable,
                str(ROOT / "run.py"),
                "--dir",
                str(folder),
                "--name",
                "smoke",
                "--ids",
                "4",
                "--metrics",
                "beat",
                "--segments-dir",
                str(segments),
                "--output-dir",
                str(folder / "out"),
            ]
            result = subprocess.run(
                command, env=environment, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = folder / "out/beat_smoke.jsonl"
            first = output.read_bytes()
            result = subprocess.run(
                command, env=environment, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
