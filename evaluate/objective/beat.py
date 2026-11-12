"""Paper Beat: lyric-segment beat-pulse / optical-flow cross-correlation.

Adapted from mvmetrics/metric4c_beat.py and eval_batch.py. The segment-wide
normalization before lag slicing intentionally preserves the archived protocol.
"""

import numpy as np
from common import extracted_audio

FPS, WIDTH, MAXLAG = 8.0, 256, 6


def flow_series(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video frame rate: {path}")
    step = max(1, int(round(fps / FPS)))
    previous, magnitudes, times, index = None, [], [], 0
    try:
        while cap.grab():
            if index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                height, width = frame.shape[:2]
                gray = cv2.cvtColor(
                    cv2.resize(frame, (WIDTH, max(1, int(height * WIDTH / width)))),
                    cv2.COLOR_BGR2GRAY,
                )
                if previous is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    magnitudes.append(float(np.mean(np.linalg.norm(flow, axis=2))))
                    times.append(index / fps)
                previous = gray
            index += 1
    finally:
        cap.release()
    return np.array(times), np.array(magnitudes)


def xcorr_max(a, b):
    if len(a) < 6 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return None
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    values = []
    for lag in range(-MAXLAG, MAXLAG + 1):
        if lag < 0:
            x, y = a[-lag:], b[: len(b) + lag]
        elif lag > 0:
            x, y = a[:-lag], b[lag:]
        else:
            x, y = a, b
        if len(x) >= 6:
            values.append(float(np.mean(x * y)))
    return max(values) if values else None


def score(video, segments):
    import librosa

    times, motion = flow_series(video)
    if len(times) < 6:
        raise ValueError("Fewer than six optical-flow samples")
    grid = np.arange(0, times[-1], 1.0 / FPS)
    envelope = np.interp(grid, times, motion)
    with extracted_audio(video, 22050) as wav:
        audio, sr = librosa.load(wav, sr=22050, mono=True)
    _, beats = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    pulse = np.zeros_like(grid)
    for time in beats:
        pulse += np.exp(-0.5 * ((grid - time) / 0.10) ** 2)
    values = []
    for segment in segments:
        mask = (grid >= segment["start"]) & (grid < segment["end"])
        value = xcorr_max(pulse[mask], envelope[mask])
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError(
            "No valid lyric segments with nonconstant beat and motion signals"
        )
    return {
        "beat": round(float(np.mean(values)), 4),
        "n_seg": len(values),
        "decoded_duration": round(float(times[-1]), 3),
    }
