"""Paper CLAP: song audio versus the MLLM video caption, not lyric text.

Adapted from mvmetrics/metric3c_clap.py: first 60 seconds, 10-second windows,
48 kHz mono, normalize each window, mean-pool, then normalize again.
"""

from common import extracted_audio

CHECKPOINT = "laion/clap-htsat-unfused"


class ClapScorer:
    def __init__(self, device="cpu"):
        from transformers import ClapModel, ClapProcessor

        self.device = device
        self.model = ClapModel.from_pretrained(CHECKPOINT).to(device).eval()
        self.processor = ClapProcessor.from_pretrained(CHECKPOINT)

    def score(self, video, caption):
        import librosa
        import torch
        import torch.nn.functional as functional

        with extracted_audio(video, 48000, duration=60) as wav:
            audio, _ = librosa.load(wav, sr=48000, mono=True)
        windows = [
            audio[start : start + 48000 * 10]
            for start in range(0, min(len(audio), 48000 * 60), 48000 * 10)
        ]
        windows = [window for window in windows if len(window) >= 48000]
        if not windows or not caption.strip():
            raise ValueError(
                "CLAP requires at least one second of audio and a video caption"
            )
        with torch.no_grad():
            embeddings = []
            for window in windows:
                inputs = self.processor(
                    audios=window, sampling_rate=48000, return_tensors="pt"
                ).to(self.device)
                embedding = self.model.get_audio_features(**inputs)
                embeddings.append(functional.normalize(embedding, dim=-1))
            audio_embedding = functional.normalize(torch.cat(embeddings).mean(0), dim=0)
            inputs = self.processor(
                text=[caption], return_tensors="pt", padding=True
            ).to(self.device)
            text_embedding = functional.normalize(
                self.model.get_text_features(**inputs), dim=-1
            )[0]
        return {
            "clap": round(float(audio_embedding @ text_embedding), 4),
            "text": caption,
            "checkpoint": CHECKPOINT,
        }
