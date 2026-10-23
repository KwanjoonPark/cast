"""Shared CLAP scorer: the exact convention validated in CAST_CLAP_EVALUATION_REPORT.md.

clap_cosine = dot(L2norm(ClapModel.get_audio_features(audio)), L2norm(ClapModel.get_text_features(text)))
Audio: 16 kHz int16 wav -> float32/32768 -> torchaudio sinc resample to 48 kHz (no clamping) -> official ClapProcessor.
Model laion/clap-htsat-unfused (fp32, eval, no_grad), loaded from the local HF snapshot with token=False.
"""
import hashlib, os
import numpy as np, torch, torchaudio
from scipy.io.wavfile import read as wav_read

MODEL = "laion/clap-htsat-unfused"


def load_audio_48k(path):
    sr, x = wav_read(path)
    info = {"input_sr": int(sr), "input_dtype": str(x.dtype), "input_len": int(len(x)), "channels": 1 if x.ndim == 1 else int(x.shape[1])}
    if x.ndim == 2: x = x.mean(axis=1)
    xf = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    info["input_peak"] = float(np.abs(xf).max())
    y = torchaudio.functional.resample(torch.from_numpy(xf), sr, 48000)
    info.update({"clap_sr": 48000, "resampled_len": int(y.numel()), "resampled_peak": float(y.abs().max())})
    return y.numpy(), info


class ClapScorer:
    def __init__(self, device="cuda"):
        from huggingface_hub import snapshot_download
        from transformers import ClapModel, ClapProcessor
        local = snapshot_download(MODEL, token=False)
        self.device = device
        self.model = ClapModel.from_pretrained(local).to(device).eval()
        self.proc = ClapProcessor.from_pretrained(local)
        fe = self.proc.feature_extractor
        self.info = {"model": MODEL, "snapshot": os.path.basename(local), "param_dtype": str(next(self.model.parameters()).dtype), "device": device,
                     "feature_extractor": {k: getattr(fe, k, None) for k in ["sampling_rate", "max_length_s", "nb_max_samples", "truncation", "padding"]},
                     "logit_scale_audio_exp": float(self.model.logit_scale_a.exp()),
                     "weights_sha256": hashlib.sha256(open(os.path.join(local, "pytorch_model.bin"), "rb").read()).hexdigest()}
        self._text_cache = {}

    @torch.no_grad()
    def text_embedding(self, text):
        if text not in self._text_cache:
            t = self.proc(text=[text], return_tensors="pt", padding=True).to(self.device)
            self._text_cache[text] = torch.nn.functional.normalize(self.model.get_text_features(**t), dim=-1)
        return self._text_cache[text]

    @torch.no_grad()
    def audio_embedding(self, wave48):
        a = self.proc(audios=[wave48], sampling_rate=48000, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.model.get_audio_features(**a), dim=-1)

    def score(self, audio_path, texts):
        """Returns ({text: cosine}, audio_info)."""
        w, info = load_audio_48k(audio_path); ae = self.audio_embedding(w)
        return {t: float((ae * self.text_embedding(t)).sum()) for t in texts}, info
