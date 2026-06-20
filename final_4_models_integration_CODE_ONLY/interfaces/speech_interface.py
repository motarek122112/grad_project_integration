"""
Speech-to-Text Inference Interface
==================================
CTC-based CNN + BiLSTM speech recognition model.

Pipeline:
    audio file -> 16 kHz mono waveform -> 80-bin mel spectrogram -> CNN + BiLSTM -> CTC decode -> text

Expected checkpoint:
    - A state_dict with keys like cnn.*, bilstm.*, fc.*
    - Can be .pth, .pt, or a torch-save zip archive such as best_speech_model.pth.zip

Usage from Python:
    from interfaces.speech_interface import load_speech_interface

    stt = load_speech_interface("models/best_speech_model.pth.zip")
    result = stt.predict("samples/sample.wav")
    print(result["text"])

Usage from command line:
    python interfaces/speech_interface.py \
        --checkpoint models/best_speech_model.pth.zip \
        --audio samples/sample.wav \
        --decoder beam \
        --json_output outputs/speech_result.json
"""

import os
import json
import math
from typing import Dict, List, Optional, Union, Any

import numpy as np
import soundfile as sf
import librosa
import torch
from torch import nn


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio preprocessing settings used during training
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160

# CTC decoding settings
BEAM_WIDTH = 10
CHARS = list("abcdefghijklmnopqrstuvwxyz '")
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
BLANK_LABEL = 0
VOCAB_SIZE = len(CHARS) + 1  # 29 classes including blank


def indices_to_text(indices: List[int]) -> str:
    """Convert decoded class indices to text."""
    return "".join(IDX_TO_CHAR[i] for i in indices if i in IDX_TO_CHAR)


class SpeechModel(nn.Module):
    """
    CNN + BiLSTM + CTC acoustic model.

    Input shape:
        x: [B, N_MELS, T]

    Output shape:
        log_probs: [B, T_out, VOCAB_SIZE]
    """
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d((2, 1)),
        )

        # After 3 MaxPool2d((2,1)) layers: 80 mel bins -> 10 bins
        self.lstm_input_size = 128 * 10

        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, VOCAB_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, 1, 80, T]
        x = self.cnn(x)     # [B, 128, 10, T_out]

        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2)      # [B, T_out, C, F]
        x = x.reshape(b, t, c * f)     # [B, T_out, 1280]

        x, _ = self.bilstm(x)          # [B, T_out, 512]
        x = self.dropout(x)
        x = self.fc(x)                 # [B, T_out, 29]
        return x.log_softmax(dim=2)


class SpeechToTextInterface:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        beam_width: int = BEAM_WIDTH,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.beam_width = beam_width

        self.checkpoint = self._load_checkpoint(checkpoint_path)
        self.model = SpeechModel()
        self._load_model_state(self.model, self.checkpoint)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_checkpoint(checkpoint_path: str) -> Any:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # torch.load can load normal .pth/.pt and torch-save zip archives directly.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint

    @staticmethod
    def _load_model_state(model: nn.Module, checkpoint: Any) -> None:
        """Load state_dict from several common checkpoint formats."""
        if isinstance(checkpoint, dict):
            if "model_state" in checkpoint:
                state = checkpoint["model_state"]
            elif "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state = checkpoint["model"]
            else:
                # In this project checkpoint is an OrderedDict state_dict directly.
                state = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format. Expected a PyTorch state_dict or checkpoint dict.")

        # Remove DataParallel prefix if present
        cleaned_state = {}
        for k, v in state.items():
            new_k = k.replace("module.", "", 1) if k.startswith("module.") else k
            cleaned_state[new_k] = v

        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if missing:
            print(f"[Warning] Missing keys while loading speech model: {len(missing)}")
        if unexpected:
            print(f"[Warning] Unexpected keys while loading speech model: {len(unexpected)}")

    @staticmethod
    def _normalize_spec(spec: np.ndarray) -> np.ndarray:
        mean = float(spec.mean())
        std = float(spec.std())
        return (spec - mean) / (std + 1e-5)

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio and convert it to normalized log-mel spectrogram."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        waveform, sr = sf.read(audio_path)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(np.float32)

        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        # Avoid all-zero / extremely quiet problems
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / max(1.0, np.max(np.abs(waveform)))

        spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            power=2.0,
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = self._normalize_spec(spec).astype(np.float32)

        return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # [1, 80, T]

    @staticmethod
    def greedy_decode(log_probs: torch.Tensor, blank: int = BLANK_LABEL) -> List[str]:
        """Basic CTC greedy decoding."""
        pred_ids = torch.argmax(log_probs, dim=2)
        results = []

        for seq in pred_ids:
            decoded = []
            prev = blank
            for idx in seq.tolist():
                if idx != blank and idx != prev:
                    decoded.append(idx)
                prev = idx
            results.append(indices_to_text(decoded))

        return results

    @staticmethod
    def _log_sum_exp(a: float, b: float) -> float:
        if a == -float("inf"):
            return b
        if b == -float("inf"):
            return a
        if a > b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))

    @classmethod
    def _ctc_beam_search_single(
        cls,
        log_probs: torch.Tensor,
        beam_width: int = BEAM_WIDTH,
        blank: int = BLANK_LABEL,
    ) -> str:
        """Small CTC beam search decoder without external language model."""
        beams = {("", blank): 0.0}

        for t in range(log_probs.size(0)):
            new_beams = {}

            for (prefix, last_char), score in beams.items():
                # Limit candidates per frame for speed
                values, indices = torch.topk(log_probs[t], k=min(12, log_probs.size(1)))
                for lp, c_tensor in zip(values, indices):
                    c = int(c_tensor.item())
                    new_score = score + float(lp.item())

                    if c == blank:
                        key = (prefix, blank)
                    else:
                        char = IDX_TO_CHAR.get(c, "")
                        new_prefix = prefix if c == last_char else prefix + char
                        key = (new_prefix, c)

                    if key not in new_beams:
                        new_beams[key] = new_score
                    else:
                        new_beams[key] = cls._log_sum_exp(new_beams[key], new_score)

            sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
            beams = dict(sorted_beams[:beam_width])

        return max(beams.items(), key=lambda x: x[1])[0][0].strip()

    @classmethod
    def beam_decode(
        cls,
        log_probs: torch.Tensor,
        beam_width: int = BEAM_WIDTH,
        blank: int = BLANK_LABEL,
    ) -> List[str]:
        return [
            cls._ctc_beam_search_single(log_probs[i], beam_width=beam_width, blank=blank)
            for i in range(log_probs.size(0))
        ]

    @staticmethod
    def estimate_confidence(log_probs: torch.Tensor, blank: int = BLANK_LABEL) -> Optional[float]:
        """
        Rough confidence estimate from max probabilities on non-blank frames.
        This is not a calibrated probability, just a helpful diagnostic score.
        """
        probs = log_probs.exp()
        max_probs, pred_ids = probs.max(dim=2)  # [B, T]
        scores = []
        for mp, ids in zip(max_probs, pred_ids):
            mask = ids != blank
            if mask.any():
                scores.append(float(mp[mask].mean().item()))
            else:
                scores.append(float(mp.mean().item()))
        return float(np.mean(scores)) if scores else None

    @torch.no_grad()
    def predict(self, audio_path: str, use_beam: bool = True) -> Dict[str, Union[str, float, None, Dict[str, Union[str, int, float, None]]]]:
        x = self.preprocess_audio(audio_path).to(self.device)
        log_probs = self.model(x)

        if use_beam:
            text = self.beam_decode(log_probs.cpu(), beam_width=self.beam_width, blank=BLANK_LABEL)[0]
            decoder_name = "beam"
        else:
            text = self.greedy_decode(log_probs.cpu(), blank=BLANK_LABEL)[0].strip()
            decoder_name = "greedy"

        confidence = self.estimate_confidence(log_probs.cpu(), blank=BLANK_LABEL)

        return {
            "text": text,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "meta": {
                "decoder": decoder_name,
                "sample_rate": self.sample_rate,
                "n_mels": self.n_mels,
                "beam_width": self.beam_width if use_beam else None,
                "vocab_size": VOCAB_SIZE,
                "blank_label": BLANK_LABEL,
            },
        }

    def get_model_info(self) -> Dict[str, Union[str, int, None]]:
        epoch = None
        stage = None
        if isinstance(self.checkpoint, dict):
            epoch = self.checkpoint.get("epoch")
            stage = self.checkpoint.get("stage")

        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "model_type": "CNN + 3-layer BiLSTM + CTC",
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "vocab_size": VOCAB_SIZE,
            "blank_label": BLANK_LABEL,
            "characters": "".join(CHARS),
            "checkpoint_epoch": epoch,
            "checkpoint_stage": stage,
        }


def load_speech_interface(
    checkpoint_path: str,
    device: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    beam_width: int = BEAM_WIDTH,
) -> SpeechToTextInterface:
    return SpeechToTextInterface(
        checkpoint_path=checkpoint_path,
        device=device,
        sample_rate=sample_rate,
        n_mels=n_mels,
        beam_width=beam_width,
    )


def speech_to_text(
    audio_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
    use_beam: bool = True,
) -> Dict[str, Union[str, float, None, Dict[str, Union[str, int, float, None]]]]:
    interface = load_speech_interface(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return interface.predict(audio_path=audio_path, use_beam=use_beam)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speech-to-text inference interface")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth/.pt checkpoint or torch zip archive")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--decoder", type=str, default="beam", choices=["beam", "greedy"])
    parser.add_argument("--beam_width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--json_output", type=str, default=None, help="Optional path to save result JSON")
    args = parser.parse_args()

    interface = load_speech_interface(
        checkpoint_path=args.checkpoint,
        device=args.device,
        beam_width=args.beam_width,
    )

    info = interface.get_model_info()
    result = interface.predict(
        audio_path=args.audio,
        use_beam=(args.decoder == "beam"),
    )

    payload = {
        "model_info": info,
        "prediction": result,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON result to: {args.json_output}")
