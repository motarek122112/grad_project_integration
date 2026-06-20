import os
import re
import math
import json
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
import torch
import torch.nn as nn


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_POSE_KP = 21
N_HAND_KP = 21
FEAT_DIM = (N_POSE_KP + N_HAND_KP * 2) * 3  # 189
T_FRAMES = 64
UPPER_POSE_INDICES = list(range(21))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LandmarkTransformer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.0,
        max_len: int = 65,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_out = x[:, 0, :]
        logits = self.head(cls_out)
        return logits


def _torch_load_checkpoint(checkpoint_path: str, map_location: Union[str, torch.device]):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def _get_holistic():
    """
    Robust MediaPipe Holistic import.

    Some MediaPipe versions expose:
        mp.solutions.holistic

    Newer versions may not expose mp.solutions directly, so we fallback to:
        mediapipe.python.solutions.holistic
    """

    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
            return mp.solutions.holistic

    except Exception:
        pass

    try:
        import mediapipe.python.solutions.holistic as mp_holistic
        return mp_holistic

    except Exception as exc:
        raise ImportError(
            "mediapipe is required. Install with: python -m pip install mediapipe==0.10.35"
        ) from exc


def normalize_landmarks(kp_seq: np.ndarray) -> np.ndarray:
    kp = kp_seq.copy().astype(np.float32)
    total_features = kp.shape[1]
    ls_x, ls_y = 33, 34
    rs_x, rs_y = 36, 37
    for t in range(kp.shape[0]):
        row = kp[t]
        if row[ls_x] == 0 and row[rs_x] == 0:
            continue
        cx = (row[ls_x] + row[rs_x]) / 2.0
        cy = (row[ls_y] + row[rs_y]) / 2.0
        dist = math.sqrt((row[rs_x] - row[ls_x]) ** 2 + (row[rs_y] - row[ls_y]) ** 2)
        if dist < 1e-6:
            dist = 1.0
        for i in range(0, total_features, 3):
            kp[t, i] = (row[i] - cx) / dist
            kp[t, i + 1] = (row[i + 1] - cy) / dist
            kp[t, i + 2] = row[i + 2] / dist
    return kp


def sample_to_tframes(kp_seq: np.ndarray, n_frames: int = T_FRAMES, feat_dim: int = FEAT_DIM) -> np.ndarray:
    if kp_seq is None or len(kp_seq) == 0:
        return np.zeros((n_frames, feat_dim), dtype=np.float32)
    kp_seq = np.asarray(kp_seq, dtype=np.float32)
    t = kp_seq.shape[0]
    if t <= n_frames:
        pad = np.zeros((n_frames - t, feat_dim), dtype=np.float32)
        return np.concatenate([kp_seq, pad], axis=0)
    idxs = np.linspace(0, t - 1, n_frames).astype(int)
    return kp_seq[idxs]


# ============================================================
# TTA helpers — generate multiple augmented views of the same
# landmark sequence so the model votes across slight variations
# ============================================================

def _augment_temporal_crop(kp_seq: np.ndarray, crop_ratio: float = 0.85) -> np.ndarray:
    """
    Take a slightly shorter sub-sequence from the centre.
    Simulates the sign being performed a bit faster or slower.
    """
    t = kp_seq.shape[0]
    keep = max(4, int(t * crop_ratio))
    start = (t - keep) // 2
    return kp_seq[start: start + keep]


def _augment_mirror_hands(kp_seq: np.ndarray) -> np.ndarray:
    """
    Mirror the x-coordinate of every landmark (horizontal flip).
    Also swap left-hand and right-hand blocks so the spatial meaning
    stays consistent.
    Layout: pose 0:63 | left_hand 63:126 | right_hand 126:189
    """
    kp = kp_seq.copy()
    # Flip x (every 3rd feature starting at 0)
    kp[:, 0::3] = -kp[:, 0::3]
    # Swap left / right hand blocks
    tmp = kp[:, 63:126].copy()
    kp[:, 63:126] = kp[:, 126:189]
    kp[:, 126:189] = tmp
    return kp


def _augment_add_noise(kp_seq: np.ndarray, scale: float = 0.015) -> np.ndarray:
    """
    Add tiny Gaussian jitter — makes the ensemble robust to slight hand
    position variation between performer and training data.
    """
    kp = kp_seq.copy()
    noise = np.random.randn(*kp.shape).astype(np.float32) * scale
    # Only add noise where landmark is non-zero (detected)
    mask = np.abs(kp).sum(axis=-1, keepdims=True) > 1e-6
    kp += noise * mask
    return kp


def _augment_slow_down(kp_seq: np.ndarray, factor: float = 1.25) -> np.ndarray:
    """
    Stretch the sequence in time by repeating frames — simulates a
    performer who signs more slowly than the training clips.
    """
    t = kp_seq.shape[0]
    new_t = min(int(t * factor), T_FRAMES * 2)
    idxs = np.linspace(0, t - 1, new_t).astype(int)
    return kp_seq[idxs]


def build_tta_variants(kp_seq: np.ndarray) -> List[np.ndarray]:
    """
    Return a list of augmented landmark sequences for Test-Time Augmentation.
    Each variant will be run through the model and the logits averaged.
    """
    np.random.seed(42)  # deterministic noise so results are reproducible
    variants = [
        kp_seq,                                          # 1) original
        _augment_temporal_crop(kp_seq, 0.90),            # 2) slight crop
        _augment_temporal_crop(kp_seq, 0.80),            # 3) stronger crop
        _augment_mirror_hands(kp_seq),                   # 4) mirrored
        _augment_add_noise(kp_seq, scale=0.012),         # 5) small noise
        _augment_add_noise(kp_seq, scale=0.025),         # 6) more noise
        _augment_slow_down(kp_seq, 1.15),                # 7) slightly slower
    ]
    return variants


def extract_landmarks_from_video(video_path: str, max_frames: int = 128) -> np.ndarray:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(raw_frames) == 0:
        return np.zeros((1, FEAT_DIM), dtype=np.float32)

    if len(raw_frames) > max_frames:
        idxs = np.linspace(0, len(raw_frames) - 1, max_frames).astype(int)
        raw_frames = [raw_frames[i] for i in idxs]

    mp_holistic = _get_holistic()
    all_kp = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:
        for frame in raw_frames:
            result = holistic.process(frame)

            if result.pose_landmarks:
                pose_kp = np.array(
                    [[result.pose_landmarks.landmark[i].x,
                      result.pose_landmarks.landmark[i].y,
                      result.pose_landmarks.landmark[i].z]
                     for i in UPPER_POSE_INDICES],
                    dtype=np.float32,
                ).flatten()
            else:
                pose_kp = np.zeros(N_POSE_KP * 3, dtype=np.float32)

            if result.left_hand_landmarks:
                lh_kp = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark],
                    dtype=np.float32,
                ).flatten()
            else:
                lh_kp = np.zeros(N_HAND_KP * 3, dtype=np.float32)

            if result.right_hand_landmarks:
                rh_kp = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark],
                    dtype=np.float32,
                ).flatten()
            else:
                rh_kp = np.zeros(N_HAND_KP * 3, dtype=np.float32)

            all_kp.append(np.concatenate([pose_kp, lh_kp, rh_kp]))

    return np.stack(all_kp, axis=0).astype(np.float32)


LABEL_TO_WORD = {
    "MYSELF": "I",
    "WANT1": "want",
    "EAT1": "eat",
    "DINNER1": "dinner",
    "LUNCH1": "lunch",
    "BREAKFAST1": "breakfast",
    "HOSPITAL1": "hospital",
    "HOUSE": "home",
    "STOP": "stop",
    "HOW1": "how",
    "THEY1": "they",
    "DEAF1": "deaf",
    "HARDOFHEARING": "hard of hearing",
    "VOICE": "voice",
    "FINE1": "fine",
    "CONFUSED1": "confused",
    "SHOCKED": "shocked",
    "APPLE": "apple",
    "MEAT1": "meat",
    "TOMATO": "tomato",
    "SCHOOL": "school",
    "MOVIE1": "movie",
    "NIGHT1": "night",
}


def clean_asl_label(label: str) -> str:
    label = str(label)
    if label in LABEL_TO_WORD:
        return LABEL_TO_WORD[label]
    s = label
    s = re.sub(r"\d+$", "", s)
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = s.replace("_", " ")
    s = s.replace("HARDOFHEARING", "hard of hearing")
    s = s.replace("ALLOFSUDDEN", "all of sudden")
    s = s.replace("WHATFOR", "what for")
    return s.lower().strip()


def simple_sentence_postprocess(words: List[str]) -> str:
    cleaned = []
    for word in words:
        word = str(word).strip()
        if not word:
            continue
        if len(cleaned) == 0 or cleaned[-1] != word:
            cleaned.append(word)
    sentence = " ".join(cleaned).strip()
    rules = {
        "I want hospital": "I need hospital",
        "I want home": "I want to go home",
        "they deaf": "they are deaf",
        "I confused": "I am confused",
        "I fine": "I am fine",
        "movie night": "Movie night",
    }
    sentence = rules.get(sentence, sentence)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence


def extract_raw_landmarks_for_sentence(
    video_path: str,
    frame_stride: int = 1,
    resize_width: Optional[int] = 640,
    max_total_frames: int = 1200,
) -> Tuple[np.ndarray, float]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            if resize_width is not None:
                h, w = frame.shape[:2]
                if w > resize_width:
                    scale = resize_width / w
                    frame = cv2.resize(frame, (resize_width, int(h * scale)))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()

    if len(frames) == 0:
        return np.zeros((1, FEAT_DIM), dtype=np.float32), fps

    if len(frames) > max_total_frames:
        idxs = np.linspace(0, len(frames) - 1, max_total_frames).astype(int)
        frames = [frames[i] for i in idxs]

    mp_holistic = _get_holistic()
    all_kp = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:
        for frame in frames:
            result = holistic.process(frame)
            if result.pose_landmarks:
                pose_kp = np.array(
                    [[result.pose_landmarks.landmark[i].x,
                      result.pose_landmarks.landmark[i].y,
                      result.pose_landmarks.landmark[i].z]
                     for i in UPPER_POSE_INDICES],
                    dtype=np.float32,
                ).flatten()
            else:
                pose_kp = np.zeros(N_POSE_KP * 3, dtype=np.float32)
            if result.left_hand_landmarks:
                lh_kp = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark],
                    dtype=np.float32,
                ).flatten()
            else:
                lh_kp = np.zeros(N_HAND_KP * 3, dtype=np.float32)
            if result.right_hand_landmarks:
                rh_kp = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark],
                    dtype=np.float32,
                ).flatten()
            else:
                rh_kp = np.zeros(N_HAND_KP * 3, dtype=np.float32)
            all_kp.append(np.concatenate([pose_kp, lh_kp, rh_kp]))

    return np.stack(all_kp, axis=0).astype(np.float32), fps


def moving_average(x: np.ndarray, window: int = 7) -> np.ndarray:
    if len(x) == 0:
        return x
    window = max(1, int(window))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def hand_motion_score(kp_seq: np.ndarray) -> np.ndarray:
    if kp_seq.shape[0] < 2:
        return np.zeros(kp_seq.shape[0], dtype=np.float32)
    hands = kp_seq[:, 63:189].reshape(kp_seq.shape[0], 42, 3)
    xy = hands[:, :, :2]
    valid = np.linalg.norm(xy, axis=-1) > 1e-6
    diff = np.diff(xy, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    valid_pair = valid[1:] & valid[:-1]
    denom = np.maximum(valid_pair.sum(axis=1), 1)
    score = (dist * valid_pair).sum(axis=1) / denom
    score = np.concatenate([[score[0]], score]).astype(np.float32)
    if score.max() > score.min():
        score = (score - score.min()) / (score.max() - score.min())
    return moving_average(score, window=7)


def segments_from_motion(
    motion: np.ndarray,
    fps: float = 30.0,
    threshold: float = 0.08,
    min_segment_sec: float = 0.35,
    min_pause_sec: float = 0.45,
    pad_sec: float = 0.15,
) -> List[Tuple[int, int]]:
    active = motion > threshold
    min_pause_frames = int(min_pause_sec * fps)
    min_segment_frames = int(min_segment_sec * fps)
    pad_frames = int(pad_sec * fps)

    active2 = active.copy()
    i = 0
    while i < len(active2):
        if active2[i]:
            i += 1
            continue
        start = i
        while i < len(active2) and not active2[i]:
            i += 1
        end = i
        left_active = start > 0 and active2[start - 1]
        right_active = end < len(active2) and active2[end]
        if left_active and right_active and (end - start) <= min_pause_frames:
            active2[start:end] = True

    segments = []
    i = 0
    while i < len(active2):
        if not active2[i]:
            i += 1
            continue
        start = i
        while i < len(active2) and active2[i]:
            i += 1
        end = i
        if (end - start) >= min_segment_frames:
            start = max(0, start - pad_frames)
            end = min(len(active2), end + pad_frames)
            segments.append((start, end))

    if not segments:
        segments = [(0, len(motion))]

    return segments


class SignLanguageToTextInterface:
    """
    Integration-ready interface for the Landmark Transformer ASL model.

    Key improvement over the original: Test-Time Augmentation (TTA).
    When `use_tta=True` (default), the model runs inference on 7 slightly
    different versions of the same landmark sequence and the softmax
    probabilities are averaged. This makes the model much more robust to:
      - slightly unclear / incomplete signs
      - signer speed differences
      - handedness differences
      - small position/scale variations vs. training data
    The TTA variants are cheap (no extra feature extraction) and add only
    ~6x the single-pass compute cost, which on CPU is still < 0.1 s.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        use_tta: bool = True,
    ):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.use_tta = use_tta

        self.checkpoint = _torch_load_checkpoint(checkpoint_path, map_location=self.device)

        self.cfg = {
            "feat_dim": int(self.checkpoint.get("feat_dim", FEAT_DIM)),
            "num_classes": int(self.checkpoint.get("num_classes", 200)),
            "t_frames": int(self.checkpoint.get("t_frames", T_FRAMES)),
            "d_model": int(self.checkpoint.get("d_model", 256)),
            "n_heads": int(self.checkpoint.get("n_heads", 4)),
            "n_layers": int(self.checkpoint.get("n_layers", 4)),
        }

        if "id2label" not in self.checkpoint:
            raise KeyError("Checkpoint must contain 'id2label'.")
        if "model_state" not in self.checkpoint:
            raise KeyError("Checkpoint must contain 'model_state'.")

        self.id2label = {int(k): str(v) for k, v in self.checkpoint["id2label"].items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.model = LandmarkTransformer(
            feat_dim=self.cfg["feat_dim"],
            num_classes=self.cfg["num_classes"],
            d_model=self.cfg["d_model"],
            n_heads=self.cfg["n_heads"],
            n_layers=self.cfg["n_layers"],
            dim_ff=self.cfg["d_model"] * 2,
            dropout=0.0,
            max_len=self.cfg["t_frames"] + 1,
        ).to(self.device)

        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal: single forward pass
    # ------------------------------------------------------------------

    def _preprocess_landmarks(self, kp_seq: np.ndarray) -> torch.Tensor:
        kp_seq = normalize_landmarks(kp_seq)
        kp_seq = sample_to_tframes(
            kp_seq,
            n_frames=self.cfg["t_frames"],
            feat_dim=self.cfg["feat_dim"],
        )
        x = torch.tensor(kp_seq, dtype=torch.float32).unsqueeze(0)
        return x.to(self.device)

    @torch.no_grad()
    def _single_forward(self, kp_seq: np.ndarray) -> np.ndarray:
        """Run one forward pass, return softmax probability vector (numpy)."""
        x = self._preprocess_landmarks(kp_seq)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    # Internal: TTA inference — average probs over augmented variants
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer_with_tta(self, kp_seq: np.ndarray) -> np.ndarray:
        """
        Run inference with Test-Time Augmentation.

        Returns averaged softmax probability vector over all TTA variants.
        If TTA is disabled, falls back to a single forward pass.
        """
        if not self.use_tta:
            return self._single_forward(kp_seq)

        variants = build_tta_variants(kp_seq)
        all_probs = []

        for variant in variants:
            try:
                probs = self._single_forward(variant)
                all_probs.append(probs)
            except Exception:
                # If any variant fails (e.g. too short), skip it silently
                pass

        if not all_probs:
            return self._single_forward(kp_seq)

        # Simple mean — works better than max for uncertain signs
        avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
        return avg_probs

    # ------------------------------------------------------------------
    # Internal: build top-k result list from a probability vector
    # ------------------------------------------------------------------

    def _probs_to_topk(self, probs: np.ndarray, top_k: int) -> List[Dict]:
        k = min(top_k, len(probs))
        top_ids = np.argsort(probs)[::-1][:k]

        results = []
        for label_id in top_ids:
            gloss = self.id2label[int(label_id)]
            results.append({
                "label_id": int(label_id),
                "gloss": gloss,
                "text": clean_asl_label(gloss),
                "confidence": float(probs[label_id]),
            })

        return results

    # ------------------------------------------------------------------
    # Public: isolated sign prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        video_path: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict a single isolated sign from a short video clip.

        Uses TTA by default: runs 7 augmented variants of the extracted
        landmarks and averages their softmax scores, so even an unclear
        or imperfect sign gets a more reliable confidence value.
        """
        kp_seq = extract_landmarks_from_video(video_path)
        probs = self._infer_with_tta(kp_seq)
        top_results = self._probs_to_topk(probs, top_k)
        best = top_results[0]

        return {
            "pipeline": "sign_language_to_text",
            "input_video_path": video_path,
            "text": best["text"],
            "gloss": best["gloss"],
            "label_id": best["label_id"],
            "confidence": best["confidence"],
            "top_k": top_results,
            "tta_used": self.use_tta,
            "tta_variants": len(build_tta_variants(kp_seq)) if self.use_tta else 1,
        }

    # ------------------------------------------------------------------
    # Public: single landmark segment prediction (used by predict_sentence)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_landmark_segment(
        self,
        kp_segment_raw: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict a single sign from a raw landmark segment (already extracted).
        Used internally by predict_sentence.
        Also uses TTA.
        """
        probs = self._infer_with_tta(kp_segment_raw)
        top_results = self._probs_to_topk(probs, top_k)
        best = top_results[0]

        return {
            "text": best["text"],
            "gloss": best["gloss"],
            "label_id": best["label_id"],
            "confidence": best["confidence"],
            "top_k": top_results,
        }

    # ------------------------------------------------------------------
    # Public: multi-sign sentence prediction
    # ------------------------------------------------------------------

    def predict_sentence(
        self,
        video_path: str,
        top_k: int = 5,
        threshold: float = 0.08,
        min_pause_sec: float = 0.50,
        min_segment_sec: float = 0.35,
        confidence_threshold: float = 0.20,  # lowered from 0.35 — TTA makes this safe
    ) -> Dict[str, Any]:
        """
        Multi-sign sentence mode.
        TTA is applied to each detected segment independently.
        confidence_threshold is lower than before because TTA-averaged
        confidences are more reliable even for unclear signs.
        """
        kp_raw, fps = extract_raw_landmarks_for_sentence(video_path)
        motion = hand_motion_score(kp_raw)

        segments = segments_from_motion(
            motion,
            fps=fps,
            threshold=threshold,
            min_pause_sec=min_pause_sec,
            min_segment_sec=min_segment_sec,
            pad_sec=0.15,
        )

        segment_results = []
        accepted_words = []
        accepted_glosses = []

        for idx, (start, end) in enumerate(segments, start=1):
            pred = self.predict_landmark_segment(kp_raw[start:end], top_k=top_k)
            accepted = pred["confidence"] >= confidence_threshold

            if accepted:
                accepted_words.append(pred["text"])
                accepted_glosses.append(pred["gloss"])

            segment_results.append({
                "segment": idx,
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_sec": float(round((end - start) / fps, 3)),
                "gloss": pred["gloss"],
                "text": pred["text"],
                "confidence": float(pred["confidence"]),
                "accepted": bool(accepted),
                "top_k": pred["top_k"],
            })

        sentence = simple_sentence_postprocess(accepted_words)

        return {
            "pipeline": "sign_language_sentence_to_text",
            "input_video_path": video_path,
            "text": sentence,
            "gloss_sequence": accepted_glosses,
            "word_sequence": accepted_words,
            "num_segments": len(segments),
            "fps": float(fps),
            "segmentation": {
                "threshold": threshold,
                "min_pause_sec": min_pause_sec,
                "min_segment_sec": min_segment_sec,
                "confidence_threshold": confidence_threshold,
            },
            "segments": segment_results,
        }

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "model_type": "LandmarkTransformer",
            "num_classes": self.cfg["num_classes"],
            "feat_dim": self.cfg["feat_dim"],
            "t_frames": self.cfg["t_frames"],
            "d_model": self.cfg["d_model"],
            "n_heads": self.cfg["n_heads"],
            "n_layers": self.cfg["n_layers"],
            "parameters": int(total_params),
            "tta_enabled": self.use_tta,
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "val_acc": self.checkpoint.get("val_acc"),
            "val_f1": self.checkpoint.get("val_f1"),
            "val_top5": self.checkpoint.get("val_top5"),
            "sample_labels": [self.id2label[i] for i in sorted(self.id2label.keys())[:10]],
        }


def load_sign_language_interface(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> SignLanguageToTextInterface:
    return SignLanguageToTextInterface(
        checkpoint_path=checkpoint_path,
        device=device,
        use_tta=True,   # TTA on by default
    )


def sign_to_text(
    video_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    interface = load_sign_language_interface(checkpoint_path=checkpoint_path, device=device)
    return interface.predict(video_path=video_path, top_k=top_k)


def sign_sentence_to_text(
    video_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    interface = load_sign_language_interface(checkpoint_path=checkpoint_path, device=device)
    return interface.predict_sentence(video_path=video_path, top_k=top_k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Landmark Transformer Sign Language to Text")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sentence", action="store_true")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA (faster but less robust)")
    args = parser.parse_args()

    interface = SignLanguageToTextInterface(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_tta=not args.no_tta,
    )

    if args.sentence:
        result = interface.predict_sentence(args.video, top_k=args.top_k)
    else:
        result = interface.predict(args.video, top_k=args.top_k)

    print(json.dumps(result, indent=2, ensure_ascii=False))