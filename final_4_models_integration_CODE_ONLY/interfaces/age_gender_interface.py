"""
Age + Gender Prediction Interface
---------------------------------
Integration-ready interface for Seif Ahmed's final age/gender models.

Pipeline:
    image input
    -> YOLOv8 face detection (optional but recommended)
    -> crop largest face
    -> EfficientNet-B4 age model
    -> custom CNN gender model
    -> JSON-ready result

Expected model files:
    models/yolov8n-face-lindevs.pt                 # or .pt.zip
    models/best_age_efficientnet_b4_finetuned.pth  # or .pth.zip
    models/best_gender_utkface.pth                 # or .pth.zip

Required packages:
    torch, torchvision, timm, ultralytics, opencv-python, pillow, numpy
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Utility helpers
# ============================================================


def _ensure_exists(path: Union[str, os.PathLike], label: str = "file") -> str:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _torch_load(path: Union[str, os.PathLike]) -> Any:
    """Load a PyTorch checkpoint from .pth/.pt or the same file renamed with .zip."""
    path = _ensure_exists(path, "Checkpoint")
    return torch.load(path, map_location="cpu")


def _copy_zip_named_weight_to_temp_pt(path: str) -> str:
    """
    Ultralytics sometimes expects a .pt suffix.
    If the uploaded YOLO file is named .pt.zip but is actually a torch/ultralytics weight,
    copy it to a temporary .pt path and load that.
    """
    suffixes = Path(path).suffixes
    if suffixes[-2:] == [".pt", ".zip"] or str(path).endswith(".pt.zip"):
        tmp_dir = tempfile.mkdtemp(prefix="yolo_face_")
        tmp_path = os.path.join(tmp_dir, Path(path).name.replace(".pt.zip", ".pt"))
        shutil.copy2(path, tmp_path)
        return tmp_path
    return path


def _to_builtin(obj: Any) -> Any:
    """Convert numpy / torch scalars to normal Python types for JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


def age_to_group(age: float) -> str:
    if age < 13:
        return "child"
    if age < 20:
        return "teen"
    if age < 35:
        return "young_adult"
    if age < 60:
        return "adult"
    return "senior"


# ============================================================
# Gender model
# ============================================================


class GenderCNN(nn.Module):
    """
    Custom CNN matching best_gender_utkface.pth.

    Input:
        RGB face image, resized to 128x128.
    Output:
        one logit. With UTKFace convention, sigmoid(logit) is usually P(female).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # features.0
            nn.BatchNorm2d(32),                           # features.1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # features.4
            nn.BatchNorm2d(64),                           # features.5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # features.8
            nn.BatchNorm2d(128),                          # features.9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),# features.12
            nn.BatchNorm2d(128),                          # features.13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # classifier.0
            nn.Linear(128 * 8 * 8, 256),                  # classifier.1
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),                           # classifier.4
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),                             # classifier.7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


# ============================================================
# Age model
# ============================================================


class AgeEfficientNetB4(nn.Module):
    """
    EfficientNet-B4 multi-task model matching best_age_efficientnet_b4_finetuned.pth.

    It returns:
        age_raw: numeric regression output
        bin_logits: logits for 7 age-bin classes

    Note:
        Many age models are trained on normalized age values. The interface below uses
        age_scale='auto' by default: if raw output looks normalized, it multiplies by 100.
    """

    def __init__(self, age_bins: int = 7) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "timm is required for the EfficientNet-B4 age model. Install it with: pip install timm"
            ) from exc

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.shared = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.age_head = nn.Linear(256, 1)
        self.bin_head = nn.Linear(256, age_bins)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        shared = self.shared(features)
        age_raw = self.age_head(shared).squeeze(1)
        bin_logits = self.bin_head(shared)
        return age_raw, bin_logits


# ============================================================
# YOLO face detector
# ============================================================


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def as_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class YOLOFaceDetector:
    """YOLOv8 face detector wrapper."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None, conf_threshold: float = 0.25):
        self.original_checkpoint_path = _ensure_exists(checkpoint_path, "YOLO face checkpoint")
        self.checkpoint_path = _copy_zip_named_weight_to_temp_pt(self.original_checkpoint_path)
        self.device = device or DEFAULT_DEVICE
        self.conf_threshold = conf_threshold

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO face detection. Install it with: pip install ultralytics"
            ) from exc

        self.model = YOLO(self.checkpoint_path)

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        h, w = image_bgr.shape[:2]
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            verbose=False,
            device=0 if str(self.device).startswith("cuda") else "cpu",
        )

        faces: List[FaceBox] = []
        if not results:
            return faces

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return faces

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

        for box, conf in zip(xyxy, confs):
            x1, y1, x2, y2 = box[:4]
            x1 = int(max(0, min(w - 1, round(x1))))
            y1 = int(max(0, min(h - 1, round(y1))))
            x2 = int(max(0, min(w, round(x2))))
            y2 = int(max(0, min(h, round(y2))))
            if x2 > x1 and y2 > y1:
                faces.append(FaceBox(x1, y1, x2, y2, float(conf)))

        return faces


def crop_face_from_image(
    image_bgr: np.ndarray,
    face_box: FaceBox,
    margin: float = 0.20,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = face_box.x1, face_box.y1, face_box.x2, face_box.y2
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    xx1 = max(0, x1 - pad_x)
    yy1 = max(0, y1 - pad_y)
    xx2 = min(w, x2 + pad_x)
    yy2 = min(h, y2 + pad_y)
    return image_bgr[yy1:yy2, xx1:xx2]


# ============================================================
# Main combined interface
# ============================================================


class AgeGenderInterface:
    """
    Combined face -> age + gender interface.

    Main usage:
        interface = AgeGenderInterface(
            age_checkpoint_path="models/best_age_efficientnet_b4_finetuned.pth.zip",
            gender_checkpoint_path="models/best_gender_utkface.pth.zip",
            face_detector_path="models/yolov8n-face-lindevs.pt.zip",
        )
        result = interface.predict("samples/person.jpg")
    """

    def __init__(
        self,
        age_checkpoint_path: str,
        gender_checkpoint_path: str,
        face_detector_path: Optional[str] = None,
        device: Optional[str] = None,
        use_face_detection: bool = True,
        face_conf_threshold: float = 0.25,
        face_margin: float = 0.20,
        age_img_size: int = 380,
        gender_img_size: int = 128,
        age_output_scale: Union[str, float] = "auto",
        positive_gender_label: str = "female",
        negative_gender_label: str = "male",
        gender_threshold: float = 0.5,
        age_bin_labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.age_checkpoint_path = _ensure_exists(age_checkpoint_path, "Age checkpoint")
        self.gender_checkpoint_path = _ensure_exists(gender_checkpoint_path, "Gender checkpoint")
        self.face_detector_path = face_detector_path
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.use_face_detection = use_face_detection
        self.face_conf_threshold = face_conf_threshold
        self.face_margin = face_margin
        self.age_img_size = age_img_size
        self.gender_img_size = gender_img_size
        self.age_output_scale = age_output_scale
        self.positive_gender_label = positive_gender_label
        self.negative_gender_label = negative_gender_label
        self.gender_threshold = gender_threshold
        self.age_bin_labels = list(age_bin_labels or ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60+"])

        self.face_detector: Optional[YOLOFaceDetector] = None
        if self.use_face_detection and face_detector_path:
            self.face_detector = YOLOFaceDetector(
                checkpoint_path=face_detector_path,
                device=str(self.device),
                conf_threshold=face_conf_threshold,
            )

        self.age_model = AgeEfficientNetB4(age_bins=7)
        age_state = _torch_load(self.age_checkpoint_path)
        if isinstance(age_state, dict) and "model_state" in age_state:
            age_state = age_state["model_state"]
        self.age_model.load_state_dict(age_state, strict=True)
        self.age_model.to(self.device)
        self.age_model.eval()

        self.gender_model = GenderCNN()
        gender_state = _torch_load(self.gender_checkpoint_path)
        if isinstance(gender_state, dict) and "model_state" in gender_state:
            gender_state = gender_state["model_state"]
        self.gender_model.load_state_dict(gender_state, strict=True)
        self.gender_model.to(self.device)
        self.gender_model.eval()

        self.age_transform = transforms.Compose([
            transforms.Resize((self.age_img_size, self.age_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.gender_transform = transforms.Compose([
            transforms.Resize((self.gender_img_size, self.gender_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _select_face(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.face_detector is None:
            return image_bgr, {
                "face_detected": False,
                "face_detection_used": False,
                "bbox": None,
                "detection_confidence": None,
                "note": "Face detector not used; full image was passed to age/gender models.",
            }

        faces = self.face_detector.detect(image_bgr)
        if not faces:
            return image_bgr, {
                "face_detected": False,
                "face_detection_used": True,
                "bbox": None,
                "detection_confidence": None,
                "note": "No face detected; full image was used as fallback.",
            }

        # For integration/demo: choose the largest detected face.
        selected = max(faces, key=lambda f: f.area)
        face_crop = crop_face_from_image(image_bgr, selected, margin=self.face_margin)
        return face_crop, {
            "face_detected": True,
            "face_detection_used": True,
            "bbox": selected.as_list(),
            "detection_confidence": round(selected.confidence, 4),
            "num_faces_detected": len(faces),
        }

    @staticmethod
    def _bgr_to_pil_rgb(image_bgr: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _scale_age(self, raw_age: float) -> float:
        if self.age_output_scale == "auto":
            # If output looks normalized, convert to years.
            age = raw_age * 100.0 if -0.5 <= raw_age <= 1.5 else raw_age
        else:
            age = raw_age * float(self.age_output_scale)
        return float(np.clip(age, 0.0, 100.0))

    @torch.no_grad()
    def predict(self, image_path: str, save_cropped_face_path: Optional[str] = None) -> Dict[str, Any]:
        image_path = _ensure_exists(image_path, "Image")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        face_bgr, face_info = self._select_face(image_bgr)

        if save_cropped_face_path:
            os.makedirs(os.path.dirname(save_cropped_face_path) or ".", exist_ok=True)
            cv2.imwrite(save_cropped_face_path, face_bgr)

        face_pil = self._bgr_to_pil_rgb(face_bgr)

        age_x = self.age_transform(face_pil).unsqueeze(0).to(self.device)
        gender_x = self.gender_transform(face_pil).unsqueeze(0).to(self.device)

        age_raw_tensor, age_bin_logits = self.age_model(age_x)
        age_raw = float(age_raw_tensor.squeeze().detach().cpu().item())
        predicted_age = self._scale_age(age_raw)

        age_bin_probs = torch.softmax(age_bin_logits, dim=1).squeeze(0).detach().cpu().numpy()
        age_bin_index = int(np.argmax(age_bin_probs))
        age_bin_label = self.age_bin_labels[age_bin_index] if age_bin_index < len(self.age_bin_labels) else str(age_bin_index)

        gender_logit = self.gender_model(gender_x)
        positive_prob = float(torch.sigmoid(gender_logit).squeeze().detach().cpu().item())
        gender = self.positive_gender_label if positive_prob >= self.gender_threshold else self.negative_gender_label
        gender_confidence = max(positive_prob, 1.0 - positive_prob)

        result = {
            "image_path": image_path,
            **face_info,
            "age": round(predicted_age, 2),
            "age_raw_output": round(age_raw, 6),
            "age_group": age_to_group(predicted_age),
            "age_bin_index": age_bin_index,
            "age_bin_label": age_bin_label,
            "age_bin_probabilities": {label: round(float(prob), 4) for label, prob in zip(self.age_bin_labels, age_bin_probs)},
            "gender": gender,
            "gender_confidence": round(gender_confidence, 4),
            "gender_positive_label": self.positive_gender_label,
            "gender_probability_positive": round(positive_prob, 4),
            "gender_probability_female": round(positive_prob, 4) if self.positive_gender_label.lower() == "female" else None,
        }
        return _to_builtin(result)

    def annotate_image(self, image_path: str, result: Dict[str, Any], output_path: str) -> str:
        image_path = _ensure_exists(image_path, "Image")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        label = f"{result.get('gender', 'unknown')} | age {result.get('age', '?')}"
        bbox = result.get("bbox")
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            y_text = max(20, y1 - 10)
            cv2.putText(image, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, image)
        return output_path

    def predict_and_save(
        self,
        image_path: str,
        json_output_path: Optional[str] = None,
        annotated_output_path: Optional[str] = None,
        cropped_face_output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.predict(image_path, save_cropped_face_path=cropped_face_output_path)

        if annotated_output_path:
            self.annotate_image(image_path, result, annotated_output_path)
            result["annotated_image_path"] = annotated_output_path

        if json_output_path:
            os.makedirs(os.path.dirname(json_output_path) or ".", exist_ok=True)
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(_to_builtin(result), f, indent=2, ensure_ascii=False)
            result["json_output_path"] = json_output_path

        return _to_builtin(result)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "interface": "AgeGenderInterface",
            "device": str(self.device),
            "age_model": "EfficientNet-B4 multitask age regression + age bins",
            "gender_model": "Custom CNN binary gender classifier",
            "face_detector": "YOLOv8 face detector" if self.face_detector else None,
            "age_checkpoint_path": self.age_checkpoint_path,
            "gender_checkpoint_path": self.gender_checkpoint_path,
            "face_detector_path": self.face_detector_path,
            "age_img_size": self.age_img_size,
            "gender_img_size": self.gender_img_size,
            "age_output_scale": self.age_output_scale,
            "gender_mapping": {
                "sigmoid_output >= threshold": self.positive_gender_label,
                "sigmoid_output < threshold": self.negative_gender_label,
                "threshold": self.gender_threshold,
            },
        }


# ============================================================
# Convenience functions
# ============================================================


def load_age_gender_interface(
    age_checkpoint_path: str,
    gender_checkpoint_path: str,
    face_detector_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> AgeGenderInterface:
    return AgeGenderInterface(
        age_checkpoint_path=age_checkpoint_path,
        gender_checkpoint_path=gender_checkpoint_path,
        face_detector_path=face_detector_path,
        device=device,
        **kwargs,
    )


def predict_age_gender(
    image_path: str,
    age_checkpoint_path: str,
    gender_checkpoint_path: str,
    face_detector_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    interface = load_age_gender_interface(
        age_checkpoint_path=age_checkpoint_path,
        gender_checkpoint_path=gender_checkpoint_path,
        face_detector_path=face_detector_path,
        device=device,
        **kwargs,
    )
    return interface.predict(image_path)


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Age + Gender inference interface")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--age_checkpoint", type=str, required=True, help="Age .pth/.zip checkpoint path")
    parser.add_argument("--gender_checkpoint", type=str, required=True, help="Gender .pth/.zip checkpoint path")
    parser.add_argument("--face_detector", type=str, default=None, help="YOLO face .pt/.zip checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--no_face_detection", action="store_true", help="Use full image instead of YOLO face crop")
    parser.add_argument("--json_output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--annotated_output", type=str, default=None, help="Optional annotated image output path")
    parser.add_argument("--cropped_face_output", type=str, default=None, help="Optional cropped face image output path")
    parser.add_argument("--age_scale", type=str, default="auto", help="auto or numeric scale, e.g. 100")
    parser.add_argument("--invert_gender", action="store_true", help="Swap male/female labels if needed")
    args = parser.parse_args()

    age_scale: Union[str, float]
    try:
        age_scale = float(args.age_scale)
    except ValueError:
        age_scale = args.age_scale

    positive_label = "male" if args.invert_gender else "female"
    negative_label = "female" if args.invert_gender else "male"

    interface = load_age_gender_interface(
        age_checkpoint_path=args.age_checkpoint,
        gender_checkpoint_path=args.gender_checkpoint,
        face_detector_path=args.face_detector,
        device=args.device,
        use_face_detection=not args.no_face_detection,
        age_output_scale=age_scale,
        positive_gender_label=positive_label,
        negative_gender_label=negative_label,
    )

    result = interface.predict_and_save(
        image_path=args.image,
        json_output_path=args.json_output,
        annotated_output_path=args.annotated_output,
        cropped_face_output_path=args.cropped_face_output,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
