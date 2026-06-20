from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILE_IMAGE_DIRECTORY = (
    PROJECT_ROOT / "data" / "profile_images"
)


def _extract_bbox(
    bbox: Any,
) -> tuple[int, int, int, int] | None:
    if isinstance(bbox, dict):
        required_keys = {"x1", "y1", "x2", "y2"}
        if required_keys.issubset(bbox):
            return (
                int(bbox["x1"]),
                int(bbox["y1"]),
                int(bbox["x2"]),
                int(bbox["y2"]),
            )

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return tuple(int(value) for value in bbox[:4])

    return None


def _expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Make the face crop looser so the profile picture looks more natural.
    """
    face_width = x2 - x1
    face_height = y2 - y1

    extra_left = int(face_width * 0.55)
    extra_right = int(face_width * 0.55)
    extra_top = int(face_height * 0.55)
    extra_bottom = int(face_height * 0.85)

    new_x1 = max(0, x1 - extra_left)
    new_y1 = max(0, y1 - extra_top)
    new_x2 = min(image_width, x2 + extra_right)
    new_y2 = min(image_height, y2 + extra_bottom)

    return new_x1, new_y1, new_x2, new_y2


def save_profile_picture(
    uploaded_image,
    *,
    user_id: int,
    bbox: Any = None,
) -> str:
    PROFILE_IMAGE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_bytes = uploaded_image.getvalue()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    normalized_bbox = _extract_bbox(bbox)

    if normalized_bbox is not None:
        x1, y1, x2, y2 = normalized_bbox

        x1 = max(0, min(x1, image.width - 1))
        y1 = max(0, min(y1, image.height - 1))
        x2 = max(x1 + 1, min(x2, image.width))
        y2 = max(y1 + 1, min(y2, image.height))

        x1, y1, x2, y2 = _expand_bbox(
            x1,
            y1,
            x2,
            y2,
            image.width,
            image.height,
        )

        image = image.crop((x1, y1, x2, y2))

    # Resize gently without over-zooming
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    output_path = (
        PROFILE_IMAGE_DIRECTORY / f"user_{user_id}.jpg"
    )

    image.save(
        output_path,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return str(output_path)