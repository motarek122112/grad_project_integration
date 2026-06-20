from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np


class BrowserVideoRecorder:
    """Thread-safe recorder used by streamlit-webrtc frame callbacks."""

    def __init__(self, fps: float = 20.0, max_seconds: int = 10) -> None:
        self.fps = fps
        self.max_frames = int(fps * max_seconds)
        self._lock = threading.Lock()
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._recording = False
        self._frame_count = 0

    def start(self) -> Path:
        with self._lock:
            self._release_writer()
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            handle.close()
            self._path = Path(handle.name)
            self._recording = True
            self._frame_count = 0
            return self._path

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            if not self._recording:
                return

            height, width = frame_bgr.shape[:2]

            if self._writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    str(self._path),
                    fourcc,
                    self.fps,
                    (width, height),
                )
                if not self._writer.isOpened():
                    self._recording = False
                    self._release_writer()
                    raise RuntimeError("OpenCV could not start the webcam video writer.")

            self._writer.write(frame_bgr)
            self._frame_count += 1

            if self._frame_count >= self.max_frames:
                self._recording = False
                self._release_writer()

    def stop(self) -> Path | None:
        with self._lock:
            self._recording = False
            self._release_writer()
            return self._path

    def reset(self) -> None:
        with self._lock:
            self._recording = False
            self._release_writer()
            self._path = None
            self._frame_count = 0

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def output_path(self) -> Path | None:
        with self._lock:
            return self._path

    def has_recording(self) -> bool:
        with self._lock:
            return (
                self._path is not None
                and self._path.exists()
                and self._path.stat().st_size > 0
                and self._frame_count > 0
            )

    def _release_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
