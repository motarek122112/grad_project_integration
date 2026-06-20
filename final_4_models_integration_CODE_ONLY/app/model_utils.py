from __future__ import annotations

import tempfile
from pathlib import Path


def save_uploaded_file(
    uploaded_file,
    suffix: str | None = None,
) -> Path:
    original_name = (
        getattr(uploaded_file, "name", "")
        or ""
    )

    original_suffix = Path(original_name).suffix

    if suffix:
        file_suffix = (
            suffix
            if suffix.startswith(".")
            else f".{suffix}"
        )
    elif original_suffix:
        file_suffix = original_suffix
    else:
        file_suffix = ".bin"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_suffix,
    ) as temporary_file:
        temporary_file.write(
            uploaded_file.getbuffer()
        )

        return Path(temporary_file.name)


def read_binary_result(result) -> bytes:
    if isinstance(result, bytes):
        return result

    output_path = Path(result)

    if not output_path.exists():
        raise FileNotFoundError(
            f"Model output was not found: {output_path}"
        )

    return output_path.read_bytes()