import cv2
import numpy as np
from pathlib import Path


class ImageLoadError(Exception):
    pass


def load_image(image_path: str | Path) -> np.ndarray:
    resolved_path = Path(image_path).resolve()

    if not resolved_path.exists():
        raise ImageLoadError(f"Image not found: {resolved_path}")

    # cv2.imread() fails silently on Windows paths containing non-ASCII characters
    # (e.g. Turkish Ü, Ş, Ç). Reading raw bytes first and decoding in-memory
    # bypasses the OS filename encoding issue entirely.
    raw_bytes = np.fromfile(str(resolved_path), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageLoadError(
            f"OpenCV could not decode the file. "
            f"Ensure it is a supported format (JPEG, PNG, BMP): {resolved_path}"
        )

    return image
