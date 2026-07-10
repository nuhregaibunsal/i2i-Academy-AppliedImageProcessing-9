import cv2
import easyocr
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_reader(language_codes: tuple[str, ...]) -> easyocr.Reader:
    return easyocr.Reader(list(language_codes), gpu=False)


def _preprocess_plate_for_ocr(plate_image: np.ndarray) -> np.ndarray:
    # Upscale so EasyOCR has more pixels per character to work with
    scale_factor = 3
    upscaled = cv2.resize(
        plate_image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    grayscale = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # CLAHE equalises local contrast — lifts dark characters on bright plate backgrounds
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(grayscale)

    # Otsu's threshold converts the plate to a clean black-and-white binary image
    _, binary = cv2.threshold(
        contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def _apply_character_corrections(raw_text: str) -> str:
    # License plates mix digits and letters in known positions.
    # These substitutions fix the most frequent single-character OCR confusions.
    corrections = {
        "O": "0",
        "I": "1",
        "S": "5",
        "B": "8",
        "G": "6",
        "Z": "2",
    }
    tokens = raw_text.split()
    corrected_tokens = []
    for token in tokens:
        if token.isdigit():
            corrected_tokens.append(token)
            continue
        if token.isalpha():
            corrected_tokens.append(token)
            continue
        # Mixed token: apply digit corrections to character-like glyphs
        corrected = "".join(corrections.get(ch, ch) for ch in token)
        corrected_tokens.append(corrected)
    return " ".join(corrected_tokens)


def extract_plate_text(
    plate_image: np.ndarray,
    languages: tuple[str, ...] = ("en",),
    confidence_threshold: float = 0.2,
) -> str:
    preprocessed = _preprocess_plate_for_ocr(plate_image)

    reader = _get_reader(languages)
    detections = reader.readtext(preprocessed)

    accepted_segments = [
        text
        for (_bounding_box, text, confidence) in detections
        if confidence >= confidence_threshold
    ]

    raw_result = " ".join(accepted_segments).strip().upper()
    return _apply_character_corrections(raw_result)
