import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DetectionConfig:
    gaussian_kernel_size: tuple[int, int] = (5, 5)
    gaussian_sigma: int = 0
    canny_low_threshold: int = 30
    canny_high_threshold: int = 200
    contour_candidate_count: int = 50
    plate_min_aspect_ratio: float = 2.0
    plate_max_aspect_ratio: float = 6.0
    plate_center_margin: float = 0.34
    # A real plate occupies only a small band of the frame; these bounds reject
    # both tiny edge noise and huge rectangles such as the whole car body/grille.
    plate_min_area_ratio: float = 0.002
    plate_max_area_ratio: float = 0.20


@dataclass(frozen=True)
class PlateRegion:
    cropped_image: np.ndarray
    bounding_rect: tuple[int, int, int, int]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(
    grayscale_image: np.ndarray,
    kernel_size: tuple[int, int],
    sigma: int,
) -> np.ndarray:
    return cv2.GaussianBlur(grayscale_image, kernel_size, sigma)


def detect_edges(
    blurred_image: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> np.ndarray:
    return cv2.Canny(blurred_image, low_threshold, high_threshold)


def find_plate_contour_candidates(
    edge_map: np.ndarray,
    candidate_count: int,
) -> list[np.ndarray]:
    all_contours, _ = cv2.findContours(
        edge_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return sorted(all_contours, key=cv2.contourArea, reverse=True)[:candidate_count]


def _is_quadrilateral(contour: np.ndarray, approximation_accuracy: float = 0.02) -> bool:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = approximation_accuracy * perimeter
    approximated_polygon = cv2.approxPolyDP(contour, epsilon, closed=True)
    return 4 <= len(approximated_polygon) <= 6


def _has_plate_aspect_ratio(
    width: int,
    height: int,
    min_ratio: float,
    max_ratio: float,
) -> bool:
    if height == 0:
        return False
    aspect_ratio = width / height
    return min_ratio <= aspect_ratio <= max_ratio


def _has_plate_area(
    width: int,
    height: int,
    image_area: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> bool:
    if image_area == 0:
        return False
    area_ratio = (width * height) / image_area
    return min_area_ratio <= area_ratio <= max_area_ratio


def _is_near_horizontal_center(
    contour_center_x: int,
    image_width: int,
    margin: float,
) -> bool:
    image_center_x = image_width / 2
    allowed_deviation = image_width * margin
    return abs(contour_center_x - image_center_x) <= allowed_deviation


def isolate_plate_region(
    candidates: list[np.ndarray],
    original_image: np.ndarray,
    config: DetectionConfig = DetectionConfig(),
) -> Optional[PlateRegion]:
    image_height, image_width = original_image.shape[:2]
    image_area = image_width * image_height
    for contour in candidates:
        if not _is_quadrilateral(contour):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if not _has_plate_aspect_ratio(
            width, height,
            config.plate_min_aspect_ratio,
            config.plate_max_aspect_ratio,
        ):
            continue
        if not _has_plate_area(
            width, height, image_area,
            config.plate_min_area_ratio,
            config.plate_max_area_ratio,
        ):
            continue
        contour_center_x = x + width // 2
        if not _is_near_horizontal_center(
            contour_center_x, image_width, config.plate_center_margin
        ):
            continue
        cropped_plate = original_image[y : y + height, x : x + width]
        return PlateRegion(
            cropped_image=cropped_plate,
            bounding_rect=(x, y, width, height),
        )
    return None


def detect_plate(
    image: np.ndarray,
    config: DetectionConfig = DetectionConfig(),
) -> Optional[PlateRegion]:
    grayscale  = to_grayscale(image)
    blurred    = apply_gaussian_blur(grayscale, config.gaussian_kernel_size, config.gaussian_sigma)
    edge_map   = detect_edges(blurred, config.canny_low_threshold, config.canny_high_threshold)
    candidates = find_plate_contour_candidates(edge_map, config.contour_candidate_count)
    plate      = isolate_plate_region(candidates, image, config)
    return plate
