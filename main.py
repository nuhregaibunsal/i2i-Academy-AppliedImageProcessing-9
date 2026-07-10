import argparse
import sys
import cv2
from pathlib import Path

from alpr import (
    load_image,
    ImageLoadError,
    detect_plate,
    DetectionConfig,
    extract_plate_text,
)


DEBUG_OUTPUT_DIR = Path("debug_output")


def _save_debug_image(image, filename: str) -> None:
    DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / filename), image)


def _run_debug_pipeline(image, config: DetectionConfig) -> None:
    from alpr.plate_detector import (
        to_grayscale,
        apply_gaussian_blur,
        detect_edges,
        find_plate_contour_candidates,
        isolate_plate_region,
    )

    grayscale  = to_grayscale(image)
    blurred    = apply_gaussian_blur(grayscale, config.gaussian_kernel_size, config.gaussian_sigma)
    edge_map   = detect_edges(blurred, config.canny_low_threshold, config.canny_high_threshold)
    candidates = find_plate_contour_candidates(edge_map, config.contour_candidate_count)
    plate      = isolate_plate_region(candidates, image)

    _save_debug_image(grayscale, "1_grayscale.jpg")
    _save_debug_image(blurred,   "2_gaussian_blur.jpg")
    _save_debug_image(edge_map,  "3_canny_edges.jpg")

    annotated = image.copy()
    cv2.drawContours(annotated, candidates, contourIdx=-1, color=(0, 255, 0), thickness=2)
    _save_debug_image(annotated, "4_contour_candidates.jpg")

    if plate is not None:
        x, y, w, h = plate.bounding_rect
        annotated_plate_box = image.copy()
        cv2.rectangle(annotated_plate_box, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        _save_debug_image(annotated_plate_box, "5_detected_plate_box.jpg")
        _save_debug_image(plate.cropped_image, "6_cropped_plate.jpg")

    print(f"  [debug] Intermediate images saved to: {DEBUG_OUTPUT_DIR.resolve()}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated License Plate Recognition (ALPR) — OpenCV + EasyOCR"
    )
    parser.add_argument(
        "--image",
        required=True,
        metavar="PATH",
        help="Path to the input car image (JPEG, PNG, BMP).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate pipeline images to ./debug_output/.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        metavar="CODES",
        help="Comma-separated EasyOCR language codes (default: en).",
    )
    return parser


def run(image_path: str, languages: tuple[str, ...], debug: bool) -> None:
    print(f"\n[1/4] Loading image: {image_path}")
    try:
        car_image = load_image(image_path)
    except ImageLoadError as error:
        print(f"  ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    config = DetectionConfig()

    if debug:
        print("[dbg] Running debug pipeline …")
        _run_debug_pipeline(car_image, config)

    print("[2/4] Running license plate detection pipeline …")
    plate_region = detect_plate(car_image, config)

    if plate_region is None:
        print(
            "  ERROR: No license plate region could be isolated.\n"
            "  Tip: Try a clearer, higher-contrast image or adjust DetectionConfig thresholds.",
            file=sys.stderr,
        )
        sys.exit(1)

    x, y, w, h = plate_region.bounding_rect
    print(f"  Plate region found at x={x}, y={y}, w={w}, h={h}")

    print("[3/4] Running OCR on the cropped plate region …")
    plate_text = extract_plate_text(plate_region.cropped_image, languages=languages)

    print("[4/4] Recognition complete.")
    print("\n" + "=" * 40)
    print(f"  LICENSE PLATE: {plate_text if plate_text else '[no text recognized]'}")
    print("=" * 40 + "\n")


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    language_codes = tuple(code.strip() for code in args.lang.split(","))

    run(
        image_path=args.image,
        languages=language_codes,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
