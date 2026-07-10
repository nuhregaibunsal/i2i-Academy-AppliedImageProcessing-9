# Applied Image Processing — Automated License Plate Recognition (ALPR)

A classical (non–deep-learning) computer-vision pipeline that locates a car's
license plate in a photograph using pure image-processing math, crops it, and
reads the characters with an OCR engine.

The project deliberately relies on **classical image processing** — grayscale
conversion, blurring, Canny edge detection and contour geometry — to find the
plate, and only uses a pretrained OCR model for the final character-reading
step.

## Demo

Running the pipeline on the sample image (`araba.jpg`):

```
[1/4] Loading image: araba.jpg
[2/4] Running license plate detection pipeline …
  Plate region found at x=216, y=290, w=103, h=23
[3/4] Running OCR on the cropped plate region …
[4/4] Recognition complete.

========================================
  LICENSE PLATE: 34FRK052
========================================
```

## How it works

The pipeline is split into small, single-responsibility functions inside the
`alpr` package:

| Stage | Function | What it does |
|-------|----------|--------------|
| 1. Load | `load_image` | Reads the image with OpenCV (Unicode-safe on Windows). |
| 2. Grayscale | `to_grayscale` | Collapses colour to a single intensity channel. |
| 3. Blur | `apply_gaussian_blur` | Gaussian blur to suppress noise before edge detection. |
| 4. Edges | `detect_edges` | Canny edge detection to expose object outlines. |
| 5. Contours | `find_plate_contour_candidates` | Finds and ranks the largest contours by area. |
| 6. Isolate | `isolate_plate_region` | Keeps the contour that is rectangular, has a plate-like aspect ratio, occupies a plate-sized area, and sits near the horizontal centre — then crops it from the original image. |
| 7. OCR | `extract_plate_text` | Upscales, contrast-enhances (CLAHE) and binarises (Otsu) the crop, then reads the text with EasyOCR. |

Pass `--debug` to save every intermediate stage (grayscale, blur, edges,
candidate contours, detected box and final crop) to `./debug_output/` so the
math at each step can be inspected.

## Theoretical background

**1. What is the difference between "Computer Vision" and classical "Image
Processing"?**
Image processing transforms an image into another image (e.g. blurring or edge
detection) without trying to understand its content, whereas computer vision
aims to *interpret* the image and extract high-level meaning such as "this
region is a license plate reading 34FRK052". In short, image processing is the
low-level preparation and computer vision is the higher-level understanding
built on top of it.

**2. Why is it almost always necessary to convert an image to grayscale before
applying edge detection?**
Edge detection looks for sharp changes in brightness (intensity gradients), and
a single-channel grayscale image expresses exactly that intensity, so the
algorithm has one clear signal to work on instead of three competing colour
channels — this makes detection simpler, faster and far less noisy.

**3. What does an OCR (Optical Character Recognition) engine do?**
An OCR engine detects text within an image and converts those pixel shapes into
machine-readable characters (a string), so that "34FRK052" printed on a plate
becomes the text `"34FRK052"` your program can use.

## Project structure

```
.
├── main.py                 # CLI entry point — orchestrates the 4-stage run
├── requirements.txt        # Python dependencies
├── araba.jpg               # Sample input image
└── alpr/                   # Core package
    ├── __init__.py         # Public API
    ├── image_loader.py     # Unicode-safe image loading
    ├── plate_detector.py   # Classical detection pipeline (stages 2–6)
    └── ocr_reader.py       # OCR pre-processing + EasyOCR reading (stage 7)
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`:
  - `opencv-python`
  - `easyocr`
  - `numpy`

## Installation

```bash
pip install -r requirements.txt
```

> On first run, EasyOCR downloads its pretrained recognition model (a few tens
> of MB). This happens once and is then cached locally.

## Usage

```bash
# Basic run
python main.py --image araba.jpg

# Save intermediate pipeline images to ./debug_output/
python main.py --image araba.jpg --debug

# Use additional OCR languages (comma-separated EasyOCR codes)
python main.py --image araba.jpg --lang en,tr
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--image PATH` | *(required)* | Path to the input car image (JPEG, PNG, BMP). |
| `--debug` | off | Save each intermediate stage to `./debug_output/`. |
| `--lang CODES` | `en` | Comma-separated EasyOCR language codes. |

## Notes on robustness

Real-world plates vary in lighting, angle and font. The detector uses a
combination of geometric constraints (rectangularity, aspect ratio, area
fraction and horizontal position) to reject distractors such as the grille or
car body, and the OCR stage applies upscaling, CLAHE contrast enhancement and
Otsu thresholding to give the recognition model the cleanest possible input.
Thresholds are centralised in `DetectionConfig` so they can be tuned per
dataset without touching the pipeline logic.
