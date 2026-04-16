from pathlib import Path

import cv2
import numpy as np

from io_helpers import is_image_file, load_grayscale_image


def create_symbol_mask(symbol_image: np.ndarray) -> np.ndarray:
    """Convert a cropped symbol into a clean foreground mask."""
    if symbol_image.ndim == 2:
        gray = symbol_image
    else:
        gray = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    foreground = cv2.findNonZero(threshold)

    if foreground is None:
        raise ValueError("No foreground pixels found in symbol image.")

    x, y, width, height = cv2.boundingRect(foreground)
    return threshold[y:y + height, x:x + width]


def normalize_symbol(
    symbol_image: np.ndarray,
    canvas_size: tuple[int, int],
    margin: int = 4,
    allow_upscale: bool = False,
) -> np.ndarray:
    """Center a symbol on a fixed canvas and only shrink it when needed."""
    canvas_height, canvas_width = canvas_size
    mask = create_symbol_mask(symbol_image)

    mask_height, mask_width = mask.shape[:2]
    max_width = max(1, canvas_width - 2 * margin)
    max_height = max(1, canvas_height - 2 * margin)

    scale = min(max_width / mask_width, max_height / mask_height)
    if not allow_upscale:
        scale = min(scale, 1.0)

    resized_width = max(1, int(round(mask_width * scale)))
    resized_height = max(1, int(round(mask_height * scale)))

    resized_mask = cv2.resize(
        mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    offset_x = (canvas_width - resized_width) // 2
    offset_y = (canvas_height - resized_height) // 2
    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized_mask

    return canvas


def match_symbol(symbol_image: np.ndarray, templates_dir: Path) -> dict:
    best_label = None
    best_score = -1.0
    best_template = None
    best_normalized_symbol = None
    best_normalized_template = None

    all_results = []

    for template_path in sorted(templates_dir.glob("*")):
        if not is_image_file(template_path):
            continue

        template = load_grayscale_image(template_path)

        if template is None:
            continue

        canvas_size = template.shape[:2]
        normalized_symbol = normalize_symbol(symbol_image, canvas_size)
        normalized_template = normalize_symbol(template, canvas_size)

        result = cv2.matchTemplate(
            normalized_symbol,
            normalized_template,
            cv2.TM_CCOEFF_NORMED,
        )

        score = float(result[0][0])
        label = template_path.stem

        all_results.append(
            (label, score, template, normalized_symbol, normalized_template)
        )
        print(f"Template match: {label} -> {score:.4f}")

        if score > best_score:
            best_score = score
            best_label = label
            best_template = template
            best_normalized_symbol = normalized_symbol
            best_normalized_template = normalized_template

    all_results.sort(key=lambda item: item[1], reverse=True)

    return {
        "best_label": best_label,
        "best_score": best_score,
        "best_template": best_template,
        "best_normalized_symbol": best_normalized_symbol,
        "best_normalized_template": best_normalized_template,
        "all_results": all_results,
    }
