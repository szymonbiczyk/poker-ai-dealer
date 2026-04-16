from pathlib import Path

import cv2
import numpy as np

from card_extraction_helpers import extract_card_from_image, extract_two_corners
from corner_quality_helpers import compare_two_corners
from detect_symbols_from_best_corner import detect_rank_and_suit
from io_helpers import load_grayscale_image, load_image, show_images, wait_for_windows
from path_helpers import (
    get_default_sample_image_path,
    get_rank_templates_dir,
    get_suit_templates_dir,
)


def preprocess_symbol(image: np.ndarray) -> np.ndarray:
    _, threshold = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return threshold


def resize_to_template_size(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    template_height, template_width = template.shape[:2]
    resized = cv2.resize(image, (template_width, template_height))
    return resized


def match_symbol(symbol_image: np.ndarray, templates_dir: Path):
    best_label = None
    best_score = -1.0

    for template_path in sorted(templates_dir.glob("*")):
        if template_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        template = load_grayscale_image(template_path)

        if template is None:
            continue

        symbol_processed = preprocess_symbol(symbol_image)
        template_processed = preprocess_symbol(template)

        resized_symbol = resize_to_template_size(symbol_processed, template_processed)

        result = cv2.matchTemplate(
            resized_symbol,
            template_processed,
            cv2.TM_CCOEFF_NORMED,
        )

        score = result[0][0]
        label = template_path.stem

        print(f"Template match: {label} -> {score:.4f}")

        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score


def main() -> None:
    image_path = get_default_sample_image_path()
    rank_templates_dir = get_rank_templates_dir()
    suit_templates_dir = get_suit_templates_dir()

    if not image_path.exists():
        print("Error: image file does not exist.")
        return

    if not rank_templates_dir.exists() or not suit_templates_dir.exists():
        print("Error: template directories do not exist.")
        print("Create data/templates/ranks and data/templates/suits first.")
        return

    image = load_image(image_path)

    if image is None:
        return

    try:
        extraction = extract_card_from_image(image)
    except ValueError as error:
        print(f"Error: {error}")
        return

    print(f"Found {len(extraction.contours)} contours.")
    print(f"Largest contour area: {extraction.contour_area}")
    print(f"Largest contour perimeter: {extraction.perimeter}")
    print(f"Approximated contour points: {len(extraction.approx)}")

    extracted_card = extraction.extracted_card
    top_left_corner, bottom_right_corner_rotated = extract_two_corners(extracted_card)
    comparison = compare_two_corners(
        top_left_corner,
        bottom_right_corner_rotated,
        first_name="top_left",
        second_name="bottom_right_rotated",
    )
    top_left_score = comparison["scores"]["top_left"]
    bottom_right_score = comparison["scores"]["bottom_right_rotated"]

    print(f"Top-left score: {top_left_score:.4f}")
    print(f"Bottom-right rotated score: {bottom_right_score:.4f}")

    selected_corner_name = comparison["selected_name"]
    selected_corner = comparison["selected_image"]

    print(f"Selected better corner: {selected_corner_name}")

    try:
        threshold, boxed, rank_region, suit_region = detect_rank_and_suit(selected_corner)
    except ValueError as error:
        print(f"Error: {error}")
        return

    rank_gray = cv2.cvtColor(rank_region, cv2.COLOR_BGR2GRAY)
    suit_gray = cv2.cvtColor(suit_region, cv2.COLOR_BGR2GRAY)

    print("\nMatching rank templates:")
    matched_rank, rank_score = match_symbol(rank_gray, rank_templates_dir)

    print("\nMatching suit templates:")
    matched_suit, suit_score = match_symbol(suit_gray, suit_templates_dir)

    if matched_rank is None or matched_suit is None:
        print("Error: failed to match rank or suit.")
        return

    card_result = {
        "rank": matched_rank,
        "suit": matched_suit,
        "card": f"{matched_rank} of {matched_suit}",
        "rank_score": round(rank_score, 4),
        "suit_score": round(suit_score, 4),
        "selected_corner": selected_corner_name,
    }

    print("\nFinal result:")
    print(card_result)

    show_images(
        [
            ("Selected Better Corner", selected_corner),
            ("Selected Corner Threshold", threshold),
            ("Selected Corner Symbol Boxes", boxed),
            ("Rank Region", rank_region),
            ("Suit Region", suit_region),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
