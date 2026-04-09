from pathlib import Path

import cv2
import numpy as np

from card_contour_helpers import preprocess_for_card_contour, approximate_card_contour
from extract_card import four_point_transform
from detect_symbols_from_best_corner import (
    analyze_corner,
    total_score,
    extract_two_corners,
    detect_rank_and_suit,
)


def load_grayscale_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: failed to load image: {path}")
        return None

    return image


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
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / "data" / "samples" / "card_test.jpg"
    rank_templates_dir = project_root / "data" / "templates" / "ranks"
    suit_templates_dir = project_root / "data" / "templates" / "suits"

    if not image_path.exists():
        print("Error: image file does not exist.")
        return

    if not rank_templates_dir.exists() or not suit_templates_dir.exists():
        print("Error: template directories do not exist.")
        print("Create data/templates/ranks and data/templates/suits first.")
        return

    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: failed to load image.")
        return

    edges = preprocess_for_card_contour(image)
    contours, largest_contour, contour_area, perimeter, approx = approximate_card_contour(edges)

    if largest_contour is None or approx is None:
        print("Error: no valid card contour found.")
        return

    print(f"Found {len(contours)} contours.")
    print(f"Largest contour area: {contour_area}")
    print(f"Largest contour perimeter: {perimeter}")
    print(f"Approximated contour points: {len(approx)}")

    if len(approx) != 4:
        print("Error: contour approximation did not return 4 points.")
        return

    points = approx.reshape(4, 2).astype("float32")
    extracted_card = four_point_transform(image, points)

    top_left_corner, bottom_right_corner_rotated = extract_two_corners(extracted_card)

    top_left_metrics = analyze_corner(top_left_corner)
    bottom_right_metrics = analyze_corner(bottom_right_corner_rotated)

    max_sharpness = max(
        top_left_metrics["sharpness"],
        bottom_right_metrics["sharpness"],
    )
    max_contrast = max(
        top_left_metrics["contrast"],
        bottom_right_metrics["contrast"],
    )

    top_left_score = total_score(top_left_metrics, max_sharpness, max_contrast)
    bottom_right_score = total_score(bottom_right_metrics, max_sharpness, max_contrast)

    print(f"Top-left score: {top_left_score:.4f}")
    print(f"Bottom-right rotated score: {bottom_right_score:.4f}")

    if top_left_score > bottom_right_score:
        selected_corner_name = "top_left"
        selected_corner = top_left_corner
    else:
        selected_corner_name = "bottom_right_rotated"
        selected_corner = bottom_right_corner_rotated

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

    cv2.imshow("Selected Better Corner", selected_corner)
    cv2.imshow("Selected Corner Threshold", threshold)
    cv2.imshow("Selected Corner Symbol Boxes", boxed)
    cv2.imshow("Rank Region", rank_region)
    cv2.imshow("Suit Region", suit_region)

    print("Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()