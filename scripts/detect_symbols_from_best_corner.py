from pathlib import Path

import cv2
import numpy as np

from card_contour_helpers import preprocess_for_card_contour, approximate_card_contour
from extract_card import four_point_transform


def save_image(output_dir: Path, filename: str, image) -> None:
    output_path = output_dir / filename
    success = cv2.imwrite(str(output_path), image)

    if success:
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")


def analyze_corner(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    foreground_ratio = np.count_nonzero(threshold) / threshold.size

    return {
        "gray": gray,
        "threshold": threshold,
        "sharpness": float(sharpness),
        "contrast": float(contrast),
        "foreground_ratio": float(foreground_ratio),
    }


def foreground_score(ratio: float, target: float = 0.22, tolerance: float = 0.22) -> float:
    return max(0.0, 1.0 - abs(ratio - target) / tolerance)


def total_score(metrics: dict, max_sharpness: float, max_contrast: float) -> float:
    sharpness_score = metrics["sharpness"] / max(max_sharpness, 1e-6)
    contrast_score = metrics["contrast"] / max(max_contrast, 1e-6)
    ratio_score = foreground_score(metrics["foreground_ratio"])

    return (
        0.50 * sharpness_score +
        0.30 * contrast_score +
        0.20 * ratio_score
    )


def extract_two_corners(extracted_card: np.ndarray,
                        corner_width_ratio: float = 0.20,
                        corner_height_ratio: float = 0.28) -> tuple[np.ndarray, np.ndarray]:
    card_height, card_width = extracted_card.shape[:2]

    corner_width = int(card_width * corner_width_ratio)
    corner_height = int(card_height * corner_height_ratio)

    top_left_corner = extracted_card[0:corner_height, 0:corner_width]

    bottom_right_corner = extracted_card[
        card_height - corner_height:card_height,
        card_width - corner_width:card_width,
    ]

    bottom_right_corner_rotated = cv2.rotate(bottom_right_corner, cv2.ROTATE_180)

    return top_left_corner, bottom_right_corner_rotated


def detect_rank_and_suit(corner: np.ndarray, min_area: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        candidates.append((x, y, w, h, area))

    if len(candidates) < 2:
        raise ValueError("Not enough valid contours found for rank and suit.")

    candidates.sort(key=lambda item: (item[1], item[0]))

    rank_box = candidates[0]
    suit_box = candidates[1]

    rx, ry, rw, rh, _ = rank_box
    sx, sy, sw, sh, _ = suit_box

    rank_region = corner[ry:ry + rh, rx:rx + rw]
    suit_region = corner[sy:sy + sh, sx:sx + sw]

    boxed = corner.copy()
    cv2.rectangle(boxed, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.rectangle(boxed, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    return threshold, boxed, rank_region, suit_region


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / "data" / "samples" / "card_test.jpg"
    output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print("Error: image file does not exist.")
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
        selected_score = top_left_score
    else:
        selected_corner_name = "bottom_right_rotated"
        selected_corner = bottom_right_corner_rotated
        selected_score = bottom_right_score

    print(f"Selected better corner: {selected_corner_name} ({selected_score:.4f})")

    try:
        threshold, boxed, rank_region, suit_region = detect_rank_and_suit(selected_corner)
    except ValueError as error:
        print(f"Error: {error}")
        return

    save_image(output_dir, "23_detect_best_corner_extracted_card.jpg", extracted_card)
    save_image(output_dir, "24_detect_best_corner_top_left.jpg", top_left_corner)
    save_image(output_dir, "25_detect_best_corner_bottom_right_rotated.jpg", bottom_right_corner_rotated)
    save_image(output_dir, "26_detect_best_corner_selected.jpg", selected_corner)
    save_image(output_dir, "27_detect_best_corner_threshold.jpg", threshold)
    save_image(output_dir, "28_detect_best_corner_symbol_boxes.jpg", boxed)
    save_image(output_dir, "29_detect_best_corner_rank_region.jpg", rank_region)
    save_image(output_dir, "30_detect_best_corner_suit_region.jpg", suit_region)

    cv2.imshow("Extracted Card", extracted_card)
    cv2.imshow("Top Left Corner", top_left_corner)
    cv2.imshow("Bottom Right Corner Rotated", bottom_right_corner_rotated)
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