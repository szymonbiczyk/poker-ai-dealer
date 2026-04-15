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


def merge_boxes(boxes: list[tuple[int, int, int, int, float]]) -> tuple[int, int, int, int]:
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[0] + box[2] for box in boxes)
    max_y = max(box[1] + box[3] for box in boxes)
    return min_x, min_y, max_x - min_x, max_y - min_y

def box_center(box: tuple[int, int, int, int, float]) -> tuple[float, float]:
    x, y, w, h, _ = box
    return x + w / 2.0, y + h / 2.0


def vertical_overlap_ratio(
    first_box: tuple[int, int, int, int, float],
    second_box: tuple[int, int, int, int, float],
) -> float:
    _, y1, _, h1, _ = first_box
    _, y2, _, h2, _ = second_box

    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)
    overlap = max(0, bottom - top)

    smaller_height = max(1, min(h1, h2))
    return overlap / smaller_height


def draw_labeled_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int, float]],
    color: tuple[int, int, int],
    prefix: str = "",
) -> np.ndarray:
    debug = image.copy()

    for idx, (x, y, w, h, area) in enumerate(boxes):
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug,
            f"{prefix}{idx}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return debug

def split_rank_and_suit_boxes(
    candidates: list[tuple[int, int, int, int, float]],
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int] | None,
    dict,
]:
    sorted_candidates = sorted(candidates, key=lambda item: (item[1], item[0]))

    def rank_anchor_score(box: tuple[int, int, int, int, float]) -> float:
        x, y, _, _, _ = box
        return y + 0.35 * x

    rank_anchor = min(
        sorted_candidates,
        key=lambda box: (rank_anchor_score(box), box[0], box[1]),
    )
    ax, ay, aw, ah, _ = rank_anchor
    anchor_center_x, anchor_center_y = box_center(rank_anchor)
    anchor_right = ax + aw

    def looks_like_second_rank_box(box: tuple[int, int, int, int, float]) -> bool:
        x, y, w, h, _ = box
        cx, cy = box_center(box)

        x_gap = x - anchor_right
        max_left_overlap = max(4, int(min(aw, w) * 0.25))
        max_x_gap = max(14, int(max(aw, w) * 0.80))
        center_y_delta = abs(cy - anchor_center_y)
        max_center_y_delta = max(10, int(max(ah, h) * 0.30))
        height_ratio = h / max(ah, 1)
        overlap = vertical_overlap_ratio(rank_anchor, box)

        same_rank_row = overlap >= 0.45 and center_y_delta <= max_center_y_delta
        reasonable_height = 0.55 <= height_ratio <= 1.60
        close_in_x = -max_left_overlap <= x_gap <= max_x_gap
        to_the_right = cx >= anchor_center_x

        return same_rank_row and reasonable_height and close_in_x and to_the_right

    second_rank_candidates = [
        candidate
        for candidate in sorted_candidates
        if candidate != rank_anchor and looks_like_second_rank_box(candidate)
    ]

    rank_boxes = [rank_anchor]

    if second_rank_candidates:
        def second_rank_score(box: tuple[int, int, int, int, float]) -> float:
            x, _, _, h, _ = box
            _, cy = box_center(box)

            x_gap = max(0.0, x - anchor_right)
            center_y_delta = abs(cy - anchor_center_y)
            height_delta = abs(h - ah)

            return x_gap + 0.75 * center_y_delta + 0.25 * height_delta

        rank_boxes.append(min(second_rank_candidates, key=second_rank_score))

    rank_boxes = sorted(rank_boxes, key=lambda item: item[0])
    remaining_boxes = [
        candidate for candidate in sorted_candidates if candidate not in rank_boxes
    ]

    rank_box = merge_boxes(rank_boxes)
    rx, ry, rw, rh = rank_box
    rank_bottom = ry + rh
    rank_center_x = rx + rw / 2.0
    rank_right = rx + rw

    suit_candidates = []

    # Suit should be below the rank and still near the same left-side column.
    max_suit_left = rank_right + max(10, int(rw * 0.25))

    for candidate in remaining_boxes:
        x, y, w, h, _ = candidate
        cx, cy = box_center(candidate)

        below_rank = y >= ry + int(rh * 0.55)
        near_rank_column = x <= max_suit_left

        if below_rank and near_rank_column:
            suit_candidates.append(candidate)

    debug_info = {
        "sorted_candidates": sorted_candidates,
        "rank_boxes": rank_boxes,
        "remaining_boxes": remaining_boxes,
        "suit_candidates": suit_candidates,
        "best_suit_box_raw": None,
    }

    if not suit_candidates:
        return rank_box, None, debug_info

    def suit_score(box: tuple[int, int, int, int, float]) -> float:
        x, y, w, h, _ = box
        cx, cy = box_center(box)

        vertical_distance = max(0.0, cy - rank_bottom)
        horizontal_distance = abs(cx - rank_center_x)

        return vertical_distance + 0.5 * horizontal_distance

    best_suit_box = min(suit_candidates, key=suit_score)
    debug_info["best_suit_box_raw"] = best_suit_box

    suit_box = merge_boxes([best_suit_box])

    return rank_box, suit_box, debug_info

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

    rank_box, suit_box, debug_info = split_rank_and_suit_boxes(candidates)

    all_candidates_debug = draw_labeled_boxes(
        corner,
        debug_info["sorted_candidates"],
        (255, 0, 0),
        "C",
    )

    rank_boxes_debug = draw_labeled_boxes(
        corner,
        debug_info["rank_boxes"],
        (0, 255, 0),
        "R",
    )

    remaining_boxes_debug = draw_labeled_boxes(
        corner,
        debug_info["remaining_boxes"],
        (0, 255, 255),
        "M",
    )

    suit_candidates_debug = draw_labeled_boxes(
        corner,
        debug_info["suit_candidates"],
        (0, 0, 255),
        "S",
    )

    project_root = Path(__file__).resolve().parent.parent
    debug_output_dir = project_root / "data" / "processed"
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(debug_output_dir / "31_debug_all_candidates.jpg"), all_candidates_debug)
    cv2.imwrite(str(debug_output_dir / "32_debug_rank_boxes.jpg"), rank_boxes_debug)
    cv2.imwrite(str(debug_output_dir / "33_debug_remaining_boxes.jpg"), remaining_boxes_debug)
    cv2.imwrite(str(debug_output_dir / "34_debug_suit_candidates.jpg"), suit_candidates_debug)

    print("\nSorted candidates:")
    for idx, box in enumerate(debug_info["sorted_candidates"]):
        print(f"C{idx}: {box}")

    print("\nRank boxes:")
    for idx, box in enumerate(debug_info["rank_boxes"]):
        print(f"R{idx}: {box}")

    print("\nRemaining boxes:")
    for idx, box in enumerate(debug_info["remaining_boxes"]):
        print(f"M{idx}: {box}")

    print("\nSuit candidates:")
    for idx, box in enumerate(debug_info["suit_candidates"]):
        print(f"S{idx}: {box}")

    if suit_box is None:
        raise ValueError("Could not separate rank and suit boxes from the corner.")

    rx, ry, rw, rh = rank_box
    sx, sy, sw, sh = suit_box

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

    save_image(output_dir, "23_detect_best_corner_extracted_card.jpg", extracted_card)
    save_image(output_dir, "24_detect_best_corner_top_left.jpg", top_left_corner)
    save_image(output_dir, "25_detect_best_corner_bottom_right_rotated.jpg", bottom_right_corner_rotated)
    save_image(output_dir, "26_detect_best_corner_selected.jpg", selected_corner)

    try:
        threshold, boxed, rank_region, suit_region = detect_rank_and_suit(selected_corner)
    except ValueError as error:
        print(f"Error: {error}")
        return

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
