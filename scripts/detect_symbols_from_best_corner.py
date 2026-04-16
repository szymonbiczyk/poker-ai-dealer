import cv2
import numpy as np

from card_extraction_helpers import extract_card_from_image, extract_two_corners
from corner_quality_helpers import compare_two_corners
from corner_symbol_helpers import BoxCandidate, find_symbol_candidates, threshold_corner
from io_helpers import load_image, save_images, show_images, wait_for_windows
from path_helpers import get_default_sample_image_path, get_processed_dir


DEBUG_IMAGE_FILENAMES = {
    "all_candidates": "31_best_corner_all_candidates.jpg",
    "rank_boxes": "32_best_corner_rank_boxes.jpg",
    "remaining_boxes": "33_best_corner_remaining_boxes.jpg",
    "suit_candidates": "34_best_corner_suit_candidates.jpg",
}


def merge_boxes(boxes: list[BoxCandidate]) -> tuple[int, int, int, int]:
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[0] + box[2] for box in boxes)
    max_y = max(box[1] + box[3] for box in boxes)
    return min_x, min_y, max_x - min_x, max_y - min_y

def box_center(box: BoxCandidate) -> tuple[float, float]:
    x, y, w, h, _ = box
    return x + w / 2.0, y + h / 2.0


def vertical_overlap_ratio(
    first_box: BoxCandidate,
    second_box: BoxCandidate,
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
    boxes: list[BoxCandidate],
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


def build_symbol_debug_images(
    corner: np.ndarray,
    debug_info: dict,
) -> list[tuple[str, np.ndarray]]:
    return [
        (
            DEBUG_IMAGE_FILENAMES["all_candidates"],
            draw_labeled_boxes(corner, debug_info["sorted_candidates"], (255, 0, 0), "C"),
        ),
        (
            DEBUG_IMAGE_FILENAMES["rank_boxes"],
            draw_labeled_boxes(corner, debug_info["rank_boxes"], (0, 255, 0), "R"),
        ),
        (
            DEBUG_IMAGE_FILENAMES["remaining_boxes"],
            draw_labeled_boxes(corner, debug_info["remaining_boxes"], (0, 255, 255), "M"),
        ),
        (
            DEBUG_IMAGE_FILENAMES["suit_candidates"],
            draw_labeled_boxes(corner, debug_info["suit_candidates"], (0, 0, 255), "S"),
        ),
    ]


def print_box_group(label: str, prefix: str, boxes: list[BoxCandidate]) -> None:
    print(f"\n{label}:")
    for idx, box in enumerate(boxes):
        print(f"{prefix}{idx}: {box}")


def print_card_extraction_summary(extraction) -> None:
    print(f"Found {len(extraction.contours)} contours.")
    print(f"Largest contour area: {extraction.contour_area}")
    print(f"Largest contour perimeter: {extraction.perimeter}")
    print(f"Approximated contour points: {len(extraction.approx)}")


def save_corner_selection_debug_images(
    output_dir,
    extracted_card: np.ndarray,
    top_left_corner: np.ndarray,
    bottom_right_corner_rotated: np.ndarray,
    selected_corner: np.ndarray,
) -> None:
    save_images(
        output_dir,
        [
            ("23_detect_best_corner_extracted_card.jpg", extracted_card),
            ("24_detect_best_corner_top_left.jpg", top_left_corner),
            ("25_detect_best_corner_bottom_right_rotated.jpg", bottom_right_corner_rotated),
            ("26_detect_best_corner_selected.jpg", selected_corner),
        ],
    )


def save_symbol_regions_debug_images(
    output_dir,
    threshold: np.ndarray,
    boxed: np.ndarray,
    rank_region: np.ndarray,
    suit_region: np.ndarray,
) -> None:
    save_images(
        output_dir,
        [
            ("27_detect_best_corner_threshold.jpg", threshold),
            ("28_detect_best_corner_symbol_boxes.jpg", boxed),
            ("29_detect_best_corner_rank_region.jpg", rank_region),
            ("30_detect_best_corner_suit_region.jpg", suit_region),
        ],
    )


def split_rank_and_suit_boxes(
    candidates: list[BoxCandidate],
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int] | None,
    dict,
]:
    sorted_candidates = sorted(candidates, key=lambda item: (item[1], item[0]))

    def rank_anchor_score(box: BoxCandidate) -> float:
        x, y, _, _, _ = box
        return y + 0.35 * x

    rank_anchor = min(
        sorted_candidates,
        key=lambda box: (rank_anchor_score(box), box[0], box[1]),
    )
    ax, ay, aw, ah, _ = rank_anchor
    anchor_center_x, anchor_center_y = box_center(rank_anchor)
    anchor_right = ax + aw

    def looks_like_second_rank_box(box: BoxCandidate) -> bool:
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
        def second_rank_score(box: BoxCandidate) -> float:
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

    def suit_score(box: BoxCandidate) -> float:
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
    _, threshold = threshold_corner(corner)
    _, candidates = find_symbol_candidates(threshold, min_area=min_area)

    if len(candidates) < 2:
        raise ValueError("Not enough valid contours found for rank and suit.")

    rank_box, suit_box, debug_info = split_rank_and_suit_boxes(candidates)

    debug_output_dir = get_processed_dir(create=True)
    save_images(debug_output_dir, build_symbol_debug_images(corner, debug_info))

    print_box_group("Sorted candidates", "C", debug_info["sorted_candidates"])
    print_box_group("Rank boxes", "R", debug_info["rank_boxes"])
    print_box_group("Remaining boxes", "M", debug_info["remaining_boxes"])
    print_box_group("Suit candidates", "S", debug_info["suit_candidates"])

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
    image_path = get_default_sample_image_path()
    output_dir = get_processed_dir(create=True)

    if not image_path.exists():
        print("Error: image file does not exist.")
        return

    image = load_image(image_path)

    if image is None:
        return

    try:
        extraction = extract_card_from_image(image)
    except ValueError as error:
        print(f"Error: {error}")
        return

    print_card_extraction_summary(extraction)

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
    selected_score = comparison["selected_score"]

    print(f"Selected better corner: {selected_corner_name} ({selected_score:.4f})")

    save_corner_selection_debug_images(
        output_dir,
        extracted_card,
        top_left_corner,
        bottom_right_corner_rotated,
        selected_corner,
    )

    try:
        threshold, boxed, rank_region, suit_region = detect_rank_and_suit(selected_corner)
    except ValueError as error:
        print(f"Error: {error}")
        return

    save_symbol_regions_debug_images(
        output_dir,
        threshold,
        boxed,
        rank_region,
        suit_region,
    )

    show_images(
        [
            ("Extracted Card", extracted_card),
            ("Top Left Corner", top_left_corner),
            ("Bottom Right Corner Rotated", bottom_right_corner_rotated),
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
