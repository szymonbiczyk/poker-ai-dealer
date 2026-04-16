from _path_setup import ensure_scripts_dir_on_path

ensure_scripts_dir_on_path()

import cv2

from corner_symbol_helpers import find_symbol_candidates, threshold_corner
from io_helpers import load_image, save_images, show_images, wait_for_windows
from path_helpers import get_processed_dir


def main() -> None:
    output_dir = get_processed_dir(create=True)
    corner_path = output_dir / "12_top_left_corner.jpg"

    if not corner_path.exists():
        print("Error: top-left corner image does not exist.")
        print("Run scripts/pipeline_steps/extract_card_corner.py first.")
        return

    corner = load_image(corner_path)

    if corner is None:
        return

    gray, threshold = threshold_corner(corner)
    contours, candidates = find_symbol_candidates(threshold, min_area=100)

    print(f"Found {len(contours)} contours before filtering.")

    print(f"Found {len(candidates)} contour candidates after filtering.")

    if len(candidates) < 2:
        print("Error: not enough valid contours found for rank and suit.")
        return

    # Sort top-to-bottom, then left-to-right
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

    save_images(
        output_dir,
        [
            ("13_corner_gray.jpg", gray),
            ("14_corner_threshold.jpg", threshold),
            ("15_corner_symbol_boxes.jpg", boxed),
            ("16_rank_region.jpg", rank_region),
            ("17_suit_region.jpg", suit_region),
        ],
    )

    show_images(
        [
            ("Corner Threshold", threshold),
            ("Corner Symbol Boxes", boxed),
            ("Rank Region", rank_region),
            ("Suit Region", suit_region),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
