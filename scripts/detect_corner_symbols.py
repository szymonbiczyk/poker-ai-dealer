from pathlib import Path

import cv2


def save_image(output_dir: Path, filename: str, image) -> None:
    output_path = output_dir / filename
    success = cv2.imwrite(str(output_path), image)

    if success:
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    corner_path = project_root / "data" / "processed" / "12_top_left_corner.jpg"
    output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not corner_path.exists():
        print("Error: top-left corner image does not exist.")
        print("Run scripts/extract_card_corner.py first.")
        return

    corner = cv2.imread(str(corner_path))

    if corner is None:
        print("Error: failed to load top-left corner image.")
        return

    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Found {len(contours)} contours before filtering.")

    min_area = 100
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        candidates.append((x, y, w, h, area))

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

    save_image(output_dir, "13_corner_gray.jpg", gray)
    save_image(output_dir, "14_corner_threshold.jpg", threshold)
    save_image(output_dir, "15_corner_symbol_boxes.jpg", boxed)
    save_image(output_dir, "16_rank_region.jpg", rank_region)
    save_image(output_dir, "17_suit_region.jpg", suit_region)

    cv2.imshow("Corner Threshold", threshold)
    cv2.imshow("Corner Symbol Boxes", boxed)
    cv2.imshow("Rank Region", rank_region)
    cv2.imshow("Suit Region", suit_region)

    print("Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()