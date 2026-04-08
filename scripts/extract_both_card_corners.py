from pathlib import Path

import cv2

from card_contour_helpers import preprocess_for_card_contour, approximate_card_contour
from extract_card import four_point_transform


def save_image(output_dir: Path, filename: str, image) -> None:
    output_path = output_dir / filename
    success = cv2.imwrite(str(output_path), image)

    if success:
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")


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

    card_height, card_width = extracted_card.shape[:2]

    corner_width_ratio = 0.20
    corner_height_ratio = 0.28

    corner_width = int(card_width * corner_width_ratio)
    corner_height = int(card_height * corner_height_ratio)

    top_left_corner = extracted_card[0:corner_height, 0:corner_width]
    bottom_right_corner = extracted_card[
        card_height - corner_height:card_height,
        card_width - corner_width:card_width
    ]

    # Rotate bottom-right corner so it has the same orientation as top-left.
    bottom_right_corner_rotated = cv2.rotate(bottom_right_corner, cv2.ROTATE_180)

    card_with_boxes = extracted_card.copy()

    cv2.rectangle(
        card_with_boxes,
        (0, 0),
        (corner_width, corner_height),
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        card_with_boxes,
        (card_width - corner_width, card_height - corner_height),
        (card_width, card_height),
        (0, 0, 255),
        2,
    )

    save_image(output_dir, "18_extracted_card_for_two_corners.jpg", extracted_card)
    save_image(output_dir, "19_card_with_two_corner_boxes.jpg", card_with_boxes)
    save_image(output_dir, "20_top_left_corner.jpg", top_left_corner)
    save_image(output_dir, "21_bottom_right_corner.jpg", bottom_right_corner)
    save_image(output_dir, "22_bottom_right_corner_rotated.jpg", bottom_right_corner_rotated)

    cv2.imshow("Extracted Card", extracted_card)
    cv2.imshow("Card With Two Corner Boxes", card_with_boxes)
    cv2.imshow("Top Left Corner", top_left_corner)
    cv2.imshow("Bottom Right Corner", bottom_right_corner)
    cv2.imshow("Bottom Right Corner Rotated", bottom_right_corner_rotated)

    print("Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()