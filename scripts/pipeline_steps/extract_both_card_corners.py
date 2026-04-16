from _path_setup import ensure_scripts_dir_on_path

ensure_scripts_dir_on_path()

import cv2

from card_extraction_helpers import (
    extract_card_from_image,
    extract_two_corners,
    get_bottom_right_corner_box,
    get_top_left_corner_box,
)
from io_helpers import load_image, save_images, show_images, wait_for_windows
from path_helpers import get_default_sample_image_path, get_processed_dir


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

    print(f"Found {len(extraction.contours)} contours.")
    print(f"Largest contour area: {extraction.contour_area}")
    print(f"Largest contour perimeter: {extraction.perimeter}")
    print(f"Approximated contour points: {len(extraction.approx)}")

    extracted_card = extraction.extracted_card
    top_left_corner, bottom_right_corner_rotated = extract_two_corners(extracted_card)
    bottom_right_x1, bottom_right_y1, bottom_right_x2, bottom_right_y2 = (
        get_bottom_right_corner_box(extracted_card)
    )
    bottom_right_corner = extracted_card[
        bottom_right_y1:bottom_right_y2,
        bottom_right_x1:bottom_right_x2,
    ]
    top_left_x1, top_left_y1, top_left_x2, top_left_y2 = get_top_left_corner_box(extracted_card)

    card_with_boxes = extracted_card.copy()

    cv2.rectangle(
        card_with_boxes,
        (top_left_x1, top_left_y1),
        (top_left_x2, top_left_y2),
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        card_with_boxes,
        (bottom_right_x1, bottom_right_y1),
        (bottom_right_x2, bottom_right_y2),
        (0, 0, 255),
        2,
    )

    save_images(
        output_dir,
        [
            ("18_extracted_card_for_two_corners.jpg", extracted_card),
            ("19_card_with_two_corner_boxes.jpg", card_with_boxes),
            ("20_top_left_corner.jpg", top_left_corner),
            ("21_bottom_right_corner.jpg", bottom_right_corner),
            ("22_bottom_right_corner_rotated.jpg", bottom_right_corner_rotated),
        ],
    )

    show_images(
        [
            ("Extracted Card", extracted_card),
            ("Card With Two Corner Boxes", card_with_boxes),
            ("Top Left Corner", top_left_corner),
            ("Bottom Right Corner", bottom_right_corner),
            ("Bottom Right Corner Rotated", bottom_right_corner_rotated),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
