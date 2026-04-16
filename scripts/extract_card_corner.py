import cv2

from card_extraction_helpers import (
    extract_card_from_image,
    extract_top_left_corner,
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
    top_left_corner = extract_top_left_corner(extracted_card)
    x1, y1, x2, y2 = get_top_left_corner_box(extracted_card)

    card_with_corner_box = extracted_card.copy()
    cv2.rectangle(
        card_with_corner_box,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2,
    )

    save_images(
        output_dir,
        [
            ("10_extracted_card.jpg", extracted_card),
            ("11_card_with_corner_box.jpg", card_with_corner_box),
            ("12_top_left_corner.jpg", top_left_corner),
        ],
    )

    show_images(
        [
            ("Extracted Card", extracted_card),
            ("Card With Corner Box", card_with_corner_box),
            ("Top Left Corner", top_left_corner),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
