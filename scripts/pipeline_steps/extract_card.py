from _path_setup import ensure_scripts_dir_on_path

ensure_scripts_dir_on_path()

import cv2

from card_extraction_helpers import extract_card_from_image
from io_helpers import load_image, save_images, show_images, wait_for_windows
from path_helpers import get_default_sample_image_path, get_processed_dir


def main() -> None:
    image_path = get_default_sample_image_path()
    output_dir = get_processed_dir(create=True)

    print(f"Looking for image at: {image_path}")

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

    print(f"Approximated contour points: {len(extraction.approx)}")

    contour_image = image.copy()
    cv2.drawContours(contour_image, [extraction.largest_contour], -1, (0, 255, 0), 3)
    cv2.drawContours(contour_image, [extraction.approx], -1, (0, 0, 255), 3)

    save_images(
        output_dir,
        [
            ("08_card_contour.jpg", contour_image),
            ("09_extracted_card.jpg", extraction.extracted_card),
        ],
    )

    show_images(
        [
            ("Detected Card Contour", contour_image),
            ("Extracted Card", extraction.extracted_card),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
