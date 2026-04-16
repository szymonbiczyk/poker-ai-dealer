import cv2

from card_extraction_helpers import detect_card_contour as detect_card_contour_data
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

    print("Image loaded successfully.")
    print(f"Original shape: {image.shape}")

    edges, contours, largest_contour, contour_area, perimeter, approx = (
        detect_card_contour_data(image)
    )

    print(f"Found {len(contours)} contours.")

    if largest_contour is None:
        print("No contours found.")
        return

    print(f"Largest contour area: {contour_area}")
    print(f"Largest contour perimeter: {perimeter}")

    print(f"Approximated contour points: {len(approx)}")

    # draw results
    contour_image = image.copy()
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 3)
    cv2.drawContours(contour_image, [approx], -1, (0, 0, 255), 3)

    save_images(
        output_dir,
        [
            ("06_edges_for_contours.jpg", edges),
            ("07_detected_card_contour.jpg", contour_image),
        ],
    )

    show_images(
        [
            ("Edges", edges),
            ("Detected Card Contour", contour_image),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
