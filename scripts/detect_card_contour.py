from pathlib import Path

import cv2

try:
    from card_contour_helpers import (
        approximate_card_contour,
        preprocess_for_card_contour,
    )
except ModuleNotFoundError:
    from scripts.card_contour_helpers import (
        approximate_card_contour,
        preprocess_for_card_contour,
    )


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

    print(f"Looking for image at: {image_path}")

    if not image_path.exists():
        print("Error: image file does not exist.")
        return

    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: failed to load image.")
        return

    print("Image loaded successfully.")
    print(f"Original shape: {image.shape}")

    edges = preprocess_for_card_contour(image)
    contours, largest_contour, contour_area, perimeter, approx = (
        approximate_card_contour(edges)
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

    save_image(output_dir, "06_edges_for_contours.jpg", edges)
    save_image(output_dir, "07_detected_card_contour.jpg", contour_image)

    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Card Contour", contour_image)

    print("Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
