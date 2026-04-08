from pathlib import Path

import cv2
import numpy as np


def save_image(output_dir: Path, filename: str, image) -> None:
    output_path = output_dir / filename
    success = cv2.imwrite(str(output_path), image)

    if success:
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    # Image coordinates start in the top-left corner: (0, 0).
    # smallest x + y -> top-left
    # largest x + y -> bottom-right
    # smallest y - x -> top-right
    # largest y - x -> bottom-left
    rect[0] = points[np.argmin(s)]      # top-left
    rect[2] = points[np.argmax(s)]      # bottom-right
    rect[1] = points[np.argmin(diff)]   # top-right
    rect[3] = points[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = int(max(height_right, height_left))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No contours found.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    print(f"Approximated contour points: {len(approx)}")

    if len(approx) != 4:
        print("Error: contour approximation did not return 4 points.")
        return

    points = approx.reshape(4, 2).astype("float32")
    warped = four_point_transform(image, points)

    contour_image = image.copy()
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 3)
    cv2.drawContours(contour_image, [approx], -1, (0, 0, 255), 3)

    save_image(output_dir, "08_card_contour.jpg", contour_image)
    save_image(output_dir, "09_extracted_card.jpg", warped)

    cv2.imshow("Detected Card Contour", contour_image)
    cv2.imshow("Extracted Card", warped)

    print("Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()