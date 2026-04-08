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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    _, threshold = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    save_image(output_dir, "01_original.jpg", image)
    save_image(output_dir, "02_gray.jpg", gray)
    save_image(output_dir, "03_blur.jpg", blur)
    save_image(output_dir, "04_edges.jpg", edges)
    save_image(output_dir, "05_threshold.jpg", threshold)

    cv2.imshow("Original", image)
    cv2.imshow("Gray", gray)
    cv2.imshow("Blur", blur)
    cv2.imshow("Edges", edges)
    cv2.imshow("Threshold", threshold)

    print("Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()