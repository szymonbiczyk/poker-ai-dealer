from _path_setup import ensure_scripts_dir_on_path

ensure_scripts_dir_on_path()

import cv2

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    _, threshold = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    save_images(
        output_dir,
        [
            ("01_original.jpg", image),
            ("02_gray.jpg", gray),
            ("03_blur.jpg", blur),
            ("04_edges.jpg", edges),
            ("05_threshold.jpg", threshold),
        ],
    )

    show_images(
        [
            ("Original", image),
            ("Gray", gray),
            ("Blur", blur),
            ("Edges", edges),
            ("Threshold", threshold),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
