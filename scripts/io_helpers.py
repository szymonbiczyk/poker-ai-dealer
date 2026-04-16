from __future__ import annotations

from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def load_image(path: Path, flags: int = cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flags)

    if image is None:
        print(f"Error: failed to load image: {path}")

    return image


def load_grayscale_image(path: Path):
    return load_image(path, cv2.IMREAD_GRAYSCALE)


def write_image(
    image_path: Path,
    image,
    *,
    params: list[int] | None = None,
    announce: bool = False,
    raise_on_failure: bool = False,
) -> bool:
    image_path.parent.mkdir(parents=True, exist_ok=True)

    if params is None:
        success = cv2.imwrite(str(image_path), image)
    else:
        success = cv2.imwrite(str(image_path), image, params)

    if announce:
        if success:
            print(f"Saved: {image_path}")
        else:
            print(f"Failed to save: {image_path}")

    if not success and raise_on_failure:
        raise ValueError(f"Failed to save image: {image_path}")

    return success


def save_image(output_dir: Path, filename: str, image) -> None:
    write_image(output_dir / filename, image, announce=True)


def save_images(output_dir: Path, named_images: list[tuple[str, object]]) -> None:
    for filename, image in named_images:
        save_image(output_dir, filename, image)


def save_jpg(image_path: Path, image, quality: int = 100) -> None:
    write_image(
        image_path,
        image,
        params=[int(cv2.IMWRITE_JPEG_QUALITY), quality],
        raise_on_failure=True,
    )


def show_images(images: list[tuple[str, object]]) -> None:
    for window_name, image in images:
        cv2.imshow(window_name, image)


def show_resizable_window(window_name: str, image, width: int = 900, height: int = 700) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.resizeWindow(window_name, width, height)


def wait_for_windows(message: str = "Press any key in an image window to close.") -> None:
    print(message)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
