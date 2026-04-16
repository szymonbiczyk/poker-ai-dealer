from io_helpers import load_image as read_image, show_images, wait_for_windows
from path_helpers import get_default_sample_image_path


def main() -> None:
    image_path = get_default_sample_image_path()

    print(f"Looking for image at: {image_path}")

    if not image_path.exists():
        print("Error: image file does not exist.")
        return

    image = read_image(image_path)

    if image is None:
        return

    height, width, channels = image.shape

    print("Image loaded successfully.")
    print(f"Width: {width}px")
    print(f"Height: {height}px")
    print(f"Channels: {channels}")

    show_images([("Loaded Image", image)])
    wait_for_windows()


if __name__ == "__main__":
    main()
