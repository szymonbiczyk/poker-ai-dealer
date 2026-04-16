from corner_quality_helpers import compare_two_corners
from io_helpers import load_image, show_images, wait_for_windows
from path_helpers import get_processed_dir


def print_metrics(label: str, metrics: dict, score: float) -> None:
    print(f"\n{label}")
    print(f"  Sharpness:        {metrics['sharpness']:.2f}")
    print(f"  Contrast:         {metrics['contrast']:.2f}")
    print(f"  Foreground ratio: {metrics['foreground_ratio']:.4f}")
    print(f"  Total score:      {score:.4f}")


def main() -> None:
    output_dir = get_processed_dir()

    top_left_path = output_dir / "20_top_left_corner.jpg"
    bottom_right_rotated_path = output_dir / "22_bottom_right_corner_rotated.jpg"

    top_left = load_image(top_left_path)
    bottom_right_rotated = load_image(bottom_right_rotated_path)

    if top_left is None or bottom_right_rotated is None:
        print("Run scripts/extract_both_card_corners.py first.")
        return

    comparison = compare_two_corners(
        top_left,
        bottom_right_rotated,
        first_name="top_left",
        second_name="bottom_right_rotated",
    )
    top_left_metrics = comparison["metrics"]["top_left"]
    bottom_right_metrics = comparison["metrics"]["bottom_right_rotated"]
    top_left_total = comparison["scores"]["top_left"]
    bottom_right_total = comparison["scores"]["bottom_right_rotated"]

    print_metrics("Top-left corner", top_left_metrics, top_left_total)
    print_metrics("Bottom-right rotated corner", bottom_right_metrics, bottom_right_total)

    if comparison["selected_name"] == "top_left":
        winner_name = "top-left corner"
        winner_image = top_left
        winner_threshold = top_left_metrics["threshold"]
    else:
        winner_name = "bottom-right rotated corner"
        winner_image = bottom_right_rotated
        winner_threshold = bottom_right_metrics["threshold"]

    print(f"\nSelected better corner: {winner_name}")

    show_images(
        [
            ("Top-left corner", top_left),
            ("Bottom-right rotated corner", bottom_right_rotated),
            ("Top-left threshold", top_left_metrics["threshold"]),
            ("Bottom-right threshold", bottom_right_metrics["threshold"]),
            ("Selected better corner", winner_image),
            ("Selected better corner threshold", winner_threshold),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
