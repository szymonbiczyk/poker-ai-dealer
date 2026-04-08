from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        print(f"Error: failed to load image: {path}")
        return None
    return image


def analyze_corner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    foreground_ratio = np.count_nonzero(threshold) / threshold.size

    return {
        "gray": gray,
        "threshold": threshold,
        "sharpness": float(sharpness),
        "contrast": float(contrast),
        "foreground_ratio": float(foreground_ratio),
    }


def foreground_score(ratio: float, target: float = 0.22, tolerance: float = 0.22) -> float:
    return max(0.0, 1.0 - abs(ratio - target) / tolerance)


def total_score(metrics, max_sharpness: float, max_contrast: float) -> float:
    sharpness_score = metrics["sharpness"] / max(max_sharpness, 1e-6)
    contrast_score = metrics["contrast"] / max(max_contrast, 1e-6)
    ratio_score = foreground_score(metrics["foreground_ratio"])

    score = (
        0.50 * sharpness_score +
        0.30 * contrast_score +
        0.20 * ratio_score
    )
    return score


def print_metrics(label: str, metrics: dict, score: float) -> None:
    print(f"\n{label}")
    print(f"  Sharpness:        {metrics['sharpness']:.2f}")
    print(f"  Contrast:         {metrics['contrast']:.2f}")
    print(f"  Foreground ratio: {metrics['foreground_ratio']:.4f}")
    print(f"  Total score:      {score:.4f}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "processed"

    top_left_path = output_dir / "20_top_left_corner.jpg"
    bottom_right_rotated_path = output_dir / "22_bottom_right_corner_rotated.jpg"

    top_left = load_image(top_left_path)
    bottom_right_rotated = load_image(bottom_right_rotated_path)

    if top_left is None or bottom_right_rotated is None:
        print("Run scripts/extract_both_card_corners.py first.")
        return

    top_left_metrics = analyze_corner(top_left)
    bottom_right_metrics = analyze_corner(bottom_right_rotated)

    max_sharpness = max(
        top_left_metrics["sharpness"],
        bottom_right_metrics["sharpness"],
    )
    max_contrast = max(
        top_left_metrics["contrast"],
        bottom_right_metrics["contrast"],
    )

    top_left_total = total_score(top_left_metrics, max_sharpness, max_contrast)
    bottom_right_total = total_score(bottom_right_metrics, max_sharpness, max_contrast)

    print_metrics("Top-left corner", top_left_metrics, top_left_total)
    print_metrics("Bottom-right rotated corner", bottom_right_metrics, bottom_right_total)

    if top_left_total > bottom_right_total:
        winner_name = "top-left corner"
        winner_image = top_left
        winner_threshold = top_left_metrics["threshold"]
    else:
        winner_name = "bottom-right rotated corner"
        winner_image = bottom_right_rotated
        winner_threshold = bottom_right_metrics["threshold"]

    print(f"\nSelected better corner: {winner_name}")

    cv2.imshow("Top-left corner", top_left)
    cv2.imshow("Bottom-right rotated corner", bottom_right_rotated)
    cv2.imshow("Top-left threshold", top_left_metrics["threshold"])
    cv2.imshow("Bottom-right threshold", bottom_right_metrics["threshold"])
    cv2.imshow("Selected better corner", winner_image)
    cv2.imshow("Selected better corner threshold", winner_threshold)

    print("Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()