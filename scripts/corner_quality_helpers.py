from __future__ import annotations

import cv2
import numpy as np


def analyze_corner(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
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


def total_score(metrics: dict, max_sharpness: float, max_contrast: float) -> float:
    sharpness_score = metrics["sharpness"] / max(max_sharpness, 1e-6)
    contrast_score = metrics["contrast"] / max(max_contrast, 1e-6)
    ratio_score = foreground_score(metrics["foreground_ratio"])

    return (
        0.50 * sharpness_score +
        0.30 * contrast_score +
        0.20 * ratio_score
    )


def compare_two_corners(
    first_image: np.ndarray,
    second_image: np.ndarray,
    *,
    first_name: str = "top_left",
    second_name: str = "bottom_right_rotated",
) -> dict:
    first_metrics = analyze_corner(first_image)
    second_metrics = analyze_corner(second_image)

    max_sharpness = max(first_metrics["sharpness"], second_metrics["sharpness"])
    max_contrast = max(first_metrics["contrast"], second_metrics["contrast"])

    first_score = total_score(first_metrics, max_sharpness, max_contrast)
    second_score = total_score(second_metrics, max_sharpness, max_contrast)

    if first_score > second_score:
        selected_name = first_name
        selected_image = first_image
        selected_score = first_score
    else:
        selected_name = second_name
        selected_image = second_image
        selected_score = second_score

    return {
        "metrics": {
            first_name: first_metrics,
            second_name: second_metrics,
        },
        "scores": {
            first_name: first_score,
            second_name: second_score,
        },
        "selected_name": selected_name,
        "selected_image": selected_image,
        "selected_score": selected_score,
    }
