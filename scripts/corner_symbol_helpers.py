from __future__ import annotations

import cv2
import numpy as np


BoxCandidate = tuple[int, int, int, int, float]


def threshold_corner(corner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    return gray, threshold


def find_symbol_candidates(
    threshold: np.ndarray,
    min_area: int = 100,
) -> tuple[list[np.ndarray], list[BoxCandidate]]:
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        candidates.append((x, y, w, h, area))

    return contours, candidates
