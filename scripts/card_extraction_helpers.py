from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from card_contour_helpers import approximate_card_contour, preprocess_for_card_contour


@dataclass
class CardExtractionResult:
    edges: np.ndarray
    contours: list[np.ndarray]
    largest_contour: np.ndarray
    contour_area: float
    perimeter: float
    approx: np.ndarray
    extracted_card: np.ndarray


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    # Image coordinates start in the top-left corner: (0, 0).
    # smallest x + y -> top-left
    # largest x + y -> bottom-right
    # smallest y - x -> top-right
    # largest y - x -> bottom-left
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

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
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def detect_card_contour(image: np.ndarray) -> tuple[
    np.ndarray,
    list[np.ndarray],
    np.ndarray | None,
    float | None,
    float | None,
    np.ndarray | None,
]:
    edges = preprocess_for_card_contour(image)
    contours, largest_contour, contour_area, perimeter, approx = approximate_card_contour(edges)
    return edges, contours, largest_contour, contour_area, perimeter, approx


def extract_card_from_image(image: np.ndarray) -> CardExtractionResult:
    edges, contours, largest_contour, contour_area, perimeter, approx = detect_card_contour(image)

    if largest_contour is None or approx is None:
        raise ValueError("no valid card contour found.")

    if len(approx) != 4:
        raise ValueError("contour approximation did not return 4 points.")

    points = approx.reshape(4, 2).astype("float32")
    extracted_card = four_point_transform(image, points)

    return CardExtractionResult(
        edges=edges,
        contours=contours,
        largest_contour=largest_contour,
        contour_area=contour_area,
        perimeter=perimeter,
        approx=approx,
        extracted_card=extracted_card,
    )


def get_corner_dimensions(
    extracted_card: np.ndarray,
    corner_width_ratio: float = 0.20,
    corner_height_ratio: float = 0.28,
) -> tuple[int, int]:
    card_height, card_width = extracted_card.shape[:2]
    corner_width = int(card_width * corner_width_ratio)
    corner_height = int(card_height * corner_height_ratio)
    return corner_width, corner_height


def get_top_left_corner_box(
    extracted_card: np.ndarray,
    corner_width_ratio: float = 0.20,
    corner_height_ratio: float = 0.28,
) -> tuple[int, int, int, int]:
    corner_width, corner_height = get_corner_dimensions(
        extracted_card,
        corner_width_ratio,
        corner_height_ratio,
    )
    return 0, 0, corner_width, corner_height


def get_bottom_right_corner_box(
    extracted_card: np.ndarray,
    corner_width_ratio: float = 0.20,
    corner_height_ratio: float = 0.28,
) -> tuple[int, int, int, int]:
    card_height, card_width = extracted_card.shape[:2]
    corner_width, corner_height = get_corner_dimensions(
        extracted_card,
        corner_width_ratio,
        corner_height_ratio,
    )
    return (
        card_width - corner_width,
        card_height - corner_height,
        card_width,
        card_height,
    )


def extract_top_left_corner(
    extracted_card: np.ndarray,
    corner_width_ratio: float = 0.20,
    corner_height_ratio: float = 0.28,
) -> np.ndarray:
    _, _, x2, y2 = get_top_left_corner_box(
        extracted_card,
        corner_width_ratio,
        corner_height_ratio,
    )
    return extracted_card[0:y2, 0:x2]


def extract_two_corners(
    extracted_card: np.ndarray,
    corner_width_ratio: float = 0.20,
    corner_height_ratio: float = 0.28,
) -> tuple[np.ndarray, np.ndarray]:
    top_left_corner = extract_top_left_corner(
        extracted_card,
        corner_width_ratio,
        corner_height_ratio,
    )

    x1, y1, x2, y2 = get_bottom_right_corner_box(
        extracted_card,
        corner_width_ratio,
        corner_height_ratio,
    )
    bottom_right_corner = extracted_card[y1:y2, x1:x2]
    bottom_right_corner_rotated = cv2.rotate(bottom_right_corner, cv2.ROTATE_180)

    return top_left_corner, bottom_right_corner_rotated
