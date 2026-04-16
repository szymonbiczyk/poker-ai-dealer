from _path_setup import ensure_scripts_dir_on_path

ensure_scripts_dir_on_path()

import cv2
import numpy as np

from io_helpers import (
    load_grayscale_image,
    show_resizable_window,
    wait_for_windows,
)
from path_helpers import get_processed_dir, get_rank_templates_dir, get_suit_templates_dir
from template_matching_helpers import match_symbol


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def pad_to_size(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    background_color=(255, 255, 255),
) -> np.ndarray:
    image = to_bgr(image)
    height, width = image.shape[:2]

    pad_bottom = max(0, target_height - height)
    pad_right = max(0, target_width - width)

    return cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=background_color,
    )


def add_label(
    image: np.ndarray,
    text: str,
    width: int = 220,
    height: int = 220,
    banner_height: int = 40,
) -> np.ndarray:
    image = to_bgr(image)
    image = pad_to_size(image, width, height)

    banner = np.full((banner_height, width, 3), 255, dtype=np.uint8)

    cv2.putText(
        banner,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([banner, image])


def stack_horizontally(images: list[np.ndarray]) -> np.ndarray:
    return np.hstack(images)


def stack_vertically(images: list[np.ndarray]) -> np.ndarray:
    return np.vstack(images)


def create_comparison_preview(
    title: str,
    original_symbol: np.ndarray,
    best_template: np.ndarray,
    normalized_symbol: np.ndarray,
    normalized_template: np.ndarray,
    best_label: str,
    best_score: float,
) -> np.ndarray:
    tile_width = 220
    tile_height = 220

    tile_1 = add_label(original_symbol, f"{title} symbol", tile_width, tile_height)
    tile_2 = add_label(best_template, f"Best: {best_label}", tile_width, tile_height)
    tile_3 = add_label(normalized_symbol, "Normalized symbol", tile_width, tile_height)
    tile_4 = add_label(normalized_template, f"Norm template {best_score:.4f}", tile_width, tile_height)

    top_row = stack_horizontally([tile_1, tile_2])
    bottom_row = stack_horizontally([tile_3, tile_4])

    return stack_vertically([top_row, bottom_row])


def print_top_results(title: str, results: list[tuple], top_k: int = 3) -> None:
    print(f"\nTop {top_k} {title}:")
    for label, score, *_ in results[:top_k]:
        print(f"  {label}: {score:.4f}")


def main() -> None:
    processed_dir = get_processed_dir()
    rank_image_path = processed_dir / "29_detect_best_corner_rank_region.jpg"
    suit_image_path = processed_dir / "30_detect_best_corner_suit_region.jpg"

    rank_templates_dir = get_rank_templates_dir()
    suit_templates_dir = get_suit_templates_dir()

    if not rank_image_path.exists() or not suit_image_path.exists():
        print("Error: rank or suit region image does not exist.")
        print("Run scripts/pipeline_steps/detect_symbols_from_best_corner.py first.")
        return

    if not rank_templates_dir.exists() or not suit_templates_dir.exists():
        print("Error: template directories do not exist.")
        print("Create data/templates/ranks and data/templates/suits first.")
        return

    rank_image = load_grayscale_image(rank_image_path)
    suit_image = load_grayscale_image(suit_image_path)

    if rank_image is None or suit_image is None:
        return

    print("\nMatching rank templates:")
    rank_match = match_symbol(rank_image, rank_templates_dir)

    print("\nMatching suit templates:")
    suit_match = match_symbol(suit_image, suit_templates_dir)

    matched_rank = rank_match["best_label"]
    rank_score = rank_match["best_score"]

    matched_suit = suit_match["best_label"]
    suit_score = suit_match["best_score"]

    if matched_rank is None or matched_suit is None:
        print("Error: failed to find valid template matches.")
        return

    print_top_results("rank matches", rank_match["all_results"], top_k=3)
    print_top_results("suit matches", suit_match["all_results"], top_k=3)

    print("\nFinal result:")
    print(f"Rank: {matched_rank} ({rank_score:.4f})")
    print(f"Suit: {matched_suit} ({suit_score:.4f})")
    print(f"Card: {matched_rank} of {matched_suit}")

    rank_preview = create_comparison_preview(
        title="Rank",
        original_symbol=rank_image,
        best_template=rank_match["best_template"],
        normalized_symbol=rank_match["best_normalized_symbol"],
        normalized_template=rank_match["best_normalized_template"],
        best_label=matched_rank,
        best_score=rank_score,
    )

    suit_preview = create_comparison_preview(
        title="Suit",
        original_symbol=suit_image,
        best_template=suit_match["best_template"],
        normalized_symbol=suit_match["best_normalized_symbol"],
        normalized_template=suit_match["best_normalized_template"],
        best_label=matched_suit,
        best_score=suit_score,
    )

    show_resizable_window("Rank Match Preview", rank_preview, 950, 800)
    show_resizable_window("Suit Match Preview", suit_preview, 950, 800)
    wait_for_windows()


if __name__ == "__main__":
    main()
