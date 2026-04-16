from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detect_symbols_from_best_corner import detect_rank_and_suit
from io_helpers import is_image_file, save_jpg
from path_helpers import (
    get_kaggle_cards_dir,
    get_processed_dir,
    get_rank_templates_dir,
    get_suit_templates_dir,
)


RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = {
    # The Kaggle files in this repository use non-standard suit letters.
    # Example: C2.png is visually a hearts card, not a clubs card.
    "C": "hearts",
    "D": "clubs",
    "H": "diamonds",
    "S": "spades",
}

# Kaggle face cards place artwork close to the corner, so this crop is a bit
# narrower than the generic card-corner script.
CORNER_WIDTH_RATIO = 0.14
CORNER_HEIGHT_RATIO = 0.28


@dataclass
class SymbolExample:
    card_name: str
    image: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rank and suit templates from Kaggle full-card images."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing template files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images to data/processed/template_debug.",
    )
    return parser.parse_args()


def parse_card_label(image_path: Path) -> tuple[str, str]:
    """Read the suit and rank directly from the file name."""
    label = image_path.stem.upper()

    if len(label) < 2:
        raise ValueError("Filename is too short to contain a card label.")

    suit_code = label[0]
    rank = label[1:]

    if suit_code not in SUITS:
        raise ValueError(f"Unknown suit code: {suit_code}")

    if rank not in RANKS:
        raise ValueError(f"Unknown rank: {rank}")

    return rank, SUITS[suit_code]


def load_card_image(image_path: Path) -> np.ndarray:
    """Load a PNG card image and place transparent pixels on a white background."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white_background = 255.0 * (1.0 - alpha)
        image = (bgr * alpha + white_background).astype(np.uint8)

    return image


def extract_top_left_corner(card_image: np.ndarray) -> np.ndarray:
    """Crop the same top-left corner region used by the existing corner scripts."""
    card_height, card_width = card_image.shape[:2]
    corner_width = int(card_width * CORNER_WIDTH_RATIO)
    corner_height = int(card_height * CORNER_HEIGHT_RATIO)
    return card_image[0:corner_height, 0:corner_width]

def save_debug_image(debug_dir: Path | None, folder_name: str, image_name: str, image: np.ndarray) -> None:
    if debug_dir is None:
        return

    save_jpg(debug_dir / folder_name / image_name, image)


def load_existing_canvas_size(template_dir: Path) -> tuple[int, int] | None:
    """Reuse the current template size if the folder already contains templates."""
    sizes = []

    for template_path in sorted(template_dir.glob("*")):
        if not is_image_file(template_path):
            continue

        image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        sizes.append((image.shape[0], image.shape[1]))

    if not sizes:
        return None

    max_height = max(size[0] for size in sizes)
    max_width = max(size[1] for size in sizes)
    return max_height, max_width


def estimate_canvas_size(images: list[np.ndarray], padding: int = 8) -> tuple[int, int]:
    """Pick a shared canvas size that fits the biggest segmented symbol plus padding."""
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    return max_height + padding, max_width + padding


def choose_canvas_size(template_dir: Path, images: list[np.ndarray]) -> tuple[int, int]:
    existing_size = load_existing_canvas_size(template_dir)

    if existing_size is not None:
        return existing_size

    return estimate_canvas_size(images)


def create_symbol_mask(symbol_image: np.ndarray) -> np.ndarray:
    """Convert a cropped symbol into a clean foreground mask."""
    if symbol_image.ndim == 2:
        gray = symbol_image
    else:
        gray = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    foreground = cv2.findNonZero(threshold)

    if foreground is None:
        raise ValueError("No foreground pixels found in symbol image.")

    x, y, width, height = cv2.boundingRect(foreground)
    return threshold[y:y + height, x:x + width]


def normalize_symbol(
    symbol_image: np.ndarray,
    canvas_size: tuple[int, int],
    margin: int = 4,
    allow_upscale: bool = False,
) -> np.ndarray:
    """Center a symbol on a fixed canvas and only shrink it when needed."""
    canvas_height, canvas_width = canvas_size
    mask = create_symbol_mask(symbol_image)

    mask_height, mask_width = mask.shape[:2]
    max_width = max(1, canvas_width - 2 * margin)
    max_height = max(1, canvas_height - 2 * margin)

    scale = min(max_width / mask_width, max_height / mask_height)
    if not allow_upscale:
        scale = min(scale, 1.0)

    resized_width = max(1, int(round(mask_width * scale)))
    resized_height = max(1, int(round(mask_height * scale)))

    resized_mask = cv2.resize(
        mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    offset_x = (canvas_width - resized_width) // 2
    offset_y = (canvas_height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized_mask

    return canvas


def mask_to_template_image(mask: np.ndarray) -> np.ndarray:
    return 255 - mask


def compare_masks(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    result = cv2.matchTemplate(first_mask, second_mask, cv2.TM_CCOEFF_NORMED)
    return float(result[0][0])


def score_example_quality(symbol_image: np.ndarray) -> float:
    """Prefer symbols with strong contrast and sharp edges."""
    if symbol_image.ndim == 2:
        gray = symbol_image
    else:
        gray = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    return float(sharpness + contrast)


def choose_best_example(
    examples: list[SymbolExample],
    canvas_size: tuple[int, int],
) -> tuple[SymbolExample, np.ndarray]:
    """Pick the single real sample that is both clean and most representative."""
    normalized_masks = [
        normalize_symbol(example.image, canvas_size)
        for example in examples
    ]

    quality_scores = [score_example_quality(example.image) for example in examples]
    max_quality = max(quality_scores, default=1.0)

    best_index = 0
    best_score = float("-inf")

    for index, mask in enumerate(normalized_masks):
        if len(normalized_masks) == 1:
            consistency_score = 1.0
        else:
            comparisons = []

            for other_index, other_mask in enumerate(normalized_masks):
                if index == other_index:
                    continue

                comparisons.append(compare_masks(mask, other_mask))

            consistency_score = sum(comparisons) / len(comparisons)

        quality_score = quality_scores[index] / max(max_quality, 1e-6)
        total_score = 0.85 * consistency_score + 0.15 * quality_score

        if total_score > best_score:
            best_score = total_score
            best_index = index

    return examples[best_index], normalized_masks[best_index]


def build_examples(
    input_dir: Path,
    debug_dir: Path | None,
) -> tuple[dict[str, list[SymbolExample]], dict[str, list[SymbolExample]], list[str]]:
    rank_examples = {rank: [] for rank in RANKS}
    suit_examples = {suit_name: [] for suit_name in SUITS.values()}
    failures = []

    for image_path in sorted(input_dir.glob("*.png")):
        try:
            rank, suit_name = parse_card_label(image_path)
            card_image = load_card_image(image_path)
            corner = extract_top_left_corner(card_image)
            threshold, boxed, rank_region, suit_region = detect_rank_and_suit(corner)
        except Exception as error:
            failures.append(f"{image_path.name}: {error}")
            continue

        rank_examples[rank].append(SymbolExample(card_name=image_path.stem, image=rank_region))
        suit_examples[suit_name].append(SymbolExample(card_name=image_path.stem, image=suit_region))

        save_debug_image(debug_dir, "corners", f"{image_path.stem}.jpg", corner)
        save_debug_image(debug_dir, "thresholds", f"{image_path.stem}.jpg", threshold)
        save_debug_image(debug_dir, "boxes", f"{image_path.stem}.jpg", boxed)
        save_debug_image(debug_dir, "rank_regions", f"{image_path.stem}.jpg", rank_region)
        save_debug_image(debug_dir, "suit_regions", f"{image_path.stem}.jpg", suit_region)

    return rank_examples, suit_examples, failures


def collect_all_images(examples_by_label: dict[str, list[SymbolExample]]) -> list[np.ndarray]:
    images = []

    for examples in examples_by_label.values():
        for example in examples:
            images.append(example.image)

    return images


def build_templates(
    examples_by_label: dict[str, list[SymbolExample]],
    output_dir: Path,
    canvas_size: tuple[int, int],
    force: bool,
    debug_dir: Path | None,
    debug_folder_name: str,
) -> None:
    for label, examples in examples_by_label.items():
        if not examples:
            print(f"Missing examples for {label}.")
            continue

        output_path = output_dir / f"{label}.jpg"

        if output_path.exists() and not force:
            print(f"Skipping existing template: {output_path.name}")
            continue

        for example in examples:
            candidate_mask = normalize_symbol(example.image, canvas_size)
            candidate_image = mask_to_template_image(candidate_mask)
            save_debug_image(
                debug_dir,
                debug_folder_name,
                f"{label}_{example.card_name}.jpg",
                candidate_image,
            )

        best_example, best_mask = choose_best_example(examples, canvas_size)
        final_template = mask_to_template_image(best_mask)

        save_jpg(output_path, final_template)
        print(f"Saved template: {output_path} from {best_example.card_name}")


def main() -> None:
    args = parse_args()

    input_dir = get_kaggle_cards_dir()
    rank_output_dir = get_rank_templates_dir()
    suit_output_dir = get_suit_templates_dir()
    debug_dir = get_processed_dir(create=args.debug) / "template_debug" if args.debug else None

    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        return

    rank_output_dir.mkdir(parents=True, exist_ok=True)
    suit_output_dir.mkdir(parents=True, exist_ok=True)

    rank_examples, suit_examples, failures = build_examples(input_dir, debug_dir)

    if failures:
        print("Skipped files:")
        for failure in failures:
            print(f"  - {failure}")

    all_rank_images = collect_all_images(rank_examples)
    all_suit_images = collect_all_images(suit_examples)

    if not all_rank_images or not all_suit_images:
        print("Error: could not collect enough symbols to build templates.")
        return

    rank_canvas_size = choose_canvas_size(rank_output_dir, all_rank_images)
    suit_canvas_size = choose_canvas_size(suit_output_dir, all_suit_images)

    print(f"Rank canvas size: {rank_canvas_size[1]}x{rank_canvas_size[0]}")
    print(f"Suit canvas size: {suit_canvas_size[1]}x{suit_canvas_size[0]}")

    build_templates(
        examples_by_label=rank_examples,
        output_dir=rank_output_dir,
        canvas_size=rank_canvas_size,
        force=args.force,
        debug_dir=debug_dir,
        debug_folder_name="normalized_ranks",
    )
    build_templates(
        examples_by_label=suit_examples,
        output_dir=suit_output_dir,
        canvas_size=suit_canvas_size,
        force=args.force,
        debug_dir=debug_dir,
        debug_folder_name="normalized_suits",
    )


if __name__ == "__main__":
    main()
