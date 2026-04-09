from pathlib import Path

import cv2


def load_grayscale_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: failed to load image: {path}")
        return None

    return image


def preprocess_symbol(image):
    _, threshold = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return threshold


def resize_to_template_size(image, template):
    template_height, template_width = template.shape[:2]
    resized = cv2.resize(image, (template_width, template_height))
    return resized


def match_symbol(symbol_image, templates_dir: Path):
    best_label = None
    best_score = -1.0

    for template_path in sorted(templates_dir.glob("*")):
        if template_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        template = load_grayscale_image(template_path)

        if template is None:
            continue

        symbol_processed = preprocess_symbol(symbol_image)
        template_processed = preprocess_symbol(template)

        resized_symbol = resize_to_template_size(symbol_processed, template_processed)

        result = cv2.matchTemplate(
            resized_symbol,
            template_processed,
            cv2.TM_CCOEFF_NORMED,
        )

        score = result[0][0]
        label = template_path.stem

        print(f"Template match: {label} -> {score:.4f}")

        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    rank_image_path = project_root / "data" / "processed" / "29_detect_best_corner_rank_region.jpg"
    suit_image_path = project_root / "data" / "processed" / "30_detect_best_corner_suit_region.jpg"

    rank_templates_dir = project_root / "data" / "templates" / "ranks"
    suit_templates_dir = project_root / "data" / "templates" / "suits"

    if not rank_image_path.exists() or not suit_image_path.exists():
        print("Error: rank or suit region image does not exist.")
        print("Run scripts/detect_symbols_from_best_corner.py first.")
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
    matched_rank, rank_score = match_symbol(rank_image, rank_templates_dir)

    print("\nMatching suit templates:")
    matched_suit, suit_score = match_symbol(suit_image, suit_templates_dir)

    if matched_rank is None or matched_suit is None:
        print("Error: failed to find valid template matches.")
        return

    print("\nFinal result:")
    print(f"Rank: {matched_rank} ({rank_score:.4f})")
    print(f"Suit: {matched_suit} ({suit_score:.4f})")
    print(f"Card: {matched_rank} of {matched_suit}")


if __name__ == "__main__":
    main()