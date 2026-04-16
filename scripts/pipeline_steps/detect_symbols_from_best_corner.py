from _path_setup import ensure_scripts_dir_on_path

ensure_scripts_dir_on_path()

from card_extraction_helpers import extract_card_from_image, extract_two_corners
from corner_quality_helpers import compare_two_corners
from io_helpers import load_image, save_images, show_images, wait_for_windows
from path_helpers import get_default_sample_image_path, get_processed_dir
from symbol_detection_helpers import detect_rank_and_suit


def print_card_extraction_summary(extraction) -> None:
    print(f"Found {len(extraction.contours)} contours.")
    print(f"Largest contour area: {extraction.contour_area}")
    print(f"Largest contour perimeter: {extraction.perimeter}")
    print(f"Approximated contour points: {len(extraction.approx)}")


def save_corner_selection_debug_images(
    output_dir,
    extracted_card,
    top_left_corner,
    bottom_right_corner_rotated,
    selected_corner,
) -> None:
    save_images(
        output_dir,
        [
            ("23_detect_best_corner_extracted_card.jpg", extracted_card),
            ("24_detect_best_corner_top_left.jpg", top_left_corner),
            (
                "25_detect_best_corner_bottom_right_rotated.jpg",
                bottom_right_corner_rotated,
            ),
            ("26_detect_best_corner_selected.jpg", selected_corner),
        ],
    )


def save_symbol_regions_debug_images(
    output_dir,
    threshold,
    boxed,
    rank_region,
    suit_region,
) -> None:
    save_images(
        output_dir,
        [
            ("27_detect_best_corner_threshold.jpg", threshold),
            ("28_detect_best_corner_symbol_boxes.jpg", boxed),
            ("29_detect_best_corner_rank_region.jpg", rank_region),
            ("30_detect_best_corner_suit_region.jpg", suit_region),
        ],
    )


def main() -> None:
    image_path = get_default_sample_image_path()
    output_dir = get_processed_dir(create=True)

    if not image_path.exists():
        print("Error: image file does not exist.")
        return

    image = load_image(image_path)

    if image is None:
        return

    try:
        extraction = extract_card_from_image(image)
    except ValueError as error:
        print(f"Error: {error}")
        return

    print_card_extraction_summary(extraction)

    extracted_card = extraction.extracted_card
    top_left_corner, bottom_right_corner_rotated = extract_two_corners(extracted_card)
    comparison = compare_two_corners(
        top_left_corner,
        bottom_right_corner_rotated,
        first_name="top_left",
        second_name="bottom_right_rotated",
    )
    top_left_score = comparison["scores"]["top_left"]
    bottom_right_score = comparison["scores"]["bottom_right_rotated"]

    print(f"Top-left score: {top_left_score:.4f}")
    print(f"Bottom-right rotated score: {bottom_right_score:.4f}")

    selected_corner_name = comparison["selected_name"]
    selected_corner = comparison["selected_image"]
    selected_score = comparison["selected_score"]

    print(f"Selected better corner: {selected_corner_name} ({selected_score:.4f})")

    save_corner_selection_debug_images(
        output_dir,
        extracted_card,
        top_left_corner,
        bottom_right_corner_rotated,
        selected_corner,
    )

    try:
        threshold, boxed, rank_region, suit_region = detect_rank_and_suit(selected_corner)
    except ValueError as error:
        print(f"Error: {error}")
        return

    save_symbol_regions_debug_images(
        output_dir,
        threshold,
        boxed,
        rank_region,
        suit_region,
    )

    show_images(
        [
            ("Extracted Card", extracted_card),
            ("Top Left Corner", top_left_corner),
            ("Bottom Right Corner Rotated", bottom_right_corner_rotated),
            ("Selected Better Corner", selected_corner),
            ("Selected Corner Threshold", threshold),
            ("Selected Corner Symbol Boxes", boxed),
            ("Rank Region", rank_region),
            ("Suit Region", suit_region),
        ]
    )
    wait_for_windows()


if __name__ == "__main__":
    main()
