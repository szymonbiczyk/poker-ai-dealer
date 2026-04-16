"""Legacy compatibility runner that delegates to the official MVP pipeline."""

from single_card_pipeline import (
    main,
    match_symbol,
    preprocess_symbol,
    resize_to_template_size,
)


if __name__ == "__main__":
    main()
