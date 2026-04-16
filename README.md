# poker-ai-dealer

Poker AI Dealer is a computer vision project focused on recognizing playing cards from images or camera input and supporting future Texas Hold’em game flow features.

The long-term goal is to build an intelligent dealer assistant that can:
- detect cards on the table,
- identify their rank and suit,
- track game state,
- and later support additional poker-related features such as chip counting, round progression, and structured game analysis.

## Project Goals

This project is being built step by step, starting from the computer vision core.

Main goals:
- detect playing cards from images or camera input,
- classify card rank and suit,
- support recognition of multiple cards on the table,
- build a structured game-state pipeline for Texas Hold’em,
- provide a foundation for future mobile and backend integration.

## MVP Scope

The current MVP focuses on the simplest useful version of the system:

- load a static image,
- detect a single card,
- extract and normalize the card region,
- choose the better corner,
- segment rank and suit,
- match them against saved templates,
- return the result in a structured format.

Example output:
- `A of spades`
- `10 of hearts`
- `Q of hearts`

Current MVP runner: `scripts/run_single_card_pipeline.py`  
Pipeline step scripts: `scripts/pipeline_steps/`  
Prerequisite: rank and suit templates must exist in `data/templates/`.

## Planned Tech Stack

- **Python** – main development language
- **OpenCV** – image processing and card detection
- **PyTorch** – future model development and classification
- **FastAPI** – future API layer
- **Mobile client** – future camera input interface

## Development Approach

The project is divided into stages:

1. **Static image recognition**
   - detect and classify cards from still images

2. **Multi-card recognition**
   - detect multiple cards in one scene

3. **Game-state logic**
   - distinguish player cards and community cards
   - track flop / turn / river

4. **API integration**
   - expose recognition results through a backend service

5. **Mobile / camera integration**
   - use a camera stream as the input source

6. **Advanced features**
   - chip counting
   - poker odds support
   - dealer assistant logic

## Current Status

Implemented MVP for static single-card recognition under controlled conditions.

Current implemented flow:
- image loading,
- contour-based card detection,
- perspective normalization,
- two-corner extraction,
- corner quality comparison,
- rank/suit segmentation,
- template-based matching,
- structured single-card output.

## Repository Structure

```text
poker-ai-dealer/
│
├── README.md
├── .gitignore
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── templates/
│   └── samples/
│
├── docs/
│   ├── ideas.md
│   ├── roadmap.md
│   └── architecture.md
│
├── notebooks/
├── tests/
├── src/
│
└── scripts/
    ├── run_single_card_pipeline.py
    ├── single_card_pipeline.py
    ├── match_card_from_best_corner.py
    ├── generate_rank_and_suit_templates.py
    ├── template_matching_helpers.py
    ├── symbol_detection_helpers.py
    ├── card_contour_helpers.py
    ├── card_extraction_helpers.py
    ├── corner_quality_helpers.py
    ├── corner_symbol_helpers.py
    ├── io_helpers.py
    ├── path_helpers.py
    └── pipeline_steps/
        ├── load_image.py
        ├── preprocess_image.py
        ├── detect_card_contour.py
        ├── extract_card.py
        ├── extract_card_corner.py
        ├── extract_both_card_corners.py
        ├── compare_corner_quality.py
        ├── detect_corner_symbols.py
        ├── detect_symbols_from_best_corner.py
        └── match_rank_and_suit_templates.py
```
## Notes

- `scripts/run_single_card_pipeline.py` is the official MVP entry point.
- `scripts/pipeline_steps/` contains step-by-step runnable scripts for understanding and inspecting the pipeline stage by stage.
- Shared logic should live in neutral helper modules, not only inside step-by-step scripts.
- `data/processed/` contains generated intermediate outputs and debug artifacts.