# poker-ai-dealer
Poker AI Dealer is a computer vision project focused on recognizing playing cards from camera input and supporting Texas Hold’em game flow.

The long-term goal is to build an intelligent dealer assistant that can detect cards on the table, identify their rank and suit, track the game state, and eventually support additional features such as chip counting, round progression, and poker-related analysis.

## Project Goals

This project is being built step by step, starting from the computer vision core.

Main goals:
- detect playing cards from images or camera input,
- classify card rank and suit,
- support recognition of multiple cards on the table,
- build a structured game-state pipeline for Texas Hold’em,
- provide a foundation for future mobile and backend integration.

## MVP Scope

The first MVP focuses on the simplest useful version of the system:

- load a static image,
- detect a single card,
- extract or crop the card region,
- recognize its rank and suit,
- return the result in a structured format.

Example output:
- `Ace of Spades`
- `10 of Hearts`

Current MVP runner: `scripts/match_card_from_best_corner.py`
Prerequisite: rank and suit templates must exist in `data/templates/`.

## Planned Tech Stack

- **Python** – main development language
- **OpenCV** – image processing and card detection
- **PyTorch** – model development and classification
- **FastAPI** – future API layer
- **Android / mobile client** – future camera input interface

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

5. **Mobile integration**
   - use a phone camera as the input source

6. **Advanced features**
   - chip counting
   - poker odds support
   - dealer assistant logic

## Current Status

Project setup and planning phase.

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
│   └── samples/
│
├── docs/
│   ├── ideas.md
│   ├── roadmap.md
│   └── architecture.md
│
├── src/
│   ├── detection/
│   ├── classification/
│   ├── pipeline/
│   └── utils/
│
├── notebooks/
├── tests/
└── scripts/
