# Roadmap

## Project vision

Poker AI Dealer is a computer vision project aimed at recognizing playing cards from images or camera input and supporting Texas Hold'em game flow.

The long-term goal is to build a system that can:
- detect cards on a table,
- identify their rank and suit,
- track the state of a hand,
- support future features such as chip counting, round progression, and poker-related analysis.

---

## Current development stage

Early MVP prototype implemented for static single-card recognition under controlled conditions.

Implemented prototype flow:
- load a static image from disk,
- detect a single card contour,
- normalize the card with perspective correction,
- extract both card corners,
- choose the more readable corner,
- segment rank and suit symbols,
- match them against saved templates,
- return a rank + suit prediction.

Current work is focused on:
- validating the pipeline on a small set of controlled test images,
- documenting current assumptions and known limitations,
- improving corner symbol separation for more card designs.

## Milestone 1 — Static single-card recognition

Goal: 
Build the simplest working prototype that can recognize one playing card from a static image under controlled conditions.

Scope:

- load a static image from disk,
- detect a single card in the image,
- crop and normalize the card region,
- extract rank and suit from a selected corner,
- classify rank and suit with template matching,
- return a structured result.

Definition of done:

- the program accepts a test image,
- a card contour is detected,
- the normalized card region can be visualized or saved,
- a corner region is selected and segmented into rank + suit,
- the system returns a rank + suit prediction,
- the prototype works on a small validation set of controlled images,
- current assumptions and failure cases are documented.

---

## Milestone 2 — Static multi-card recognition

Goal:
Detect and recognize multiple cards in a single image.

Scope:
- detect multiple card contours or regions,
- extract each card separately,
- classify each detected card,
- return a list of recognized cards.

Definition of done:
- the system works on table-like scenes with multiple cards,
- multiple cards can be extracted and labeled correctly,
- results are returned in a consistent structured format.

---

## Milestone 3 — Texas Hold'em table understanding

Goal:
Introduce basic game-state awareness.

Scope:
- distinguish community cards from player cards,
- represent flop / turn / river states,
- structure recognized cards into a game model.

Definition of done:
- the system can represent a Hold'em board state,
- detected cards are grouped logically,
- the program can track visible game progress.

---

## Milestone 4 — API layer

Goal:
Expose card recognition through a backend API.

Scope:
- create a FastAPI service,
- accept image input,
- return detected cards in JSON format,
- prepare the project for future client integration.

Definition of done:
- an API endpoint accepts an image,
- the response contains recognized card data,
- the pipeline can be triggered outside a local script.

---

## Milestone 5 — Live camera support

Goal:
Move from static images to real-time or near-real-time input.

Scope:
- connect camera input,
- process frames continuously or periodically,
- stabilize recognition across multiple frames.

Definition of done:
- the system reads from a camera source,
- cards can be recognized from a live stream,
- results are stable enough for demo usage.

---

## Milestone 6 — Advanced features

Possible future extensions:
- chip counting,
- poker hand evaluation,
- odds estimation,
- dealer assistant logic,
- mobile integration,
- game-history logging.

---

## Immediate next steps

- validate Milestone 1 on a small set of representative test cards,
- document the current single-card MVP entry point and required template setup,
- reduce duplication in debug / visualization helpers,
- improve project structure by separating experimental scripts from the main MVP path,
- update architecture notes to reflect the current rank/suit box-selection logic,
- decide whether the next iteration should improve classical CV heuristics or introduce a learned classifier for rank/suit recognition.