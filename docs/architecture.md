# Architecture

## Overview

The system is planned as a modular computer vision pipeline for playing card recognition.

Initial focus:
- static images,
- single-card detection,
- clean and debuggable processing pipeline.

Later stages will extend the system toward:
- multi-card scenes,
- game-state understanding,
- API integration,
- live camera input.

---

## High-level pipeline


Input image
-> preprocessing
-> card detection
-> card extraction / normalization
-> rank and suit classification
-> structured output

## Main modules

### 1. Input layer

Responsible for reading input data.

Possible sources:
- image files,
- video files,
- live camera feed,
- future mobile client input.

Responsibilities:
- load image data,
- validate input,
- pass data to the pipeline.

---

### 2. Preprocessing

Responsible for preparing the image for detection.

Possible operations:
- resizing,
- grayscale conversion,
- blur / denoising,
- thresholding,
- edge detection,
- contrast normalization.

Goal:
improve consistency and make card detection easier.

---

### 3. Card detection

Responsible for locating candidate card regions in the image.

Possible approaches:
- contour-based detection,
- perspective-aware rectangle detection,
- object detection model in later versions.

Initial preferred approach:
classical computer vision with OpenCV, because it is simpler, faster to debug, and better for an early MVP.

---

### 4. Card extraction / normalization

Responsible for isolating each detected card.

Possible operations:
- crop bounding region,
- perspective transform,
- rotation normalization,
- size normalization.

Goal:
produce a clean card image suitable for classification.

---

### 5. Rank and suit classification

Responsible for identifying card value and suit.

Possible approaches:
- template matching,
- corner-symbol extraction,
- classical CV heuristics,
- CNN / PyTorch model.

Initial direction:
start with the simplest approach that works on controlled inputs, then evaluate whether a learned model is necessary.

---

### 6. Structured output

Responsible for converting recognition results into a consistent format.

Example:

```json
{
  "cards": [
    {
      "rank": "A",
      "suit": "spades"
    }
  ]
}
```
This structure will later support:
- multiple cards,
- player/board grouping,
- API responses,
- game-state logic.

---

## Project structure

```text
src/
├── detection/
├── classification/
├── pipeline/
└── utils/
```
### detection
Card localization logic.

### classification
Rank and suit recognition logic.

### pipeline
High-level orchestration of the full recognition flow.

### utils
Shared helper functions.

---

## Design principles

- keep the pipeline modular,
- separate experimentation from reusable code,
- prefer simple approaches before complex ML solutions,
- make intermediate outputs easy to inspect,
- build incrementally from static images to live input.

---

## Early technical decisions

### Decision 1: start with static images
Reason:
- easier debugging,
- easier visualization,
- lower complexity than live video.

### Decision 2: start with classical CV for detection
Reason:
- fast iteration,
- lower setup cost,
- easier to understand and debug.

### Decision 3: keep classification approach open
Reason:
- template matching may be enough for early controlled scenarios,
- a PyTorch-based model can be introduced later if needed.

---

## Risks and challenges

- variable lighting conditions,
- perspective distortion,
- overlapping cards,
- noisy backgrounds,
- card orientation,
- generalization from controlled to real-world scenes.

---

## Future extensions

- API service with FastAPI,
- real-time inference,
- mobile camera input,
- chip detection and counting,
- poker hand evaluation,
- game-state tracking.

## Current assumptions

- the card should be visually separable from the background,
- early contour detection is expected to work best with high card-to-background contrast,
- the current MVP is focused on controlled image conditions,
- handling low-contrast scenes will be improved in later iterations.

## Corner extraction note

The current corner extraction uses fixed width/height ratios based on the extracted card image.

This is acceptable for the MVP and controlled conditions, but it may become unreliable for:
- cards with different proportions,
- different card designs,
- different margin sizes,
- more diverse real-world inputs.

A future improvement should normalize extracted cards to a fixed canonical size before corner cropping, or detect the rank/suit region more explicitly inside the corner area.