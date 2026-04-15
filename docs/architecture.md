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
## Current repository note

The current MVP implementation lives mainly in the `scripts/` directory.

At this stage, the repository contains:
- step-by-step experimental scripts used during development and debugging,
- reusable helper code for contour detection and card extraction,
- an end-to-end single-card runner for the current MVP flow.

A later refactor may move the stable pipeline into the planned `src/` package structure.

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
- the current classification stage depends on stable corner crops and template quality,
- template matching is treated as an early MVP classifier, not a final general solution,
- handling more diverse card layouts and more robust corner segmentation will be improved in later iterations.

## Corner extraction note

The current corner extraction uses fixed width/height ratios based on the extracted card image.

This is acceptable for the MVP and controlled conditions, but it may become unreliable for:
- cards with different proportions,
- different card designs,
- different margin sizes,
- more diverse real-world inputs.

A future improvement should normalize extracted cards to a fixed canonical size before corner cropping, or detect the rank/suit region more explicitly inside the corner area.

## Current MVP corner-segmentation note

The current MVP extracts rank and suit from a fixed-ratio corner crop taken from the normalized card image.

Current symbol separation logic:
- threshold the selected corner,
- find external contours,
- filter out very small candidates,
- choose a rank anchor with a strong top-left / top-row preference,
- optionally merge a second nearby box into the rank to support multipart ranks such as "10",
- build the final rank box first,
- then choose the suit candidate below the rank and near the same X column,
- treat the remaining mismatched candidates as noise or inner-card artifacts.

This works for many controlled inputs, but it can fail when:
- the fixed corner crop includes part of a large inner suit symbol,
- the card design places artwork or large pips too close to the corner area,
- the rank and suit spacing differs from the layouts used during early testing,
- the input resolution is too low for stable contour separation.

Known implication:
- the current MVP is suitable for controlled validation and architecture proof-of-concept,
- but corner segmentation is still a brittle step and is a likely target for improvement in the next iteration.

Possible future improvements:
- reduce the crop width dynamically,
- normalize extracted cards to a canonical size before corner cropping,
- filter candidate contours by stronger positional priors inside the corner,
- detect rank/suit regions more explicitly inside the corner,
- replace template-based symbol classification with learned rank/suit classifiers while keeping earlier card-normalization stages.