# COSC 594 — Project 3: Gait Recognition Capstone

## Overview

This project investigates methods for improving gait recognition under limited video-frame budgets.

It builds upon previous work:

- **Project 1:** Gait Energy Image (GEI) baseline.
- **Project 2:** Reproduction and evaluation of GaitPart using CASIA-B.
- **Project 3:** Investigation of frame-selection strategies for improving gait identification.

## Research Problem

Gait recognition performance can decrease when fewer silhouette frames are available during inference.

This project investigates whether selecting informative frames can improve recognition compared with simple frame truncation.

## Proposed Approach

I plan to compare:

1. First-N frame selection
2. Uniform frame sampling
3. Motion-diversity-based sampling
4. Quality-aware sampling
5. Combined quality- and diversity-aware sampling

The existing trained GaitPart model will initially be used without retraining to support controlled comparisons.

## Research Questions

- Can informative frame selection improve identification accuracy under a fixed frame budget?
- Do different selection methods perform differently across walking conditions and camera viewpoints?
- What computational overhead is introduced by more sophisticated selection methods?

## Dataset

CASIA-B, using the established training and testing subject partition.

The dataset is not included in this repository.

## Evaluation

Planned evaluation metrics include:

- Rank-1 identification accuracy
- Performance across NM, BG, and CL walking conditions
- Recognition accuracy at different frame budgets
- Frame-selection runtime and computational overhead

All methods will be compared under consistent access to sequence information.

## Reproducibility

The repository will contain experimental configurations, source code, environment specifications, documented procedures, and results.

## Project Status

Research planning and repository setup.


