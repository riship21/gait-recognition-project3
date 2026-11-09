# COSC 594 — Project 3 Research Proposal

## Project Title

Budget-Constrained Gait Recognition Using Quality and Diversity-Aware Frame Selection

## 1. Background

Gait recognition identifies individuals based on their walking patterns.

In Project 1, a Gait Energy Image (GEI) baseline was implemented using silhouette averaging and Euclidean-distance matching.

In Project 2, GaitPart was reproduced on the CASIA-B dataset using the OpenGait framework. The model was trained for 30,000 iterations and evaluated under normal walking (NM), bag-carrying (BG), and clothing-variation (CL) conditions.

Project 2 also investigated how limiting the number of available silhouette frames affects recognition performance.

## 2. Problem Statement

Gait recognition performance decreases when fewer frames are available during inference.

For example, the Project 2 GaitPart model achieved the following cross-view Rank-1 identification results:

| Maximum Frames | NM | BG | CL |
|---|---:|---:|---:|
| 15 | 63.99% | 54.19% | 42.32% |
| 30 | 86.25% | 76.34% | 62.16% |
| 50 | 91.77% | 84.22% | 70.13% |
| Unlimited | 94.20% | 87.19% | 73.03% |

These results were obtained by retaining the first N available frames.

However, the first N frames may not always represent the most informative walking poses.

This motivates investigating whether selecting frames according to temporal coverage, pose diversity, or silhouette quality can improve identification performance.

## 3. Research Objective

Develop and evaluate lightweight, training-free frame-selection strategies for an existing GaitPart model.

The primary objective is to determine whether more informative frame selection can improve identification accuracy under fixed frame-processing budgets.

## 4. Research Questions

**RQ1:** Does uniform frame sampling outperform first-N frame selection under equal frame budgets?

**RQ2:** Can motion-diversity-based sampling improve recognition by selecting different walking poses?

**RQ3:** Does silhouette-quality-based selection improve recognition under different walking conditions?

**RQ4:** Can a combined quality- and diversity-aware strategy outperform simpler sampling methods?

**RQ5:** What computational overhead do these selection methods introduce?

## 5. Proposed Methods

### Method A — First-N Sampling

Select the first N frames from a silhouette sequence.

This reproduces the Project 2 temporal truncation baseline.

### Method B — Uniform Sampling

Select N frames distributed across the complete available sequence.

This provides a simple baseline for evaluating the benefits of temporal coverage.

### Method C — Motion-Diversity Sampling

Select frames representing different silhouette poses.

Candidate approaches include measuring pairwise silhouette differences or using a greedy diversity-selection procedure.

The final algorithm will be specified before experimental evaluation.

### Method D — Quality-Aware Sampling

Assign each silhouette a quality score using measurable properties such as foreground completeness, abnormal foreground area, and silhouette position.

Select frames based on the resulting quality scores.

These are proposed quality indicators, not established ground-truth measures of biometric usefulness.

### Method E — Combined Quality and Diversity

Combine quality and diversity criteria to avoid selecting many similar silhouettes while preferring frames that satisfy basic quality requirements.

The combination procedure and any weights will be defined and tested using a development partition.

## 6. Experimental Design

### Dataset

CASIA-B.

Training subjects: 001–074.

Held-out testing subjects: 075–124.

### Model

GaitPart, using the existing 30,000-iteration checkpoint from Project 2.

The model will initially remain frozen during all frame-selection comparisons.

### Frame Budgets

- 15 frames
- 30 frames
- 50 frames

Unlimited-frame inference will serve as a reference, not as an equal-budget competitor.

### Controlled Variables

The experiments will preserve:

- GaitPart checkpoint and model architecture
- Silhouette preprocessing
- Gallery and probe definitions
- Evaluation subjects
- Distance metric
- Rank-1 evaluation protocol
- Frame budget within each comparison

### Evaluation Protocol

The primary experiment will evaluate selection from an already available complete walking sequence.

Each method will select at most N frames from the same input sequence.

Selection algorithms must not use identity labels or recognition outcomes when choosing frames.

Any hyperparameters will be determined using a separate development partition rather than optimized on the final held-out test results.

A limited-observation-window experiment may be included as a separate analysis if time permits.

### Evaluation Metrics

- Rank-1 identification accuracy
- NM, BG, and CL cross-view performance
- Absolute accuracy difference from first-N and uniform baselines
- Frame-selection execution time
- End-to-end inference time, where measurable

## 7. Hypotheses

**H1:** Uniform sampling will outperform first-N truncation under some limited-frame conditions.

**H2:** Diversity-aware sampling may improve recognition by reducing redundant poses.

**H3:** Quality-aware sampling may provide improvements when sequences contain incomplete or unreliable silhouettes.

**H4:** Combining quality and diversity may improve performance, but the additional computational cost may not always be justified.

These hypotheses will be evaluated experimentally and are not assumed to be true.

## 8. Expected Contributions

The project aims to provide:

1. A reproducible comparison of frame-selection strategies for GaitPart.
2. A lightweight, training-free quality- and diversity-aware sampling method.
3. An analysis of recognition accuracy under fixed frame budgets.
4. An evaluation of the trade-off between recognition accuracy and computational overhead.
5. A discussion of where each selection strategy succeeds or fails.

The work does not claim that quality-aware frame selection is a new research concept. Its contribution will be the specific method, controlled evaluation, and analysis within this GaitPart setting.

## 9. Limitations

- The initial study uses one trained GaitPart checkpoint.
- The CASIA-B dataset may not represent every real-world gait-recognition environment.
- Complete-sequence sampling assumes access to all frames before selection.
- Quality measures based on silhouettes may not perfectly predict recognition usefulness.
- Improvements may vary by viewing angle and walking condition.
- A more complicated method may perform worse than uniform sampling.

## 10. Reproducibility

The repository will include:

- Source code for sampling methods
- Unit tests
- Experimental configurations
- Environment and dependency information
- Evaluation scripts
- Original experiment logs
- Results and figures
- Instructions for reproducing the experiments

The CASIA-B dataset will not be redistributed.

## 11. Refined Capstone Research Contribution

Motivation

Our initial research investigates whether frame selection can improve GaitPart recognition under limited frame budgets.

To extend this into a stronger capstone contribution, we will investigate how frame selection should adapt when silhouette quality varies within a walking sequence.

Proposed Adaptive Method

We propose an adaptive quality-diversity frame-selection strategy.

Unlike our existing combined method, which uses a fixed quality weight of 0.5, the proposed method will adjust the balance between quality and diversity based on measurable sequence characteristics.

For example, a sequence containing inconsistent or incomplete silhouettes may benefit from emphasizing quality, while a sequence containing consistently reliable silhouettes may benefit from emphasizing pose diversity.

The exact adaptation rule will be defined before evaluating the final held-out test data.

Robustness Experiments

We will evaluate the proposed method on both original CASIA-B sequences and sequences containing controlled synthetic corruption.

Potential corruptions include partial occlusion, missing silhouette regions, and foreground noise.

All methods will use the same corruption settings, input sequences, frame budgets, and pretrained GaitPart checkpoint.

Research Evaluation

We will compare the adaptive method against:

First-N sampling

Uniform sampling

Motion-diversity sampling

Quality-aware sampling

Fixed-weight quality-diversity sampling

We will measure Rank-1 identification accuracy, computational overhead, and performance across walking conditions and viewpoints.

Expected Research Contribution

The goal is to determine whether adapting frame-selection behavior to sequence reliability improves recognition compared with fixed sampling strategies.

We will report improvements, failures, and trade-offs rather than assuming the adaptive method must outperform existing approaches.