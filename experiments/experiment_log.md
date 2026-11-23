# Project 3 Experiment Log

## Project Goal

This project investigates training-free frame-selection strategies for a frozen GaitPart model under limited-frame inference.

The primary question is whether selecting informative frames can improve gait recognition compared with simply using the first N frames.

The main experiments use:

- Dataset: CASIA-B
- Training subjects: 001–074
- Test subjects: 075–124
- Conditions: NM, BG, CL
- Frozen GaitPart checkpoint: 30,000 iterations
- Primary frame budget: 15 frames
- Distance metric: Euclidean distance

The original Project 2 checkpoint and dataset preprocessing are preserved.

---

## 1. Project 2 Baseline Equivalence

Before introducing new sampling logic, Project 3 was checked against the Project 2 first-N implementation.

The Project 3 OpenGait integration was configured so that disabling the new methods preserves original behavior.

The 15-frame first-N result reproduced the Project 2 reference:

| Method | NM | BG | CL |
|---|---:|---:|---:|
| Project 2 first-N, 15 frames | 63.99% | 54.19% | 42.32% |
| Project 3 first-N, 15 frames | 63.99% | 54.19% | 42.32% |

This established behavioral equivalence before testing new frame-selection methods.

Relevant files:

```text
results/raw/baseline_equivalence/
patches/opengait_project3_sampling.patch
```

---

## 2. Clean 15-Frame Sampling Experiments

The following methods were evaluated under the same 15-frame budget:

- First-N
- Uniform
- Diversity
- Quality
- Combined quality-diversity
- Adaptive quality-diversity

Additional fixed combined weights of 0.25 and 0.90 were also tested.

### Results

| Method | NM | BG | CL |
|---|---:|---:|---:|
| First-N | 63.99% | 54.19% | 42.32% |
| Uniform | 86.36% | 77.17% | 56.33% |
| Diversity | 86.98% | 77.52% | 57.07% |
| Quality | 80.35% | 71.15% | 54.51% |
| Combined 0.5 | **87.45%** | **78.27%** | **57.95%** |
| Adaptive | 86.43% | 77.14% | 57.61% |
| Combined 0.25 | 86.43% | 77.12% | 57.54% |
| Combined 0.90 | 84.44% | 74.92% | 56.98% |

### Findings

Uniform sampling produced a large improvement over first-N truncation, showing that temporal coverage alone is important.

Diversity slightly improved over uniform sampling.

Quality-only selection was weaker than diversity-based approaches.

The fixed combined method with quality weight 0.5 performed best overall on clean sequences.

A quality weight of 0.90 reduced performance, suggesting that overemphasizing silhouette quality can remove useful pose diversity.

Results:

```text
results/tables/clean_15_results.csv
results/raw/clean_15/
```

---

## 3. Adaptive-Weight Audit

The adaptive strategy uses the fraction of low-quality frames to determine the quality-diversity weight:

```text
alpha = 0.25 + 0.65 * unreliable_fraction
```

The method was audited over 5,485 test sequences.

### Clean-Sequence Audit

- Mean adaptive weight: 0.2517
- Median adaptive weight: 0.25
- Minimum: 0.25
- Maximum: approximately 0.51
- Mean low-quality-frame fraction: approximately 0.0026
- 99.74% of clean sequences were closest to fixed weight 0.25
- Adaptive selection exactly matched fixed 0.25 in approximately 98.30% of sequences

### Interpretation

Clean CASIA-B sequences rarely trigger the quality mechanism.

This explains why adaptive and fixed 0.25 produce nearly identical clean recognition performance.

Result:

```text
results/tables/adaptive_weight_audit.csv
```

---

## 4. Controlled Corruption Implementation

A deterministic lower-body silhouette corruption was added to test robustness.

Protocol:

- Probe sequences only
- Gallery sequences remain clean
- 30% of probe frames corrupted in the primary experiment
- Lower 50% of the silhouette foreground removed from corrupted frames
- Master seed: 2026
- Original Project 2 dataset remains unchanged

Probe sequences:

```text
nm-05
nm-06
bg-01
bg-02
cl-01
cl-02
```

The generated test dataset contained:

- 3,298 corrupted probe sequence files
- 2,187 clean gallery/non-probe links
- 5,485 total test sequences

Two theoretically expected CASIA-B probe files were already absent from the original processed data:

```text
079/bg-02/054/054.pkl
079/bg-02/162/162.pkl
```

No files were fabricated to replace them.

### Independent Verification

The generated corruption was independently checked.

A sample sequence reproduced exactly using the recorded deterministic seed, and a corrupted sequence differed from the clean source only in the expected selected frames.

---

## 5. Corruption Quality Audit

The primary 30% corruption dataset was audited before recognition experiments.

Across 3,298 probe sequences:

- Actual corruption fraction: approximately 30.06%
- Corrupted frames below quality threshold: 99.92%
- Uncorrupted frames below threshold: 0.43%
- Mean corrupted-frame quality: 0.2990
- Mean uncorrupted-frame quality: 0.8623
- Clean adaptive alpha mean: 0.2517
- Corrupted adaptive alpha mean: 0.4470
- Median corrupted adaptive alpha: approximately 0.446
- Adaptive weight increased for 100% of corrupted sequences

### Interpretation

At moderate corruption, the quality heuristic successfully detects the damaged silhouettes and causes the adaptive method to increase its quality emphasis.

Result:

```text
results/tables/corruption_audit.csv
```

---

## 6. 30% Corruption Recognition Experiment

### Results

| Method | NM | BG | CL |
|---|---:|---:|---:|
| First-N | 56.32% | 46.74% | 35.17% |
| Uniform | 80.25% | 70.76% | 50.17% |
| Diversity | 85.51% | 75.50% | 55.12% |
| Quality | 64.98% | 57.15% | 39.77% |
| Combined 0.5 | **86.89%** | **77.14%** | **56.12%** |
| Adaptive | 85.55% | 76.18% | 55.44% |
| Combined 0.25 | 85.29% | 75.55% | 54.87% |
| Combined 0.90 | 73.60% | 64.48% | 46.19% |

### Findings

Combined 0.5 remained the best method.

Diversity was also highly robust.

Adaptive exceeded fixed 0.25 under corruption, showing a modest benefit from increasing the quality weight when unreliable frames are detected.

Quality-only and fixed 0.90 performed poorly despite strongly avoiding corrupted frames, indicating that avoiding damaged silhouettes alone is not sufficient if temporal diversity is lost.

Result:

```text
results/tables/corrupt_15_results.csv
results/raw/corrupt_15/
```

---

## 7. Selection-Behavior Audit

The selected frames were analyzed directly under 30% corruption.

| Method | Corrupted Frames Selected | Zero-Corruption Sequences |
|---|---:|---:|
| First-N | 30.23% | 0.24% |
| Uniform | 29.84% | 0.33% |
| Diversity | 9.94% | 0.00% |
| Quality | 0.02% | 99.91% |
| Combined 0.25 | 8.69% | 0.03% |
| Adaptive | 6.13% | 13.55% |
| Combined 0.5 | 3.40% | 51.09% |
| Combined 0.90 | 0.02% | 99.91% |

### Interpretation

Quality-only almost completely avoids corrupted frames but sacrifices pose coverage.

Diversity still selects some damaged frames but preserves strong recognition performance.

Combined 0.5 provides the strongest balance between avoiding corruption and preserving temporal diversity.

Result:

```text
results/tables/selection_behavior_audit.csv
```

---

## 8. Corruption Severity Study

Additional corruption levels were created at approximately:

- 15%
- 30%
- 50%

The severity sets were made nested:

```text
15% subset of 30% subset of 50%
```

This ensures that increasing severity expands the same underlying corruption rather than independently sampling unrelated frames.

Four methods were evaluated:

- Uniform
- Diversity
- Combined 0.5
- Adaptive

### 15% Corruption

| Method | NM | BG | CL |
|---|---:|---:|---:|
| Uniform | 83.85% | 74.80% | 53.45% |
| Diversity | 85.64% | 75.68% | 55.52% |
| Combined 0.5 | **87.29%** | **76.98%** | **57.03%** |
| Adaptive | 86.36% | 76.39% | 56.15% |

### 50% Corruption

| Method | NM | BG | CL |
|---|---:|---:|---:|
| Uniform | 71.91% | 61.55% | 42.48% |
| Diversity | **85.33%** | **75.81%** | **55.07%** |
| Combined 0.5 | 81.32% | 71.04% | 51.42% |
| Adaptive | 66.65% | 53.07% | 41.47% |

### Severity Trend

Combined 0.5 performs best under clean, mild, and moderate corruption.

Diversity is the strongest method under severe 50% corruption and remains remarkably stable as corruption increases.

Results:

```text
results/tables/severity_results.csv
results/figures/severity_nm.png
results/figures/severity_bg.png
results/figures/severity_cl.png
```

---

## 9. Severe-Corruption Failure Analysis

The 50% experiment exposed an important limitation of the adaptive quality heuristic.

The quality metric uses sequence-relative statistics.

When corrupted frames become approximately half or more of the sequence, the sequence reference can shift toward the corrupted silhouettes.

For sequences with more than 50% corrupted frames:

- Corrupted frames below the quality threshold: approximately 0.53%
- Clean frames below the quality threshold: approximately 98.19%
- Mean corrupted quality: approximately 0.797
- Mean clean quality: approximately 0.272

### Finding

The quality heuristic can reverse its interpretation when damaged frames become the majority.

This caused the adaptive method to increase emphasis on an incorrect quality reference and explains its severe 50% performance collapse.

The adaptive method was not retuned after observing this result.

The failure case is retained as part of the final analysis.

---

## 10. Expanded GEI Baseline Comparison

Project 2 compared GEI and GaitPart only at the 90° normal-walking view.

Project 3 expanded the comparison to:

- 11 camera viewpoints
- NM, BG, and CL
- Same-view gallery/probe matching
- Subjects 075–124

The exact Project 1 GEI preprocessing implementation was copied and validated using the original raw PNG silhouettes.

### Validation

The original Project 1 protocol was reproduced exactly:

```text
Subjects included: 50
Subjects excluded: 0
Correct: 95
Total probes: 100
Rank-1 accuracy: 95.00%
```

The raw GEI dataset and processed GaitPart dataset were also confirmed to have the same 15 unavailable sequences.

### Expanded Same-View Results

| Condition | GEI | GaitPart | Difference |
|---|---:|---:|---:|
| NM | 95.18% | **100.00%** | +4.82 pp |
| BG | 54.10% | **99.36%** | +45.26 pp |
| CL | 18.91% | **91.91%** | +73.00 pp |

### Interpretation

The expanded experiment provides a much stronger baseline comparison than the original single-view result.

GEI remains strong under normal walking but degrades substantially with bag carrying and clothing variation.

GaitPart is considerably more robust under these nuisance conditions.

Results:

```text
results/tables/gei_vs_gaitpart_same_view.csv
results/tables/baseline_comparison_summary.csv
results/figures/gei_vs_gaitpart_nm.png
results/figures/gei_vs_gaitpart_bg.png
results/figures/gei_vs_gaitpart_cl.png
```

---

## 11. Computational Overhead

An initial synthetic benchmark was used during development.

A final benchmark was then performed on real CASIA-B sequences.

Protocol:

- 200 deterministic test sequences
- Mean sequence length: 83.02 frames
- Median: 82 frames
- Minimum: 41
- Maximum: 146
- Frame budget: 15
- Disk I/O excluded
- GaitPart inference excluded

### Real-Sequence Runtime

| Method | Mean CPU Time | Median | P95 |
|---|---:|---:|---:|
| First-N | 0.0266 ms | 0.0248 ms | 0.0325 ms |
| Uniform | 0.0309 ms | 0.0290 ms | 0.0426 ms |
| Diversity | 4.1826 ms | 3.8630 ms | 6.0469 ms |
| Quality | 0.9826 ms | 0.8139 ms | 1.7752 ms |
| Combined | 4.9358 ms | 4.5096 ms | 7.2070 ms |
| Adaptive | 5.4173 ms | 5.0265 ms | 7.9558 ms |

### Finding

Uniform and first-N have essentially negligible selection overhead.

Quality selection remains below 1 ms mean CPU time.

Diversity and combined methods cost several milliseconds.

Adaptive is the most expensive method tested but still averages only approximately 5.4 ms of CPU frame-selection time per sequence.

Results:

```text
results/tables/sampling_benchmark.csv
results/tables/sampling_benchmark_real.csv
```

---

## 12. Correctness and Integration Validation

The final Project 3 test suite contains **87 tests**.

The tests cover:

- First-N and uniform sampling
- Diversity
- Quality
- Combined
- Adaptive weighting
- Controlled corruption
- OpenGait adapter behavior
- OpenGait CollateFn integration
- Independent reference calculations
- Invalid inputs
- Input preservation
- Sorted/unique indices
- Short and empty sequences

Final verification:

```text
Ran 87 tests in 0.220s

OK
```

These tests validate implementation correctness and integration behavior.

Recognition accuracy is evaluated separately through the CASIA-B experiments described above.

---

## Final Experimental Conclusions

The experiments support several conclusions:

1. **Temporal coverage matters.** Uniform sampling dramatically improves over first-N truncation.

2. **Diversity is consistently useful.** Diversity-aware selection improves over simple uniform sampling and is especially robust under severe corruption.

3. **Quality alone is insufficient.** Selecting only visually clean silhouettes can reduce important gait-pose coverage.

4. **Balanced quality and diversity works well.** Combined 0.5 performs best on clean, mild, and moderate corruption.

5. **Adaptive weighting helps only under some conditions.** It provides a modest improvement over fixed low-quality weighting under moderate corruption.

6. **The adaptive heuristic has a real failure case.** When damaged silhouettes become dominant, the sequence-relative quality reference can reverse and incorrectly penalize clean frames.

7. **The computational cost is small relative to model inference.** The most expensive tested selector adds approximately 5.4 ms of CPU preprocessing per sequence in the real-data benchmark.

The final results therefore do not identify one universally optimal frame-selection strategy.

Instead, they demonstrate a trade-off between temporal coverage, pose diversity, silhouette quality, corruption robustness, and computational cost.