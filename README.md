# COSC 594 — Project 3: Budget-Constrained Gait Recognition

## Overview

This capstone investigates whether frame selection can improve silhouette-based gait recognition when only a limited number of frames may be processed.

The project builds on two earlier stages:

- **Project 1:** Gait Energy Image (GEI) baseline.
- **Project 2:** GaitPart reproduction on CASIA-B using OpenGait.
- **Project 3:** Training-free frame-selection methods for a frozen GaitPart model, including robustness experiments under controlled silhouette corruption.

The main contribution is a controlled study of temporal coverage, silhouette diversity, silhouette quality, fixed quality-diversity combinations, and an adaptive quality-diversity strategy.

## Research Questions

This project investigates:

1. Does uniform temporal sampling outperform first-N truncation under the same frame budget?
2. Can pose-diversity-based selection improve limited-frame gait recognition?
3. Does silhouette-quality-based selection help when frames are unreliable?
4. Can combining quality and diversity provide a stronger trade-off?
5. Can an adaptive quality-diversity strategy respond to changing sequence reliability?
6. What computational overhead do the different frame-selection methods introduce?

## Dataset and Model

The experiments use the **CASIA-B** gait dataset.

- Training subjects: **001–074**
- Testing subjects: **075–124**
- Walking conditions:
  - NM — normal walking
  - BG — carrying a bag
  - CL — clothing variation
- Camera viewpoints: **11 views from 000° to 180°**
- Recognition model: **GaitPart**
- GaitPart checkpoint: **30,000 training iterations**
- Matching metric: **Euclidean distance**
- Primary inference budget: **15 frames**

The GaitPart model remains frozen during Project 3. Frame-selection experiments modify only which frames are supplied during inference; the model is not retrained.

The checkpoint used for the experiments is:

```text
GaitPart_Full_B32-30000.pt
SHA256: 86840d68de646c817683344c3886eea6998696df402f3f0b7aa00564ae0abc34
```

CASIA-B and the model checkpoint are not redistributed in this repository.

## Frame-Selection Methods

### First-N

Select the first N frames available in the sequence.

This reproduces the temporal truncation baseline used in Project 2.

### Uniform

Select N frames distributed approximately uniformly across the complete sequence.

This tests whether simple temporal coverage is better than taking only the beginning of a walking sequence.

### Diversity

Use greedy maximin selection based on mean absolute silhouette difference.

The goal is to select visually different gait poses instead of redundant frames.

### Quality

Score frames using a silhouette-quality heuristic based on foreground area and spatial consistency relative to the sequence.

Frames with unusual silhouette size, location, or missing foreground are penalized.

### Combined Quality + Diversity

Combine the quality and diversity criteria.

The main fixed configuration uses a quality weight of **0.5**.

Additional fixed weights of **0.25** and **0.90** were evaluated to understand the effect of changing the balance between quality and diversity.

### Adaptive Quality + Diversity

The adaptive method estimates the fraction of frames whose quality score is below 0.5.

It then computes:

```text
alpha = 0.25 + 0.65 * unreliable_fraction
```

The resulting quality weight ranges from **0.25 to 0.90**.

Clean sequences generally produce weights near 0.25 because very few frames are classified as unreliable.

When more low-quality frames appear, the method increases its emphasis on silhouette quality.

## Clean 15-Frame Results

The primary clean experiment uses a fixed budget of **15 frames**.

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

### Clean-Sequence Findings

Uniform sampling produces a large improvement over first-N truncation.

This shows that simple temporal coverage is already much more effective than taking only the beginning of the sequence.

Diversity performs slightly better than uniform sampling.

The **combined 0.5 strategy performs best overall** on the clean 15-frame experiment.

Quality-only selection performs worse than diversity-based approaches, showing that selecting only visually clean silhouettes does not necessarily provide enough pose coverage.

Giving quality too much weight, as in the 0.90 configuration, also reduces recognition performance.

## Adaptive-Weight Analysis

The adaptive method was audited over **5,485 test sequences**.

On clean CASIA-B sequences:

- Mean adaptive quality weight: approximately **0.252**
- Median quality weight: **0.25**
- Approximately **99.74%** of sequences were closest to the fixed 0.25 configuration.
- Adaptive selection exactly matched fixed 0.25 selection in approximately **98.30%** of clean sequences.

This explains why adaptive and fixed 0.25 have nearly identical clean recognition performance.

The adaptive method becomes more distinct when silhouette reliability decreases.

## Controlled Corruption Study

To evaluate robustness, controlled lower-body silhouette corruption was introduced into held-out probe sequences.

The primary corruption protocol:

- Corrupts approximately **30% of probe frames**
- Removes the lower **50% of the silhouette foreground**
- Leaves gallery sequences clean
- Uses deterministic random seeds
- Does not modify the original Project 2 dataset

The corruption was applied only to probe sequences:

- `nm-05`
- `nm-06`
- `bg-01`
- `bg-02`
- `cl-01`
- `cl-02`

The generated corrupted dataset contains the same CASIA-B test protocol used by the clean experiment.

## 30% Corruption Results

At a 15-frame budget:

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

The combined 0.5 method again performs best under moderate corruption.

Adaptive selection performs better than fixed 0.25 under this setting, showing that changing the weight according to sequence reliability provides a modest benefit.

## Selection-Behavior Analysis

The 30% corruption experiment was also analyzed at the selected-frame level.

| Method | Corrupted Frames Selected | Sequences Selecting Zero Corrupted Frames |
|---|---:|---:|
| First-N | 30.23% | 0.24% |
| Uniform | 29.84% | 0.33% |
| Diversity | 9.94% | 0.00% |
| Quality | 0.02% | 99.91% |
| Combined 0.25 | 8.69% | 0.03% |
| Adaptive | 6.13% | 13.55% |
| Combined 0.5 | 3.40% | 51.09% |
| Combined 0.90 | 0.02% | 99.91% |

Quality-only selection nearly eliminates corrupted frames but sacrifices temporal and pose diversity.

Diversity remains robust even though it still selects some corrupted frames.

Combined 0.5 provides a stronger balance by maintaining temporal diversity while substantially reducing exposure to damaged silhouettes.

Detailed analysis is available in:

```text
results/tables/selection_behavior_audit.csv
results/tables/corruption_audit.csv
results/tables/adaptive_weight_audit.csv
```

## Corruption Severity Study

Nested corruption datasets were created at approximately:

- **15% corrupted frames**
- **30% corrupted frames**
- **50% corrupted frames**

The severity sets are nested so that increasing corruption extends the same underlying corrupted-frame set instead of independently sampling a completely different set.

Four representative methods were evaluated:

- Uniform
- Diversity
- Combined 0.5
- Adaptive

### Normal Walking Severity Results

| Corruption | Uniform | Diversity | Combined 0.5 | Adaptive |
|---|---:|---:|---:|---:|
| Clean | 86.36% | 86.98% | **87.45%** | 86.43% |
| 15% | 83.85% | 85.64% | **87.29%** | 86.36% |
| 30% | 80.25% | 85.51% | **86.89%** | 85.55% |
| 50% | 71.91% | **85.33%** | 81.32% | 66.65% |

The same trends occur across BG and CL conditions.

Detailed results are stored in:

```text
results/tables/severity_results.csv
results/figures/severity_nm.png
results/figures/severity_bg.png
results/figures/severity_cl.png
```

## Severe-Corruption Failure Case

The 50% corruption experiment reveals an important failure mode in the adaptive quality heuristic.

The quality score is **sequence-relative**.

When corrupted frames make up approximately half or more of the sequence, the reference statistics used by the quality heuristic can shift toward the damaged silhouettes.

In the severe-corruption audit, sequences with more than 50% corrupted frames showed the quality rule frequently treating damaged silhouettes as normal and clean silhouettes as abnormal.

For sequences above 50% corruption:

- Corrupted frames below the quality threshold: approximately **0.53%**
- Clean frames below the quality threshold: approximately **98.19%**
- Mean corrupted-frame quality: approximately **0.797**
- Mean clean-frame quality: approximately **0.272**

This explains the adaptive method's performance collapse at 50% corruption.

This failure case was retained as a research result rather than retuning the method after seeing the final evaluation.

The severe experiment also shows that the diversity-only strategy is significantly more robust when the quality reference becomes unreliable.

## Expanded GEI vs. GaitPart Baseline

Project 2 compared GEI and GaitPart only under a 90° normal-walking protocol.

Project 3 expands this comparison across:

- All **11 CASIA-B viewpoints**
- NM, BG, and CL conditions
- Same-view gallery/probe matching
- Held-out subjects 075–124

The exact Project 1 GEI preprocessing code was retained.

Before expanding the experiment, the original GEI result was independently reproduced:

```text
Subjects included: 50
Subjects excluded: 0
Correct: 95
Total probes: 100
Rank-1 accuracy: 95.00%
```

The raw GEI dataset and processed GaitPart dataset were also verified to contain the same **15 unavailable sequences**, ensuring matched data availability.

### Same-View Rank-1 Results

| Condition | GEI | GaitPart | GaitPart Improvement |
|---|---:|---:|---:|
| NM | 95.18% | **100.00%** | +4.82 pp |
| BG | 54.10% | **99.36%** | +45.26 pp |
| CL | 18.91% | **91.91%** | +73.00 pp |

This broader experiment shows that the difference between GEI and GaitPart becomes much larger under bag-carrying and clothing-change conditions.

These values are specifically **same-view results**.

They should not be confused with OpenGait's standard cross-view evaluation.

The same 30,000-iteration GaitPart checkpoint achieves:

| Condition | Cross-View Rank-1 |
|---|---:|
| NM | 94.20% |
| BG | 87.19% |
| CL | 73.03% |

Detailed comparison data is stored in:

```text
results/tables/gei_vs_gaitpart_same_view.csv
results/tables/baseline_comparison_summary.csv
results/figures/gei_vs_gaitpart_nm.png
results/figures/gei_vs_gaitpart_bg.png
results/figures/gei_vs_gaitpart_cl.png
```

## Computational Overhead

Frame-selection overhead was measured separately from GaitPart inference.

The final real-data benchmark used:

- **200 deterministic CASIA-B test sequences**
- Mean sequence length: **83.02 frames**
- Median sequence length: **82 frames**
- Minimum sequence length: **41 frames**
- Maximum sequence length: **146 frames**
- Frame budget: **15**
- Disk loading excluded from timing

### Real CASIA-B Selection Runtime

| Method | Mean | Median | 95th Percentile |
|---|---:|---:|---:|
| First-N | 0.0266 ms | 0.0248 ms | 0.0325 ms |
| Uniform | 0.0309 ms | 0.0290 ms | 0.0426 ms |
| Diversity | 4.1826 ms | 3.8630 ms | 6.0469 ms |
| Quality | 0.9826 ms | 0.8139 ms | 1.7752 ms |
| Combined | 4.9358 ms | 4.5096 ms | 7.2070 ms |
| Adaptive | 5.4173 ms | 5.0265 ms | 7.9558 ms |

Even the adaptive strategy adds only approximately **5.4 ms of mean CPU frame-selection overhead per sequence** in this benchmark.

These measurements exclude:

- Disk I/O
- GaitPart model inference
- GPU execution

Results are stored in:

```text
results/tables/sampling_benchmark_real.csv
results/tables/sampling_benchmark.csv
```

## Correctness Validation

The Project 3 implementation contains tests covering:

- First-N selection
- Uniform selection
- Diversity selection
- Quality scoring and selection
- Combined selection
- Adaptive weighting
- Deterministic lower-body corruption
- OpenGait adapter behavior
- Actual OpenGait `CollateFn` integration
- Independent hand-calculated sampling references
- Invalid-input behavior
- Input preservation
- Sorted and unique frame indices
- Short and empty sequences

Run the complete test suite with:

```bash
python -m unittest discover -s tests -v
```

Final verified result:

```text
Ran 87 tests in 0.220s

OK
```

The tests validate the implementation and integration logic.

Recognition accuracy is evaluated separately using CASIA-B experiments and is not inferred from the unit tests.

## Experimental Environment

The experiments were performed using:

```text
Operating system: Ubuntu 22.04 under WSL2
Python: 3.9.25
NumPy: 1.23.5
PyTorch: 2.2.2+cu121
CUDA runtime: 12.1
GPU: NVIDIA RTX 3060 Laptop GPU, 6 GB
```

The working Conda environment is stored in:

```text
environment.yml
```

To recreate it:

```bash
conda env create -f environment.yml
conda activate gaitpart-p2
```

The environment file records the working configuration used for the experiments.

Depending on hardware and CUDA availability, PyTorch may need to be installed separately using the appropriate CUDA build.

## OpenGait Setup

This project uses the OpenGait framework pinned to commit:

```text
4d228e154ae6f7473db49e0ef33493be0c137601
```

Clone the repository from the Project 3 root:

```bash
git clone https://github.com/ShiqiYu/OpenGait.git third_party/OpenGait
```

Check out the pinned revision:

```bash
git -C third_party/OpenGait checkout \
  4d228e154ae6f7473db49e0ef33493be0c137601
```

Install Project 3 sampling integration with:

```bash
python -m scripts.install_opengait_integration
```

The equivalent patch is also preserved in:

```text
patches/opengait_project3_sampling.patch
```

For the expanded same-view GEI/GaitPart evaluation, apply:

```bash
git -C third_party/OpenGait apply \
  ../../patches/gaitpart_same_view_all.patch
```

The sampling integration modifies OpenGait frame selection only when a Project 3 method is explicitly enabled in a configuration.

The same-view patch adds reporting for identical-view evaluation cells without changing GaitPart's identification calculations.

## Dataset Layout

The Project 3 GaitPart configurations expect the processed CASIA-B dataset in the sibling Project 2 repository:

```text
gait-recognition-project2/
└── data/
    └── processed/
        └── CASIA-B-pkl-compatible/
```

Project 3 and Project 2 are therefore expected to have a sibling layout such as:

```text
COSC594/
├── gait-recognition-project2/
└── gait-recognition-project3/
```

The CASIA-B dataset itself is not included in Git.

## Checkpoint Layout

The 30,000-iteration GaitPart checkpoint is expected under the Project 3 OpenGait checkout:

```text
third_party/OpenGait/
└── output/
    └── CASIA-B/
        └── GaitPart/
            └── GaitPart_Full_B32/
                └── checkpoints/
                    └── GaitPart_Full_B32-30000.pt
```

The checkpoint is excluded from Git because of its size.

## Raw GEI Data

The broader GEI comparison requires the original CASIA-B PNG silhouettes.

Set the environment variable `CASIAB_RAW_ROOT` to the directory containing the held-out subject folders:

```text
075/
076/
...
124/
```

For example:

```bash
CASIAB_RAW_ROOT=/path/to/GaitDatasetB-silh \
python -m scripts.evaluate_gei_broad
```

## Reproducing Key Analyses

### Run all tests

```bash
python -m unittest discover -s tests -v
```

### Benchmark selection overhead on real CASIA-B sequences

```bash
python -m scripts.benchmark_sampling_real
```

### Reproduce the broader GEI evaluation

```bash
CASIAB_RAW_ROOT=/path/to/GaitDatasetB-silh \
python -m scripts.evaluate_gei_broad
```

### Full-Sequence GaitPart Same-View Evaluation

From:

```text
third_party/OpenGait
```

run:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone \
  --nproc_per_node=1 \
  opengait/main.py \
  --cfgs ../../configs/gaitpart_full_same_view.yaml \
  --phase test \
  --log_to_file
```

### Parse GaitPart Same-View Results

From the Project 3 root:

```bash
python scripts/parse_gaitpart_same_view.py
```

### Rebuild GEI vs. GaitPart Comparison

```bash
python scripts/build_baseline_comparison.py
```

### Generate Corrupted Dataset

The controlled-corruption implementation is located in:

```text
src/corruption/occlusion.py
```

Dataset-generation and auditing scripts include:

```text
scripts/generate_corrupted_dataset.py
scripts/generate_corruption_severity.py
scripts/audit_corruption.py
scripts/audit_selection_behavior.py
scripts/audit_adaptive_weights.py
```

Generated corrupted datasets are excluded from Git.

Structured manifests and experiment results are retained under:

```text
results/tables/
```

## Repository Structure

```text
configs/            OpenGait experiment configurations
docs/               research proposal and project documentation
patches/            reproducible OpenGait modifications
results/figures/    generated experiment figures
results/raw/        preserved raw experiment logs
results/tables/     structured results, audits, and manifests
scripts/            experiment, evaluation, audit, and benchmark scripts
src/corruption/     controlled silhouette corruption
src/sampling/       frame-selection methods and OpenGait adapter
tests/              algorithm, reference, corruption, and integration tests
environment.yml     working Conda environment
```

## Limitations

The study uses a single GaitPart checkpoint and one dataset.

The results therefore should not automatically be generalized to other gait-recognition architectures or datasets.

The primary sampling experiments assume that the complete silhouette sequence is available before frame selection. These methods reduce the number of frames supplied to GaitPart but do not represent a strictly online streaming system.

The silhouette-quality measure is a heuristic rather than a ground-truth measure of biometric usefulness.

The severe 50% corruption experiment demonstrates a concrete failure of the sequence-relative quality reference.

The controlled lower-body corruption represents one form of silhouette segmentation failure. Real-world failures may contain different artifacts, occlusions, noise patterns, or segmentation errors.

The runtime benchmark measures CPU frame-selection time separately from GaitPart inference.

## Conclusion

The experiments demonstrate that **which frames are selected matters substantially when GaitPart operates under a limited inference budget**.

Uniform temporal sampling produces a large improvement over first-N truncation.

Diversity-aware selection provides additional robustness and remains especially strong under severe silhouette corruption.

A fixed combination of quality and diversity performs best under clean, mild, and moderate corruption.

The adaptive method responds appropriately under moderate corruption but exposes a clear failure mode when corrupted silhouettes become dominant and distort the sequence-relative quality reference.

The project therefore does not support a single frame-selection strategy as universally optimal.

Instead, the results demonstrate a measurable trade-off among temporal coverage, silhouette diversity, silhouette quality, robustness, and computational cost.