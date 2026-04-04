# ml-bench

Continuous leakage landscape benchmarks for the [ml](https://github.com/epagogy/ml) package.

Results from 13 experiments across 2,000+ datasets measuring where ML workflow errors matter.

## Experiments

| ID | Description | Datasets |
|----|------------|----------|
| an | N-scaling: leakage x sample size | ~1,849 |
| ap | Seed dose-response (K=5..100) | ~2,047 |
| ao | CV coverage gap (95% CI) | ~2,047 |
| an_peeking | Model selection bias x N (19 configs) | ~997 |
| ac2 | Compound class II (screen/tune/seed) | ~1,849 |
| at_tune | HP tuning inflation x n x algorithm | ~1,849 |
| at_temporal | Temporal HP tuning leakage | ~1,592 |
| boundary | Temporal + group boundary leakage | ~1,592 |

## Data

All results in `data/` as append-only JSONL. Every row independently verifiable.

## Papers

- Roth (2026). *The Leakage Landscape*. [Zenodo](https://doi.org/10.5281/zenodo.19406148)
- Roth (2026). *A Grammar of Machine Learning*. [Zenodo](https://doi.org/10.5281/zenodo.19406355)
