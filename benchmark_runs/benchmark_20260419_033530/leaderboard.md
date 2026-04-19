# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| KCF | Sequences=2, Frames=1760 |
| KCF_TLD | Sequences=2, Frames=1760 |

## Overlap Quality
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.0215 | 0.0173 |
| Overlap AUC | Higher is better | 0.0306 | 0.0268 |
| IoU >= 0.50 Rate | Higher is better | 1.36% | 1.36% |
| IoU >= 0.75 Rate | Higher is better | 1.02% | 1.02% |

## Localization Accuracy
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 96.973 | 141.517 |
| Median Center Error (px) | Lower is better | 92.516 | 137.522 |
| Center Error <= 20px Rate | Higher is better | 5.06% | 3.92% |
| Normalized Precision@0.05 | Higher is better | 8.92% | 6.59% |

## Tracking Stability
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 1.19% | 3.01% |
| Failure Frame Ratio | Lower is better | 98.81% | 96.99% |
| Longest Failure Streak Ratio | Lower is better | 98.81% | 60.91% |

## Computational Efficiency
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 62.49 | 230.36 |
