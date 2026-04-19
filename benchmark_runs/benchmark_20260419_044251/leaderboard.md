# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| KCF_TLD | Sequences=2, Frames=618 |
| KCF | Sequences=2, Frames=618 |

## Overlap Quality
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.0571 | 0.0442 |
| Overlap AUC | Higher is better | 0.0658 | 0.0531 |
| IoU >= 0.50 Rate | Higher is better | 3.88% | 3.88% |
| IoU >= 0.75 Rate | Higher is better | 2.91% | 2.91% |

## Localization Accuracy
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 68.171 | 83.531 |
| Median Center Error (px) | Lower is better | 65.508 | 84.750 |
| Center Error <= 20px Rate | Higher is better | 14.24% | 11.17% |
| Normalized Precision@0.05 | Higher is better | 24.27% | 18.77% |

## Tracking Stability
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 5.02% | 3.40% |
| Failure Frame Ratio | Lower is better | 94.98% | 96.60% |
| Longest Failure Streak Ratio | Lower is better | 88.51% | 96.60% |

## Computational Efficiency
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 109.67 | 319.16 |
