# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| KCF | Sequences=2, Frames=618 |
| KCF_TLD | Sequences=2, Frames=618 |

## Overlap Quality
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.0442 | 0.0438 |
| Overlap AUC | Higher is better | 0.0531 | 0.0529 |
| IoU >= 0.50 Rate | Higher is better | 3.88% | 3.88% |
| IoU >= 0.75 Rate | Higher is better | 2.91% | 2.91% |

## Localization Accuracy
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 83.531 | 112.915 |
| Median Center Error (px) | Lower is better | 84.750 | 111.369 |
| Center Error <= 20px Rate | Higher is better | 11.17% | 11.17% |
| Normalized Precision@0.05 | Higher is better | 18.77% | 18.77% |

## Tracking Stability
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 3.40% | 5.99% |
| Failure Frame Ratio | Lower is better | 96.60% | 94.01% |
| Longest Failure Streak Ratio | Lower is better | 96.60% | 53.72% |

## Computational Efficiency
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 317.04 | 155.41 |
