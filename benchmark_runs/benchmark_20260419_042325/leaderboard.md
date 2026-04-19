# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| KCF_TLD | Sequences=2, Frames=618 |
| KCF | Sequences=2, Frames=618 |

## Overlap Quality
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.0587 | 0.0442 |
| Overlap AUC | Higher is better | 0.0672 | 0.0531 |
| IoU >= 0.50 Rate | Higher is better | 3.88% | 3.88% |
| IoU >= 0.75 Rate | Higher is better | 2.91% | 2.91% |

## Localization Accuracy
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 130.123 | 83.531 |
| Median Center Error (px) | Lower is better | 135.932 | 84.750 |
| Center Error <= 20px Rate | Higher is better | 15.05% | 11.17% |
| Normalized Precision@0.05 | Higher is better | 26.21% | 18.77% |

## Tracking Stability
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 69.58% | 3.40% |
| Failure Frame Ratio | Lower is better | 30.42% | 96.60% |
| Longest Failure Streak Ratio | Lower is better | 0.65% | 96.60% |

## Computational Efficiency
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 119.64 | 295.68 |
