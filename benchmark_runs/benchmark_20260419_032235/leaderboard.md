# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| KCF_TLD | Sequences=2, Frames=1760 |
| KCF | Sequences=2, Frames=1760 |

## Overlap Quality
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.0520 | 0.0215 |
| Overlap AUC | Higher is better | 0.0606 | 0.0306 |
| IoU >= 0.50 Rate | Higher is better | 2.50% | 1.36% |
| IoU >= 0.75 Rate | Higher is better | 1.02% | 1.02% |

## Localization Accuracy
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 92.978 | 96.973 |
| Median Center Error (px) | Lower is better | 78.726 | 92.516 |
| Center Error <= 20px Rate | Higher is better | 10.57% | 5.06% |
| Normalized Precision@0.05 | Higher is better | 21.42% | 8.92% |

## Tracking Stability
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 6.70% | 1.19% |
| Failure Frame Ratio | Lower is better | 93.30% | 98.81% |
| Longest Failure Streak Ratio | Lower is better | 41.70% | 98.81% |

## Computational Efficiency
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 141.31 | 315.76 |
