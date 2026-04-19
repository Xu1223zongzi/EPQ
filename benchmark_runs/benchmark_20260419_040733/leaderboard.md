# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| KCF_TLD | Sequences=2, Frames=1433 |
| KCF | Sequences=2, Frames=1433 |

## Overlap Quality
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.0610 | 0.0366 |
| Overlap AUC | Higher is better | 0.0695 | 0.0457 |
| IoU >= 0.50 Rate | Higher is better | 4.68% | 3.07% |
| IoU >= 0.75 Rate | Higher is better | 2.44% | 2.44% |

## Localization Accuracy
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 105.864 | 87.389 |
| Median Center Error (px) | Lower is better | 113.118 | 87.576 |
| Center Error <= 20px Rate | Higher is better | 15.70% | 9.28% |
| Normalized Precision@0.05 | Higher is better | 18.98% | 15.91% |

## Tracking Stability
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 19.82% | 2.65% |
| Failure Frame Ratio | Lower is better | 80.18% | 97.35% |
| Longest Failure Streak Ratio | Lower is better | 46.06% | 97.35% |

## Computational Efficiency
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 113.04 | 247.79 |
