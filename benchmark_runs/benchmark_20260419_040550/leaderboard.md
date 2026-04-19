# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| KCF_TLD | Sequences=1, Frames=3085 |
| KCF | Sequences=1, Frames=3085 |

## Overlap Quality
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.0200 | 0.0154 |
| Overlap AUC | Higher is better | 0.0294 | 0.0246 |
| IoU >= 0.50 Rate | Higher is better | 1.33% | 0.58% |
| IoU >= 0.75 Rate | Higher is better | 0.49% | 0.49% |

## Localization Accuracy
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 230.468 | 103.538 |
| Median Center Error (px) | Lower is better | 244.325 | 101.531 |
| Center Error <= 20px Rate | Higher is better | 3.89% | 2.17% |
| Normalized Precision@0.05 | Higher is better | 4.02% | 4.89% |

## Tracking Stability
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 18.06% | 0.45% |
| Failure Frame Ratio | Lower is better | 81.94% | 99.55% |
| Longest Failure Streak Ratio | Lower is better | 31.22% | 99.55% |

## Computational Efficiency
| Metric | Direction | KCF_TLD | KCF |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 105.22 | 101.71 |
