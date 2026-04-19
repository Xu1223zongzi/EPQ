# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| KCF | Sequences=1, Frames=217 |
| KCF_TLD | Sequences=1, Frames=217 |

## Overlap Quality
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.0676 | 0.0676 |
| Overlap AUC | Higher is better | 0.0767 | 0.0767 |
| IoU >= 0.50 Rate | Higher is better | 6.91% | 6.91% |
| IoU >= 0.75 Rate | Higher is better | 5.07% | 5.07% |

## Localization Accuracy
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 59.554 | 59.554 |
| Median Center Error (px) | Lower is better | 42.535 | 42.535 |
| Center Error <= 20px Rate | Higher is better | 25.81% | 25.81% |
| Normalized Precision@0.05 | Higher is better | 35.48% | 35.48% |

## Tracking Stability
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 6.45% | 6.45% |
| Failure Frame Ratio | Lower is better | 93.55% | 93.55% |
| Longest Failure Streak Ratio | Lower is better | 93.55% | 93.55% |

## Computational Efficiency
| Metric | Direction | KCF | KCF_TLD |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 342.81 | 15.33 |
