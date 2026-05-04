# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| CSRT | Sequences=2, Frames=1760 |
| KCF_TLD | Sequences=2, Frames=1760 |

## Overlap Quality
| Metric | Direction | CSRT | KCF_TLD |
| --- | --- | :---: | :---: |
| Mean IoU | Higher is better | 0.4771 | 0.0520 |
| Overlap AUC | Higher is better | 0.4778 | 0.0606 |
| IoU >= 0.50 Rate | Higher is better | 56.48% | 2.50% |
| IoU >= 0.75 Rate | Higher is better | 10.28% | 1.02% |

## Localization Accuracy
| Metric | Direction | CSRT | KCF_TLD |
| --- | --- | :---: | :---: |
| Mean Center Error (px) | Lower is better | 9.502 | 92.978 |
| Median Center Error (px) | Lower is better | 6.580 | 78.726 |
| Center Error <= 20px Rate | Higher is better | 93.47% | 10.57% |
| Normalized Precision@0.05 | Higher is better | 95.51% | 21.42% |

## Tracking Stability
| Metric | Direction | CSRT | KCF_TLD |
| --- | --- | :---: | :---: |
| Tracking Availability | Higher is better | 100.00% | 6.70% |
| Failure Frame Ratio | Lower is better | 0.00% | 93.30% |
| Longest Failure Streak Ratio | Lower is better | 0.00% | 41.70% |

## Computational Efficiency
| Metric | Direction | CSRT | KCF_TLD |
| --- | --- | :---: | :---: |
| Average FPS | Higher is better | 35.55 | 137.37 |
