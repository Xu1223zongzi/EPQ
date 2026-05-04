# Benchmark Leaderboard

| Item | Value |
| --- | --- |
| CSRT | Sequences=2, Frames=618 |
| TLD | Sequences=2, Frames=618 |
| KCF | Sequences=2, Frames=618 |
| KCF_TLD | Sequences=2, Frames=618 |

## Overlap Quality
| Metric | Direction | CSRT | TLD | KCF | KCF_TLD |
| --- | --- | :---: | :---: | :---: | :---: |
| Mean IoU | Higher is better | 0.3836 | 0.1534 | 0.0442 | 0.0438 |
| Overlap AUC | Higher is better | 0.3864 | 0.1595 | 0.0531 | 0.0529 |
| IoU >= 0.50 Rate | Higher is better | 36.89% | 6.31% | 3.88% | 3.88% |
| IoU >= 0.75 Rate | Higher is better | 9.06% | 0.81% | 2.91% | 2.91% |

## Localization Accuracy
| Metric | Direction | CSRT | TLD | KCF | KCF_TLD |
| --- | --- | :---: | :---: | :---: | :---: |
| Mean Center Error (px) | Lower is better | 15.408 | 107.963 | 83.531 | 98.853 |
| Median Center Error (px) | Lower is better | 10.590 | 94.837 | 84.750 | 95.513 |
| Center Error <= 20px Rate | Higher is better | 82.69% | 28.96% | 11.17% | 11.17% |
| Normalized Precision@0.05 | Higher is better | 87.22% | 42.23% | 18.77% | 18.77% |

## Tracking Stability
| Metric | Direction | CSRT | TLD | KCF | KCF_TLD |
| --- | --- | :---: | :---: | :---: | :---: |
| Tracking Availability | Higher is better | 100.00% | 100.00% | 3.40% | 5.02% |
| Failure Frame Ratio | Lower is better | 0.00% | 0.00% | 96.60% | 94.98% |
| Longest Failure Streak Ratio | Lower is better | 0.00% | 0.00% | 96.60% | 53.72% |

## Computational Efficiency
| Metric | Direction | CSRT | TLD | KCF | KCF_TLD |
| --- | --- | :---: | :---: | :---: | :---: |
| Average FPS | Higher is better | 84.74 | 56.58 | 321.74 | 107.37 |
