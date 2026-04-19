# EPQ

统一的无人机目标跟踪实验框架，当前支持 KCF、CSRT、TLD 和 KCF+TLD 重定位四种算法。

常用运行命令已固定保存到 [COMMANDS.md](COMMANDS.md)。如果聊天窗口里的指令看不到了，直接打开这个文件即可。

## UAV123 单序列测评

UAV123 目录结构默认约定为：

- data_seq/UAV123/<sequence_name>/
- anno/UAV123/<sequence_name>.txt

可以直接使用数据集根目录和序列名启动，例如：

```bash
python KCF/KCF.py --uav123-root D:/UAV123 --sequence-name bike1 --save-video
```

KCF 主跟踪 + TLD 重定位模式可这样启动：

```bash
python KCF_TLD/KCF_TLD.py --uav123-root D:/UAV123 --sequence-name bike1 --save-video
```

也支持直接把 --uav123-root 指到 data_seq/UAV123 这一层，例如：

```bash
python CSRT/CSRT.py --uav123-root D:/论文/UAV123/data_seq/UAV123 --sequence-name bike1 --save-video
```

也可以手动指定图像序列目录和标注文件：

```bash
python CSRT/CSRT.py --sequence-dir D:/UAV123/data_seq/UAV123/bike1 --annotation-file D:/UAV123/anno/UAV123/bike1.txt
```

运行完成后，会在 experiment_runs 下生成：

- frame_log.csv：逐帧日志
- summary.json：汇总指标
- overlay.mp4：带框可视化视频（仅在 --save-video 时生成）

当前 summary.json 会输出以下 benchmark 指标：

- average_iou
- overlap_auc
- average_center_error
- median_center_error
- overlap_recall_iou_0_5
- overlap_recall_iou_0_75
- center_precision_20px
- normalized_precision_0_05
- tracking_availability
- failure_frame_ratio
- longest_failure_streak_ratio

## UAV123 批量测评与论文图表

可以使用批量脚本一次性评估多种算法，并生成论文中常见的汇总表格与曲线图：

```bash
python uav123_benchmark.py --uav123-root D:/论文/UAV123/data_seq/UAV123 --algorithms KCF CSRT TLD KCF_TLD --max-sequences 5
```

常见输出文件包括：

- per_sequence_results.csv：每个序列、每个算法的详细结果
- aggregate_results.csv：算法总体平均结果
- leaderboard_metrics.csv：按指标转置后的总表 CSV
- leaderboard.md：论文风格 Markdown 总表
- leaderboard.png：论文风格 PNG 总表
- overlap_curve.png：IoU 阈值召回曲线
- center_precision_curve.png：像素中心误差阈值曲线
- normalized_precision_curve.png：归一化中心误差阈值曲线
- overlap_metrics.png：重叠类指标分组图
- localization_metrics.png：定位类指标分组图
- stability_metrics.png：稳定性指标分组图
- efficiency_metrics.png：效率指标分组图

如果只想测试几个指定序列，例如 bike1、car1_s、uav3：

```bash
python uav123_benchmark.py --uav123-root D:/论文/UAV123/data_seq/UAV123 --algorithms KCF CSRT KCF_TLD --sequence-names bike1 car1_s uav3
```

如果担心某些序列特别慢，可以增加跳帧采样、降低评测分辨率，并给单序列设置超时，例如：

```bash
python uav123_benchmark.py --uav123-root D:/论文/UAV123/data_seq/UAV123 --algorithms KCF CSRT TLD KCF_TLD --frame-step 2 --frame-width 480 --frame-height 360 --sequence-timeout-seconds 300 --max-sequences 5
```

