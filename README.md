# EPQ

统一的无人机目标跟踪实验框架，当前支持 KCF、CSRT、TLD 和 Fusion 四种算法。

## UAV123 单序列测评

UAV123 目录结构默认约定为：

- data_seq/UAV123/<sequence_name>/
- anno/UAV123/<sequence_name>.txt

可以直接使用数据集根目录和序列名启动，例如：

```bash
python Fusion/Fusion.py --uav123-root D:/UAV123 --sequence-name bike1 --save-video
```

也支持直接把 --uav123-root 指到 data_seq/UAV123 这一层，例如：

```bash
python Fusion/Fusion.py --uav123-root D:/论文/UAV123/data_seq/UAV123 --sequence-name bike1 --save-video
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
- average_center_error
- success_rate_iou_0_5
- precision_20px

