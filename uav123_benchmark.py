import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tracker_framework import (
    CSRTTrackerAdapter,
    KCFTrackerAdapter,
    KCFWithTLDRelocalizationAdapter,
    TLDTrackerAdapter,
    bbox_distance,
    bbox_iou,
    load_annotation_file,
    load_image_with_unicode_path,
    load_sequence_images,
    normalize_bbox_for_tracker,
    normalize_uav123_root,
    scale_bbox_to_frame,
)


ALGORITHM_FACTORIES = {
    "KCF": KCFTrackerAdapter,
    "CSRT": CSRTTrackerAdapter,
    "TLD": TLDTrackerAdapter,
    "KCF_TLD": KCFWithTLDRelocalizationAdapter,
}


OVERLAP_CURVE_THRESHOLDS = [index / 100.0 for index in range(0, 101)]
CENTER_PRECISION_THRESHOLDS = list(range(0, 51))
NORMALIZED_PRECISION_THRESHOLDS = [index / 200.0 for index in range(0, 51)]

METRIC_GROUPS = [
    {
        "key": "overlap",
        "title": "Overlap Quality",
        "plot_file": "overlap_metrics.png",
        "metrics": [
            "average_iou",
            "overlap_auc",
            "overlap_recall_iou_0_5",
            "overlap_recall_iou_0_75",
        ],
    },
    {
        "key": "localization",
        "title": "Localization Accuracy",
        "plot_file": "localization_metrics.png",
        "metrics": [
            "average_center_error",
            "median_center_error",
            "center_precision_20px",
            "normalized_precision_0_05",
        ],
    },
    {
        "key": "stability",
        "title": "Tracking Stability",
        "plot_file": "stability_metrics.png",
        "metrics": [
            "tracking_availability",
            "failure_frame_ratio",
            "longest_failure_streak_ratio",
        ],
    },
    {
        "key": "efficiency",
        "title": "Computational Efficiency",
        "plot_file": "efficiency_metrics.png",
        "metrics": [
            "average_fps",
        ],
    },
]

METRIC_SPECS = {
    "average_iou": {
        "label": "Mean IoU",
        "direction": "higher",
        "formatter": "float4",
        "axis_label": "Mean IoU",
    },
    "overlap_auc": {
        "label": "Overlap AUC",
        "direction": "higher",
        "formatter": "float4",
        "axis_label": "AUC",
    },
    "overlap_recall_iou_0_5": {
        "label": "IoU >= 0.50 Rate",
        "direction": "higher",
        "formatter": "percent",
        "axis_label": "Rate",
    },
    "overlap_recall_iou_0_75": {
        "label": "IoU >= 0.75 Rate",
        "direction": "higher",
        "formatter": "percent",
        "axis_label": "Rate",
    },
    "average_center_error": {
        "label": "Mean Center Error (px)",
        "direction": "lower",
        "formatter": "float3",
        "axis_label": "Pixels",
    },
    "median_center_error": {
        "label": "Median Center Error (px)",
        "direction": "lower",
        "formatter": "float3",
        "axis_label": "Pixels",
    },
    "center_precision_20px": {
        "label": "Center Error <= 20px Rate",
        "direction": "higher",
        "formatter": "percent",
        "axis_label": "Rate",
    },
    "normalized_precision_0_05": {
        "label": "Normalized Precision@0.05",
        "direction": "higher",
        "formatter": "percent",
        "axis_label": "Rate",
    },
    "tracking_availability": {
        "label": "Tracking Availability",
        "direction": "higher",
        "formatter": "percent",
        "axis_label": "Rate",
    },
    "failure_frame_ratio": {
        "label": "Failure Frame Ratio",
        "direction": "lower",
        "formatter": "percent",
        "axis_label": "Rate",
    },
    "longest_failure_streak_ratio": {
        "label": "Longest Failure Streak Ratio",
        "direction": "lower",
        "formatter": "percent",
        "axis_label": "Ratio",
    },
    "average_fps": {
        "label": "Average FPS",
        "direction": "higher",
        "formatter": "float2",
        "axis_label": "FPS",
    },
}


def iter_metric_keys():
    for group in METRIC_GROUPS:
        for metric_key in group["metrics"]:
            yield metric_key


def compute_overlap_auc(ious):
    if not ious:
        return 0.0
    curve = [sum(1 for value in ious if value >= threshold) / len(ious) for threshold in OVERLAP_CURVE_THRESHOLDS]
    return sum(curve) / len(curve)


def compute_rate(values, predicate):
    if not values:
        return 0.0
    return sum(1 for value in values if predicate(value)) / len(values)


def compute_median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def format_percent(value):
    return f"{value * 100:.2f}%"


def format_metric_value(metric_key, value):
    formatter = METRIC_SPECS[metric_key]["formatter"]
    if formatter == "percent":
        return format_percent(value)
    if formatter == "float2":
        return f"{value:.2f}"
    if formatter == "float3":
        return f"{value:.3f}"
    return f"{value:.4f}"


def format_direction(direction):
    return "Higher is better" if direction == "higher" else "Lower is better"


def is_rate_metric(metric_key):
    return METRIC_SPECS[metric_key]["formatter"] == "percent"


def build_metric_matrix_rows(aggregate_rows):
    rows = []
    for group in METRIC_GROUPS:
        for metric_key in group["metrics"]:
            row = {
                "category": group["title"],
                "metric": METRIC_SPECS[metric_key]["label"],
                "direction": format_direction(METRIC_SPECS[metric_key]["direction"]),
                "metric_key": metric_key,
            }
            for aggregate_row in aggregate_rows:
                row[aggregate_row["algorithm"]] = aggregate_row[metric_key]
            rows.append(row)
    return rows


def create_benchmark_adapter(algorithm_name, args):
    if algorithm_name == "KCF_TLD":
        return KCFWithTLDRelocalizationAdapter(
            probe_interval_frames=args.kcf_tld_probe_interval_frames,
            relocalize_iou_floor=args.kcf_tld_relocalize_iou_floor,
            recovery_update_interval_frames=args.kcf_tld_recovery_interval_frames,
            recovery_confirmation_frames=args.kcf_tld_recovery_confirmation_frames,
            recovery_consistency_iou_floor=args.kcf_tld_recovery_consistency_iou_floor,
            tld_activation_failures=args.kcf_tld_activation_failures,
            unstable_continuity_iou_floor=args.kcf_tld_unstable_continuity_iou_floor,
            unstable_area_ratio_floor=args.kcf_tld_unstable_area_ratio_floor,
            unstable_area_ratio_ceiling=args.kcf_tld_unstable_area_ratio_ceiling,
            unstable_center_shift_ratio_ceiling=2.5,
            recovery_max_frames=args.kcf_tld_recovery_max_frames,
        )
    return ALGORITHM_FACTORIES[algorithm_name]()


def resolve_dataset_dirs(uav123_root):
    root = normalize_uav123_root(uav123_root)
    sequence_root = root / "data_seq" / "UAV123"
    annotation_root = root / "anno" / "UAV123"
    if not sequence_root.is_dir():
        raise RuntimeError(f"图像序列根目录不存在: {sequence_root}")
    if not annotation_root.is_dir():
        raise RuntimeError(f"标注根目录不存在: {annotation_root}")
    return root, sequence_root, annotation_root


def build_sequence_specs(sequence_root, annotation_root, requested_sequences=None, max_sequences=None):
    requested = set(requested_sequences or [])
    specs = []

    for annotation_path in sorted(annotation_root.glob("*.txt")):
        annotation_name = annotation_path.stem
        candidate_dirs = [sequence_root / annotation_name]
        if "_" in annotation_name:
            candidate_dirs.append(sequence_root / annotation_name.split("_")[0])

        sequence_dir = next((path for path in candidate_dirs if path.is_dir()), None)
        if sequence_dir is None:
            continue

        sequence_key = annotation_name
        if requested and sequence_key not in requested and sequence_dir.name not in requested:
            continue

        specs.append(
            {
                "sequence_key": sequence_key,
                "sequence_dir_name": sequence_dir.name,
                "sequence_dir": sequence_dir,
                "annotation_file": annotation_path,
            }
        )

    if max_sequences is not None:
        specs = specs[: max(0, int(max_sequences))]

    if requested and not specs:
        raise RuntimeError("未找到任何匹配的 UAV123 序列，请检查 sequence 名称。")
    if not specs:
        raise RuntimeError("未发现可用的 UAV123 序列/标注对。")
    return specs


def evaluate_sequence(
    adapter,
    sequence_spec,
    max_frames=None,
    frame_step=1,
    progress_every=100,
    sequence_timeout_seconds=None,
    frame_width=480,
    frame_height=360,
):
    images = load_sequence_images(sequence_spec["sequence_dir"])
    annotations = load_annotation_file(sequence_spec["annotation_file"])
    usable_length = min(len(images), len(annotations))
    if usable_length < 2:
        raise RuntimeError(f"序列可用帧数不足: {sequence_spec['sequence_key']}")

    if max_frames is not None:
        usable_length = min(usable_length, max(2, int(max_frames)))

    frame_step = max(1, int(frame_step))
    progress_every = max(1, int(progress_every))

    images = images[:usable_length]
    annotations = annotations[:usable_length]

    if frame_step > 1:
        sampled_indices = list(range(0, usable_length, frame_step))
        if sampled_indices[-1] != usable_length - 1:
            sampled_indices.append(usable_length - 1)
        images = [images[index] for index in sampled_indices]
        annotations = [annotations[index] for index in sampled_indices]

    if len(images) < 2:
        raise RuntimeError(f"序列抽样后可用帧数不足: {sequence_spec['sequence_key']}")

    frame_width = max(64, int(frame_width))
    frame_height = max(64, int(frame_height))
    frame_size = (frame_width, frame_height)

    first_frame = load_image_with_unicode_path(images[0])
    resized_first = cv2.resize(first_frame, frame_size)
    initial_bbox = scale_bbox_to_frame(annotations[0], first_frame.shape, resized_first.shape)
    initial_bbox = normalize_bbox_for_tracker(initial_bbox, resized_first.shape)

    tracker = adapter.create_tracker()
    started_at = time.perf_counter()
    init_ok = tracker.init(resized_first, initial_bbox)
    if init_ok is False:
        raise RuntimeError(f"{adapter.algorithm_name} 无法在序列 {sequence_spec['sequence_key']} 上初始化。")

    ious = [bbox_iou(initial_bbox, initial_bbox)]
    center_errors = [0.0]
    normalized_center_errors = [0.0]
    tracked_frames = 1
    update_ok_flags = [1]
    last_bbox = initial_bbox
    tracker_source_counts = {"init": 1}
    frame_diagonal = math.hypot(frame_width, frame_height)
    longest_failure_streak = 0
    current_failure_streak = 0

    for processed_index, (frame_path, annotation) in enumerate(zip(images[1:], annotations[1:]), start=2):
        if sequence_timeout_seconds is not None and (time.perf_counter() - started_at) > sequence_timeout_seconds:
            raise TimeoutError(
                f"序列超时: {sequence_spec['sequence_key']} 已运行超过 {sequence_timeout_seconds:.1f} 秒"
            )

        frame = load_image_with_unicode_path(frame_path)
        resized = cv2.resize(frame, frame_size)
        gt_bbox = scale_bbox_to_frame(annotation, frame.shape, resized.shape)
        ok, bbox = tracker.update(resized)
        tracker_source = getattr(tracker, "last_source", adapter.algorithm_name) or adapter.algorithm_name
        tracker_source_counts[tracker_source] = tracker_source_counts.get(tracker_source, 0) + 1
        if ok:
            predicted_bbox = normalize_bbox_for_tracker(bbox, resized.shape)
            if predicted_bbox is not None:
                last_bbox = predicted_bbox
                tracked_frames += 1
                update_ok_flags.append(1)
                current_failure_streak = 0
            else:
                predicted_bbox = last_bbox
                update_ok_flags.append(0)
                current_failure_streak += 1
                longest_failure_streak = max(longest_failure_streak, current_failure_streak)
                tracker_source = f"{tracker_source}_INVALID"
                tracker_source_counts[tracker_source] = tracker_source_counts.get(tracker_source, 0) + 1
        else:
            predicted_bbox = last_bbox
            update_ok_flags.append(0)
            current_failure_streak += 1
            longest_failure_streak = max(longest_failure_streak, current_failure_streak)

        iou = bbox_iou(predicted_bbox, gt_bbox)
        center_error = bbox_distance(predicted_bbox, gt_bbox)
        ious.append(iou)
        center_errors.append(center_error)
        normalized_center_errors.append(center_error / max(frame_diagonal, 1e-9))

        if processed_index % progress_every == 0 or processed_index == len(images):
            print(
                f"[{adapter.algorithm_name}] {sequence_spec['sequence_key']} 进度 "
                f"{processed_index}/{len(images)} 当前IoU={iou:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started_at
    frame_count = len(ious)
    average_iou = sum(ious) / frame_count
    overlap_auc = compute_overlap_auc(ious)
    average_center_error = sum(center_errors) / frame_count
    median_center_error = compute_median(center_errors)
    overlap_recall_iou_0_5 = compute_rate(ious, lambda value: value >= 0.5)
    overlap_recall_iou_0_75 = compute_rate(ious, lambda value: value >= 0.75)
    center_precision_20px = compute_rate(center_errors, lambda value: value <= 20.0)
    normalized_precision_0_05 = compute_rate(normalized_center_errors, lambda value: value <= 0.05)
    tracking_availability = sum(update_ok_flags) / frame_count
    failure_frame_ratio = 1.0 - tracking_availability
    longest_failure_streak_ratio = longest_failure_streak / frame_count
    average_fps = frame_count / elapsed if elapsed > 1e-9 else 0.0

    return {
        "algorithm": adapter.algorithm_name,
        "sequence_key": sequence_spec["sequence_key"],
        "sequence_dir_name": sequence_spec["sequence_dir_name"],
        "annotation_file": str(sequence_spec["annotation_file"]),
        "frames": frame_count,
        "tracked_frames": tracked_frames,
        "average_iou": average_iou,
        "overlap_auc": overlap_auc,
        "average_center_error": average_center_error,
        "median_center_error": median_center_error,
        "overlap_recall_iou_0_5": overlap_recall_iou_0_5,
        "overlap_recall_iou_0_75": overlap_recall_iou_0_75,
        "center_precision_20px": center_precision_20px,
        "normalized_precision_0_05": normalized_precision_0_05,
        "tracking_availability": tracking_availability,
        "failure_frame_ratio": failure_frame_ratio,
        "longest_failure_streak_ratio": longest_failure_streak_ratio,
        "average_fps": average_fps,
        "ious": ious,
        "center_errors": center_errors,
        "normalized_center_errors": normalized_center_errors,
        "tracker_source_counts": tracker_source_counts,
    }


def aggregate_results(per_sequence_results):
    grouped = {}
    for result in per_sequence_results:
        grouped.setdefault(result["algorithm"], []).append(result)

    aggregate = []
    for algorithm, items in grouped.items():
        total_frames = sum(item["frames"] for item in items)
        aggregate_row = {
            "algorithm": algorithm,
            "sequence_count": len(items),
            "total_frames": total_frames,
        }
        for metric_key in iter_metric_keys():
            weighted_value = sum(item[metric_key] * item["frames"] for item in items)
            aggregate_row[metric_key] = weighted_value / total_frames if total_frames else 0.0
        aggregate.append(aggregate_row)

    aggregate.sort(key=lambda item: item["overlap_auc"], reverse=True)
    return aggregate


def write_csv(file_path, fieldnames, rows):
    with file_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown_table(aggregate_rows):
    algorithms = [row["algorithm"] for row in aggregate_rows]
    sections = ["# Benchmark Leaderboard", ""]

    meta_lines = ["| Item | Value |", "| --- | --- |"]
    for row in aggregate_rows:
        meta_lines.append(
            f"| {row['algorithm']} | Sequences={row['sequence_count']}, Frames={row['total_frames']} |"
        )
    sections.extend(meta_lines)
    sections.append("")

    for group in METRIC_GROUPS:
        sections.append(f"## {group['title']}")
        header = "| Metric | Direction | " + " | ".join(algorithms) + " |"
        align = "| --- | --- | " + " | ".join([":---:" for _ in algorithms]) + " |"
        sections.extend([header, align])
        for metric_key in group["metrics"]:
            row_values = [
                format_metric_value(metric_key, algorithm_row[metric_key])
                for algorithm_row in aggregate_rows
            ]
            sections.append(
                "| {metric_label} | {direction} | {values} |".format(
                    metric_label=METRIC_SPECS[metric_key]["label"],
                    direction=format_direction(METRIC_SPECS[metric_key]["direction"]),
                    values=" | ".join(row_values),
                )
            )
        sections.append("")

    return "\n".join(sections).strip() + "\n"


def render_table_png(aggregate_rows, output_path):
    algorithms = [row["algorithm"] for row in aggregate_rows]
    fig, axes = plt.subplots(len(METRIC_GROUPS), 1, figsize=(13.5, 2.2 + len(METRIC_GROUPS) * 2.7))
    if len(METRIC_GROUPS) == 1:
        axes = [axes]

    for axis, group in zip(axes, METRIC_GROUPS):
        axis.axis("off")
        columns = ["Metric", "Direction"] + algorithms
        cell_text = []
        for metric_key in group["metrics"]:
            row = [
                METRIC_SPECS[metric_key]["label"],
                format_direction(METRIC_SPECS[metric_key]["direction"]),
            ]
            row.extend(format_metric_value(metric_key, aggregate_row[metric_key]) for aggregate_row in aggregate_rows)
            cell_text.append(row)

        table = axis.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.25)
        axis.set_title(group["title"], fontsize=12, fontweight="bold", pad=10)
        for (row_index, col_index), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#dbeafe")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f8fafc")

    fig.suptitle("Benchmark Leaderboard", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_overlap_curve(per_sequence_results, output_path):
    grouped = {}
    for result in per_sequence_results:
        grouped.setdefault(result["algorithm"], []).extend(result["ious"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algorithm, values in grouped.items():
        curve = [sum(1 for value in values if value >= threshold) / len(values) for threshold in OVERLAP_CURVE_THRESHOLDS]
        ax.plot(OVERLAP_CURVE_THRESHOLDS, curve, linewidth=2.0, label=algorithm)

    ax.set_title("UAV123 Overlap Recall Curve")
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Overlap Recall")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_center_precision_curve(per_sequence_results, output_path):
    grouped = {}
    for result in per_sequence_results:
        grouped.setdefault(result["algorithm"], []).extend(result["center_errors"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algorithm, values in grouped.items():
        curve = [sum(1 for value in values if value <= threshold) / len(values) for threshold in CENTER_PRECISION_THRESHOLDS]
        ax.plot(CENTER_PRECISION_THRESHOLDS, curve, linewidth=2.0, label=algorithm)

    ax.set_title("UAV123 Center Precision Curve")
    ax.set_xlabel("Center Error Threshold (px)")
    ax.set_ylabel("Localization Recall")
    ax.set_xlim(0, 50)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_normalized_precision_curve(per_sequence_results, output_path):
    grouped = {}
    for result in per_sequence_results:
        grouped.setdefault(result["algorithm"], []).extend(result["normalized_center_errors"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algorithm, values in grouped.items():
        curve = [
            sum(1 for value in values if value <= threshold) / len(values)
            for threshold in NORMALIZED_PRECISION_THRESHOLDS
        ]
        ax.plot(NORMALIZED_PRECISION_THRESHOLDS, curve, linewidth=2.0, label=algorithm)

    ax.set_title("UAV123 Normalized Precision Curve")
    ax.set_xlabel("Normalized Center Error Threshold")
    ax.set_ylabel("Localization Recall")
    ax.set_xlim(0.0, NORMALIZED_PRECISION_THRESHOLDS[-1])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_metric_group_plots(aggregate_rows, output_dir):
    algorithms = [row["algorithm"] for row in aggregate_rows]
    colors = ["#2563eb", "#0f766e", "#ca8a04", "#b91c1c", "#7c3aed", "#0891b2"]

    for group in METRIC_GROUPS:
        metric_count = len(group["metrics"])
        fig, axes = plt.subplots(metric_count, 1, figsize=(9.5, 2.8 + metric_count * 2.2))
        if metric_count == 1:
            axes = [axes]

        for axis, metric_key in zip(axes, group["metrics"]):
            values = [row[metric_key] for row in aggregate_rows]
            bars = axis.bar(algorithms, values, color=colors[: len(algorithms)], alpha=0.88)
            axis.set_title(
                f"{METRIC_SPECS[metric_key]['label']} ({format_direction(METRIC_SPECS[metric_key]['direction'])})",
                fontsize=11,
            )
            axis.set_ylabel(METRIC_SPECS[metric_key]["axis_label"])
            axis.grid(True, axis="y", linestyle="--", alpha=0.3)
            if is_rate_metric(metric_key):
                axis.set_ylim(0.0, 1.0)
            for label in axis.get_xticklabels():
                label.set_rotation(0)
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    format_metric_value(metric_key, value),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.suptitle(group["title"], fontsize=13, fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(output_dir / group["plot_file"], dpi=220)
        plt.close(fig)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="UAV123 批量 benchmark 与论文风格图表生成脚本")
    parser.add_argument("--uav123-root", required=True, help="UAV123 根目录，或 data_seq/UAV123 这一层路径。")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["KCF", "CSRT", "TLD", "KCF_TLD"],
        choices=sorted(ALGORITHM_FACTORIES.keys()),
        help="要参与测评的算法列表。",
    )
    parser.add_argument(
        "--sequence-names",
        nargs="*",
        default=None,
        help="可选，只测指定序列名或标注名，例如 bike1 bird1_1 uav1_2。",
    )
    parser.add_argument("--max-sequences", type=int, default=None, help="可选，仅测前 N 个序列。")
    parser.add_argument("--max-frames", type=int, default=None, help="可选，每个序列最多评估多少帧。")
    parser.add_argument("--frame-step", type=int, default=1, help="可选，每隔多少帧采样一次进行评测；1 表示逐帧评测。")
    parser.add_argument("--progress-every", type=int, default=100, help="可选，每处理多少帧打印一次序列内进度。")
    parser.add_argument("--sequence-timeout-seconds", type=float, default=None, help="可选，单个序列最长允许运行秒数，超时后跳过。")
    parser.add_argument("--frame-width", type=int, default=480, help="评测时统一缩放后的帧宽度，默认 480。")
    parser.add_argument("--frame-height", type=int, default=360, help="评测时统一缩放后的帧高度，默认 360。")
    parser.add_argument("--kcf-tld-probe-interval-frames", type=int, default=120, help="KCF_TLD 正常跟踪时隔多少帧才用 TLD 探测一次，默认 120。")
    parser.add_argument("--kcf-tld-relocalize-iou-floor", type=float, default=0.3, help="KCF_TLD 接受 TLD 恢复候选时要求与上一框的最小 IoU，默认 0.3。")
    parser.add_argument("--kcf-tld-recovery-interval-frames", type=int, default=3, help="KCF_TLD 丢失后隔多少帧才重试一次 TLD 恢复，默认 3。")
    parser.add_argument("--kcf-tld-recovery-confirmation-frames", type=int, default=2, help="KCF_TLD 连续多少次稳定命中后才接受恢复，默认 2。")
    parser.add_argument("--kcf-tld-recovery-consistency-iou-floor", type=float, default=0.45, help="KCF_TLD 连续两次恢复候选之间要求的最小 IoU，默认 0.45。")
    parser.add_argument("--kcf-tld-activation-failures", type=int, default=2, help="KCF_TLD 在 KCF 连续失败或异常多少次后才激活 TLD，默认 2。")
    parser.add_argument("--kcf-tld-unstable-continuity-iou-floor", type=float, default=0.15, help="KCF_TLD 判定 KCF 结果明显不连续的 IoU 阈值，默认 0.15。")
    parser.add_argument("--kcf-tld-unstable-area-ratio-floor", type=float, default=0.5, help="KCF_TLD 判定 KCF 框面积异常时的最小面积比例，默认 0.5。")
    parser.add_argument("--kcf-tld-unstable-area-ratio-ceiling", type=float, default=2.0, help="KCF_TLD 判定 KCF 框面积异常时的最大面积比例，默认 2.0。")
    parser.add_argument("--kcf-tld-recovery-max-frames", type=int, default=120, help="KCF_TLD 单次恢复态最长持续多少帧，默认 120。")
    parser.add_argument("--output-dir", default="benchmark_runs", help="批量测评结果输出目录。")
    return parser


def main():
    args = build_argument_parser().parse_args()
    _, sequence_root, annotation_root = resolve_dataset_dirs(args.uav123_root)
    sequence_specs = build_sequence_specs(
        sequence_root=sequence_root,
        annotation_root=annotation_root,
        requested_sequences=args.sequence_names,
        max_sequences=args.max_sequences,
    )

    run_dir = Path(args.output_dir) / datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    per_sequence_results = []
    failures = []

    print(f"测评序列数: {len(sequence_specs)}")
    print(f"算法列表: {', '.join(args.algorithms)}")
    print(f"输出目录: {run_dir}")
    print(f"采样步长: {max(1, int(args.frame_step))}")
    print(f"评测分辨率: {max(64, int(args.frame_width))}x{max(64, int(args.frame_height))}")
    if args.sequence_timeout_seconds is not None:
        print(f"单序列超时: {args.sequence_timeout_seconds:.1f} 秒")

    for algorithm_name in args.algorithms:
        adapter = create_benchmark_adapter(algorithm_name, args)
        for sequence_spec in sequence_specs:
            print(f"[{algorithm_name}] {sequence_spec['sequence_key']} 开始", flush=True)
            try:
                result = evaluate_sequence(
                    adapter,
                    sequence_spec,
                    max_frames=args.max_frames,
                    frame_step=args.frame_step,
                    progress_every=args.progress_every,
                    sequence_timeout_seconds=args.sequence_timeout_seconds,
                    frame_width=args.frame_width,
                    frame_height=args.frame_height,
                )
                per_sequence_results.append(result)
                print(
                    f"[{algorithm_name}] {sequence_spec['sequence_key']} 完成: "
                    f"frames={result['frames']} avg_iou={result['average_iou']:.4f} fps={result['average_fps']:.2f} "
                    f"sources={json.dumps(result['tracker_source_counts'], ensure_ascii=False, sort_keys=True)}",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "algorithm": algorithm_name,
                        "sequence_key": sequence_spec["sequence_key"],
                        "error": str(exc),
                    }
                )
                print(f"[{algorithm_name}] {sequence_spec['sequence_key']} 失败: {exc}", flush=True)

    aggregate_rows = aggregate_results(per_sequence_results)
    metric_matrix_rows = build_metric_matrix_rows(aggregate_rows)

    per_sequence_csv_rows = []
    for result in per_sequence_results:
        per_sequence_csv_rows.append(
            {
                "algorithm": result["algorithm"],
                "sequence_key": result["sequence_key"],
                "sequence_dir_name": result["sequence_dir_name"],
                "frames": result["frames"],
                "tracked_frames": result["tracked_frames"],
                "average_iou": round(result["average_iou"], 6),
                "overlap_auc": round(result["overlap_auc"], 6),
                "average_center_error": round(result["average_center_error"], 6),
                "median_center_error": round(result["median_center_error"], 6),
                "overlap_recall_iou_0_5": round(result["overlap_recall_iou_0_5"], 6),
                "overlap_recall_iou_0_75": round(result["overlap_recall_iou_0_75"], 6),
                "center_precision_20px": round(result["center_precision_20px"], 6),
                "normalized_precision_0_05": round(result["normalized_precision_0_05"], 6),
                "tracking_availability": round(result["tracking_availability"], 6),
                "failure_frame_ratio": round(result["failure_frame_ratio"], 6),
                "longest_failure_streak_ratio": round(result["longest_failure_streak_ratio"], 6),
                "average_fps": round(result["average_fps"], 6),
                "annotation_file": result["annotation_file"],
            }
        )

    aggregate_csv_rows = []
    for row in aggregate_rows:
        aggregate_csv_rows.append(
            {
                "algorithm": row["algorithm"],
                "sequence_count": row["sequence_count"],
                "total_frames": row["total_frames"],
                "average_iou": round(row["average_iou"], 6),
                "overlap_auc": round(row["overlap_auc"], 6),
                "average_center_error": round(row["average_center_error"], 6),
                "median_center_error": round(row["median_center_error"], 6),
                "overlap_recall_iou_0_5": round(row["overlap_recall_iou_0_5"], 6),
                "overlap_recall_iou_0_75": round(row["overlap_recall_iou_0_75"], 6),
                "center_precision_20px": round(row["center_precision_20px"], 6),
                "normalized_precision_0_05": round(row["normalized_precision_0_05"], 6),
                "tracking_availability": round(row["tracking_availability"], 6),
                "failure_frame_ratio": round(row["failure_frame_ratio"], 6),
                "longest_failure_streak_ratio": round(row["longest_failure_streak_ratio"], 6),
                "average_fps": round(row["average_fps"], 6),
            }
        )

    write_csv(
        run_dir / "per_sequence_results.csv",
        [
            "algorithm",
            "sequence_key",
            "sequence_dir_name",
            "frames",
            "tracked_frames",
            "average_iou",
            "overlap_auc",
            "average_center_error",
            "median_center_error",
            "overlap_recall_iou_0_5",
            "overlap_recall_iou_0_75",
            "center_precision_20px",
            "normalized_precision_0_05",
            "tracking_availability",
            "failure_frame_ratio",
            "longest_failure_streak_ratio",
            "average_fps",
            "annotation_file",
        ],
        per_sequence_csv_rows,
    )
    write_csv(
        run_dir / "aggregate_results.csv",
        [
            "algorithm",
            "sequence_count",
            "total_frames",
            "average_iou",
            "overlap_auc",
            "average_center_error",
            "median_center_error",
            "overlap_recall_iou_0_5",
            "overlap_recall_iou_0_75",
            "center_precision_20px",
            "normalized_precision_0_05",
            "tracking_availability",
            "failure_frame_ratio",
            "longest_failure_streak_ratio",
            "average_fps",
        ],
        aggregate_csv_rows,
    )
    write_csv(
        run_dir / "leaderboard_metrics.csv",
        ["category", "metric", "direction", "metric_key", *[row["algorithm"] for row in aggregate_rows]],
        metric_matrix_rows,
    )

    markdown_table = build_markdown_table(aggregate_rows)
    (run_dir / "leaderboard.md").write_text(markdown_table, encoding="utf-8")
    render_table_png(aggregate_rows, run_dir / "leaderboard.png")
    render_overlap_curve(per_sequence_results, run_dir / "overlap_curve.png")
    render_center_precision_curve(per_sequence_results, run_dir / "center_precision_curve.png")
    render_normalized_precision_curve(per_sequence_results, run_dir / "normalized_precision_curve.png")
    render_metric_group_plots(aggregate_rows, run_dir)

    summary = {
        "run_dir": str(run_dir),
        "sequence_count": len(sequence_specs),
        "algorithms": args.algorithms,
        "metric_groups": METRIC_GROUPS,
        "aggregate_results": aggregate_csv_rows,
        "failures": failures,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("批量测评完成")
    print(f"总表 CSV: {run_dir / 'aggregate_results.csv'}")
    print(f"转置指标表 CSV: {run_dir / 'leaderboard_metrics.csv'}")
    print(f"论文风格 Markdown 表: {run_dir / 'leaderboard.md'}")
    print(f"论文风格 PNG 表: {run_dir / 'leaderboard.png'}")
    print(f"Overlap 曲线: {run_dir / 'overlap_curve.png'}")
    print(f"Center Precision 曲线: {run_dir / 'center_precision_curve.png'}")
    print(f"Normalized Precision 曲线: {run_dir / 'normalized_precision_curve.png'}")
    for group in METRIC_GROUPS:
        print(f"{group['title']} 图: {run_dir / group['plot_file']}")
    if failures:
        print(f"失败条目数: {len(failures)}")


if __name__ == "__main__":
    main()