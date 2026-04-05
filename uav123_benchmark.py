import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tracker_framework import (
    CSRTTrackerAdapter,
    FusionTrackerAdapter,
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
    "FUSION": FusionTrackerAdapter,
    "KCF_TLD": KCFWithTLDRelocalizationAdapter,
}


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


def evaluate_sequence(adapter, sequence_spec, max_frames=None):
    images = load_sequence_images(sequence_spec["sequence_dir"])
    annotations = load_annotation_file(sequence_spec["annotation_file"])
    usable_length = min(len(images), len(annotations))
    if usable_length < 2:
        raise RuntimeError(f"序列可用帧数不足: {sequence_spec['sequence_key']}")

    if max_frames is not None:
        usable_length = min(usable_length, max(2, int(max_frames)))

    images = images[:usable_length]
    annotations = annotations[:usable_length]

    first_frame = load_image_with_unicode_path(images[0])
    resized_first = cv2.resize(first_frame, (640, 480))
    initial_bbox = scale_bbox_to_frame(annotations[0], first_frame.shape, resized_first.shape)
    initial_bbox = normalize_bbox_for_tracker(initial_bbox, resized_first.shape)

    tracker = adapter.create_tracker()
    started_at = time.perf_counter()
    init_ok = tracker.init(resized_first, initial_bbox)
    if init_ok is False:
        raise RuntimeError(f"{adapter.algorithm_name} 无法在序列 {sequence_spec['sequence_key']} 上初始化。")

    ious = [bbox_iou(initial_bbox, initial_bbox)]
    center_errors = [0.0]
    tracked_frames = 1
    success_frames = 1
    last_bbox = initial_bbox

    for frame_path, annotation in zip(images[1:], annotations[1:]):
        frame = load_image_with_unicode_path(frame_path)
        resized = cv2.resize(frame, (640, 480))
        gt_bbox = scale_bbox_to_frame(annotation, frame.shape, resized.shape)
        ok, bbox = tracker.update(resized)
        if ok:
            predicted_bbox = normalize_bbox_for_tracker(bbox, resized.shape)
            last_bbox = predicted_bbox
            tracked_frames += 1
        else:
            predicted_bbox = last_bbox

        iou = bbox_iou(predicted_bbox, gt_bbox)
        center_error = bbox_distance(predicted_bbox, gt_bbox)
        ious.append(iou)
        center_errors.append(center_error)
        if ok:
            success_frames += 1

    elapsed = time.perf_counter() - started_at
    frame_count = len(ious)
    average_iou = sum(ious) / frame_count
    average_center_error = sum(center_errors) / frame_count
    success_iou_05 = sum(1 for value in ious if value >= 0.5) / frame_count
    precision_20px = sum(1 for value in center_errors if value <= 20.0) / frame_count
    average_fps = frame_count / elapsed if elapsed > 1e-9 else 0.0

    return {
        "algorithm": adapter.algorithm_name,
        "sequence_key": sequence_spec["sequence_key"],
        "sequence_dir_name": sequence_spec["sequence_dir_name"],
        "annotation_file": str(sequence_spec["annotation_file"]),
        "frames": frame_count,
        "tracked_frames": tracked_frames,
        "success_frames": success_frames,
        "average_iou": average_iou,
        "average_center_error": average_center_error,
        "success_rate_iou_0_5": success_iou_05,
        "precision_20px": precision_20px,
        "average_fps": average_fps,
        "ious": ious,
        "center_errors": center_errors,
    }


def aggregate_results(per_sequence_results):
    grouped = {}
    for result in per_sequence_results:
        grouped.setdefault(result["algorithm"], []).append(result)

    aggregate = []
    for algorithm, items in grouped.items():
        total_frames = sum(item["frames"] for item in items)
        weighted_iou = sum(item["average_iou"] * item["frames"] for item in items)
        weighted_center_error = sum(item["average_center_error"] * item["frames"] for item in items)
        weighted_success = sum(item["success_rate_iou_0_5"] * item["frames"] for item in items)
        weighted_precision = sum(item["precision_20px"] * item["frames"] for item in items)
        weighted_fps = sum(item["average_fps"] * item["frames"] for item in items)

        aggregate.append(
            {
                "algorithm": algorithm,
                "sequence_count": len(items),
                "total_frames": total_frames,
                "average_iou": weighted_iou / total_frames if total_frames else 0.0,
                "average_center_error": weighted_center_error / total_frames if total_frames else 0.0,
                "success_rate_iou_0_5": weighted_success / total_frames if total_frames else 0.0,
                "precision_20px": weighted_precision / total_frames if total_frames else 0.0,
                "average_fps": weighted_fps / total_frames if total_frames else 0.0,
            }
        )

    aggregate.sort(key=lambda item: item["average_iou"], reverse=True)
    return aggregate


def write_csv(file_path, fieldnames, rows):
    with file_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_percent(value):
    return f"{value * 100:.2f}"


def build_markdown_table(aggregate_rows):
    header = [
        "| Algorithm | Avg IoU | Success@0.5 (%) | Precision@20px (%) | Avg Center Error | Avg FPS | Sequences |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = []
    for row in aggregate_rows:
        body.append(
            "| {algorithm} | {average_iou:.4f} | {success} | {precision} | {average_center_error:.3f} | {average_fps:.2f} | {sequence_count} |".format(
                algorithm=row["algorithm"],
                average_iou=row["average_iou"],
                success=format_percent(row["success_rate_iou_0_5"]),
                precision=format_percent(row["precision_20px"]),
                average_center_error=row["average_center_error"],
                average_fps=row["average_fps"],
                sequence_count=row["sequence_count"],
            )
        )
    return "\n".join(header + body) + "\n"


def render_table_png(aggregate_rows, output_path):
    columns = [
        "Algorithm",
        "Avg IoU",
        "Success@0.5 (%)",
        "Precision@20px (%)",
        "Avg Center Error",
        "Avg FPS",
        "Seqs",
    ]
    cell_text = []
    for row in aggregate_rows:
        cell_text.append(
            [
                row["algorithm"],
                f"{row['average_iou']:.4f}",
                format_percent(row["success_rate_iou_0_5"]),
                format_percent(row["precision_20px"]),
                f"{row['average_center_error']:.3f}",
                f"{row['average_fps']:.2f}",
                str(row["sequence_count"]),
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 0.8 + max(1, len(cell_text)) * 0.55))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (row_index, col_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbeafe")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f8fafc")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_success_plot(per_sequence_results, output_path):
    thresholds = [index / 100.0 for index in range(0, 101)]
    grouped = {}
    for result in per_sequence_results:
        grouped.setdefault(result["algorithm"], []).extend(result["ious"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algorithm, values in grouped.items():
        curve = [sum(1 for value in values if value >= threshold) / len(values) for threshold in thresholds]
        ax.plot(thresholds, curve, linewidth=2.0, label=algorithm)

    ax.set_title("UAV123 Success Plot")
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("Success Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_precision_plot(per_sequence_results, output_path):
    thresholds = list(range(0, 51))
    grouped = {}
    for result in per_sequence_results:
        grouped.setdefault(result["algorithm"], []).extend(result["center_errors"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for algorithm, values in grouped.items():
        curve = [sum(1 for value in values if value <= threshold) / len(values) for threshold in thresholds]
        ax.plot(thresholds, curve, linewidth=2.0, label=algorithm)

    ax.set_title("UAV123 Precision Plot")
    ax.set_xlabel("Center Error Threshold (px)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 50)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="UAV123 批量 benchmark 与论文风格图表生成脚本")
    parser.add_argument("--uav123-root", required=True, help="UAV123 根目录，或 data_seq/UAV123 这一层路径。")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["KCF", "CSRT", "TLD", "FUSION", "KCF_TLD"],
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

    for algorithm_name in args.algorithms:
        adapter = ALGORITHM_FACTORIES[algorithm_name]()
        for sequence_spec in sequence_specs:
            print(f"[{algorithm_name}] {sequence_spec['sequence_key']} 开始")
            try:
                result = evaluate_sequence(adapter, sequence_spec, max_frames=args.max_frames)
                per_sequence_results.append(result)
            except Exception as exc:
                failures.append(
                    {
                        "algorithm": algorithm_name,
                        "sequence_key": sequence_spec["sequence_key"],
                        "error": str(exc),
                    }
                )
                print(f"[{algorithm_name}] {sequence_spec['sequence_key']} 失败: {exc}")

    aggregate_rows = aggregate_results(per_sequence_results)

    per_sequence_csv_rows = []
    for result in per_sequence_results:
        per_sequence_csv_rows.append(
            {
                "algorithm": result["algorithm"],
                "sequence_key": result["sequence_key"],
                "sequence_dir_name": result["sequence_dir_name"],
                "frames": result["frames"],
                "tracked_frames": result["tracked_frames"],
                "success_frames": result["success_frames"],
                "average_iou": round(result["average_iou"], 6),
                "average_center_error": round(result["average_center_error"], 6),
                "success_rate_iou_0_5": round(result["success_rate_iou_0_5"], 6),
                "precision_20px": round(result["precision_20px"], 6),
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
                "average_center_error": round(row["average_center_error"], 6),
                "success_rate_iou_0_5": round(row["success_rate_iou_0_5"], 6),
                "precision_20px": round(row["precision_20px"], 6),
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
            "success_frames",
            "average_iou",
            "average_center_error",
            "success_rate_iou_0_5",
            "precision_20px",
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
            "average_center_error",
            "success_rate_iou_0_5",
            "precision_20px",
            "average_fps",
        ],
        aggregate_csv_rows,
    )

    markdown_table = build_markdown_table(aggregate_rows)
    (run_dir / "leaderboard.md").write_text(markdown_table, encoding="utf-8")
    render_table_png(aggregate_rows, run_dir / "leaderboard.png")
    render_success_plot(per_sequence_results, run_dir / "success_plot.png")
    render_precision_plot(per_sequence_results, run_dir / "precision_plot.png")

    summary = {
        "run_dir": str(run_dir),
        "sequence_count": len(sequence_specs),
        "algorithms": args.algorithms,
        "aggregate_results": aggregate_csv_rows,
        "failures": failures,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("批量测评完成")
    print(f"总表 CSV: {run_dir / 'aggregate_results.csv'}")
    print(f"论文风格 Markdown 表: {run_dir / 'leaderboard.md'}")
    print(f"论文风格 PNG 表: {run_dir / 'leaderboard.png'}")
    print(f"Success 曲线: {run_dir / 'success_plot.png'}")
    print(f"Precision 曲线: {run_dir / 'precision_plot.png'}")
    if failures:
        print(f"失败条目数: {len(failures)}")


if __name__ == "__main__":
    main()