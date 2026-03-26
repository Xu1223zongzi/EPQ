import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
from djitellopy import Tello


WINDOW_NAME = "Tello TLD Tracker"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MANUAL_SPEED = 35
MAX_YAW_SPEED = 45
MAX_UP_DOWN_SPEED = 35
MAX_FORWARD_BACK_SPEED = 30
MIN_BOX_SIZE = 20
OUTPUT_FPS = 20.0


def clamp(value, low, high):
	return max(low, min(high, int(value)))


def parse_video_source(value):
	if value is None:
		return None
	if isinstance(value, str):
		value = value.strip()
		if value.isdigit():
			return int(value)
	return value


def create_tld_tracker():
	if hasattr(cv2, "TrackerTLD_create"):
		return cv2.TrackerTLD_create()
	if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerTLD_create"):
		return cv2.legacy.TrackerTLD_create()
	raise RuntimeError("当前 OpenCV 不支持 TLD，请安装 opencv-contrib-python。")


class ExperimentRecorder:
	def __init__(self, output_dir, save_video):
		run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
		self.run_dir = Path(output_dir) / run_name
		self.run_dir.mkdir(parents=True, exist_ok=True)

		self.csv_path = self.run_dir / "telemetry.csv"
		self.summary_path = self.run_dir / "summary.json"
		self.video_path = self.run_dir / "overlay.mp4" if save_video else None
		self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8-sig")
		self.writer = csv.DictWriter(
			self.csv_file,
			fieldnames=[
				"timestamp",
				"frame_index",
				"source_mode",
				"event",
				"tracking_active",
				"tracker_ok",
				"bbox_x",
				"bbox_y",
				"bbox_w",
				"bbox_h",
				"error_x",
				"error_y",
				"area_ratio",
				"planned_lr",
				"planned_fb",
				"planned_ud",
				"planned_yaw",
				"sent_lr",
				"sent_fb",
				"sent_ud",
				"sent_yaw",
				"flying",
				"battery_level",
				"processing_ms",
			],
		)
		self.writer.writeheader()
		self.video_writer = None
		self.total_frames = 0
		self.tracking_frames = 0
		self.tracker_update_frames = 0
		self.tracker_success_frames = 0
		self.frames_with_bbox = 0
		self.total_processing_ms = 0.0
		self.max_processing_ms = 0.0
		self.total_abs_error_x = 0.0
		self.total_abs_error_y = 0.0
		self.max_abs_error_x = 0.0
		self.max_abs_error_y = 0.0
		self.total_area_ratio = 0.0
		self.area_ratio_count = 0
		self.command_activity_frames = 0
		self.event_counts = {}

	def log_frame(
		self,
		frame_index,
		source_mode,
		event,
		tracking_active,
		tracker_ok,
		bbox,
		metrics,
		planned_command,
		sent_command,
		flying,
		battery_level,
		processing_ms,
	):
		bbox = bbox or (None, None, None, None)
		metrics = metrics or {}
		self.total_frames += 1
		self.total_processing_ms += processing_ms
		self.max_processing_ms = max(self.max_processing_ms, processing_ms)

		if tracking_active:
			self.tracking_frames += 1

		if tracker_ok is not None:
			self.tracker_update_frames += 1
			if tracker_ok:
				self.tracker_success_frames += 1

		if bbox[0] is not None:
			self.frames_with_bbox += 1

		error_x = metrics.get("error_x")
		if error_x is not None:
			abs_error_x = abs(float(error_x))
			self.total_abs_error_x += abs_error_x
			self.max_abs_error_x = max(self.max_abs_error_x, abs_error_x)

		error_y = metrics.get("error_y")
		if error_y is not None:
			abs_error_y = abs(float(error_y))
			self.total_abs_error_y += abs_error_y
			self.max_abs_error_y = max(self.max_abs_error_y, abs_error_y)

		area_ratio = metrics.get("area_ratio")
		if area_ratio is not None:
			self.total_area_ratio += float(area_ratio)
			self.area_ratio_count += 1

		if any(sent_command):
			self.command_activity_frames += 1

		if event:
			for event_name in event.split(","):
				if not event_name:
					continue
				self.event_counts[event_name] = self.event_counts.get(event_name, 0) + 1

		self.writer.writerow(
			{
				"timestamp": datetime.now().isoformat(timespec="milliseconds"),
				"frame_index": frame_index,
				"source_mode": source_mode,
				"event": event,
				"tracking_active": tracking_active,
				"tracker_ok": tracker_ok,
				"bbox_x": bbox[0],
				"bbox_y": bbox[1],
				"bbox_w": bbox[2],
				"bbox_h": bbox[3],
				"error_x": metrics.get("error_x"),
				"error_y": metrics.get("error_y"),
				"area_ratio": metrics.get("area_ratio"),
				"planned_lr": planned_command[0],
				"planned_fb": planned_command[1],
				"planned_ud": planned_command[2],
				"planned_yaw": planned_command[3],
				"sent_lr": sent_command[0],
				"sent_fb": sent_command[1],
				"sent_ud": sent_command[2],
				"sent_yaw": sent_command[3],
				"flying": flying,
				"battery_level": battery_level,
				"processing_ms": round(processing_ms, 3),
			}
		)
		self.csv_file.flush()

	def build_summary(self):
		average_processing_ms = self.total_processing_ms / self.total_frames if self.total_frames else 0.0
		average_abs_error_x = self.total_abs_error_x / self.frames_with_bbox if self.frames_with_bbox else 0.0
		average_abs_error_y = self.total_abs_error_y / self.frames_with_bbox if self.frames_with_bbox else 0.0
		average_area_ratio = self.total_area_ratio / self.area_ratio_count if self.area_ratio_count else 0.0
		tracking_ratio = self.tracking_frames / self.total_frames if self.total_frames else 0.0
		command_activity_ratio = self.command_activity_frames / self.total_frames if self.total_frames else 0.0
		tracker_success_ratio = (
			self.tracker_success_frames / self.tracker_update_frames if self.tracker_update_frames else 0.0
		)

		return {
			"run_dir": str(self.run_dir),
			"csv_path": str(self.csv_path),
			"video_path": str(self.video_path) if self.video_path is not None else None,
			"total_frames": self.total_frames,
			"frames_with_bbox": self.frames_with_bbox,
			"tracking_frames": self.tracking_frames,
			"tracker_update_frames": self.tracker_update_frames,
			"tracker_success_frames": self.tracker_success_frames,
			"tracking_ratio": round(tracking_ratio, 6),
			"tracker_success_ratio": round(tracker_success_ratio, 6),
			"command_activity_frames": self.command_activity_frames,
			"command_activity_ratio": round(command_activity_ratio, 6),
			"average_processing_ms": round(average_processing_ms, 3),
			"max_processing_ms": round(self.max_processing_ms, 3),
			"average_abs_error_x": round(average_abs_error_x, 3),
			"average_abs_error_y": round(average_abs_error_y, 3),
			"max_abs_error_x": round(self.max_abs_error_x, 3),
			"max_abs_error_y": round(self.max_abs_error_y, 3),
			"average_area_ratio": round(average_area_ratio, 6),
			"event_counts": self.event_counts,
		}

	def write_frame(self, frame):
		if self.video_path is None or frame is None:
			return

		if self.video_writer is None:
			frame_height, frame_width = frame.shape[:2]
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			self.video_writer = cv2.VideoWriter(
				str(self.video_path), fourcc, OUTPUT_FPS, (frame_width, frame_height)
			)

		self.video_writer.write(frame)

	def close(self):
		if self.video_writer is not None:
			self.video_writer.release()
			self.video_writer = None
		self.csv_file.close()
		summary = self.build_summary()
		self.summary_path.write_text(
			json.dumps(summary, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)
		return summary


class TelloTLDTracker:
	def __init__(
		self,
		video_source=None,
		output_dir="experiment_runs",
		save_video=False,
		no_fly=False,
		target_area_ratio=1.0,
		target_y_offset=0.0,
	):
		self.video_source = parse_video_source(video_source)
		self.use_tello = self.video_source is None
		self.allow_flight = self.use_tello and not no_fly
		self.target_area_ratio = float(target_area_ratio)
		self.target_y_offset = float(target_y_offset)

		self.tello = Tello() if self.use_tello else None
		self.capture = None
		self.frame_reader = None
		self.current_frame = None
		self.display_frame = None
		self.battery_level = -1
		self.last_battery_update = 0.0
		self.stream_finished = False
		self.video_paused = False
		self.video_pause_reason = ""
		self.video_frame_interval = 0.0

		self.tracker = None
		self.tracking_bbox = None
		self.reference_area = None
		self.tracking_active = False

		self.drag_start = None
		self.drag_current = None
		self.dragging = False

		self.flying = False
		self.last_rc = (0, 0, 0, 0)
		self.frame_index = 0
		self.target_lost_count = 0
		self.recorder = ExperimentRecorder(output_dir=output_dir, save_video=save_video)
		self.summary = None

		print(f"实验输出目录: {self.recorder.run_dir}")
		print(f"CSV 日志: {self.recorder.csv_path}")
		print(f"统计摘要: {self.recorder.summary_path}")
		if self.recorder.video_path is not None:
			print(f"叠加视频: {self.recorder.video_path}")

	def connect(self):
		if self.use_tello:
			self.tello.connect()
			self.battery_level = self.tello.get_battery()
			self.last_battery_update = time.time()
			print(f"电池电量: {self.battery_level}%")
			self.tello.streamon()
			time.sleep(2)
			self.frame_reader = self.tello.get_frame_read()
			if not self.allow_flight:
				print("当前为观测模式：不会向真机发送起飞或 RC 指令。")
			return

		self.capture = cv2.VideoCapture(self.video_source)
		if not self.capture.isOpened():
			raise RuntimeError(f"无法打开视频源: {self.video_source}")
		fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
		if fps > 1e-6:
			self.video_frame_interval = 1.0 / fps
		self.video_paused = True
		self.video_pause_reason = "waiting_for_selection"
		if not self._read_next_video_frame():
			raise RuntimeError(f"视频源可打开，但无法读取首帧: {self.video_source}")
		print(f"已进入离线实验模式，视频源: {self.video_source}")
		if self.video_frame_interval > 0:
			print(f"检测到视频帧率: {1.0 / self.video_frame_interval:.2f} FPS")
		print("视频已暂停在首帧，请先框选目标；框选成功后将自动开始播放。")

	def _set_frame(self, frame, count_frame=True):
		self.current_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
		self.display_frame = self.current_frame.copy()
		if count_frame:
			self.frame_index += 1

	def _read_next_video_frame(self):
		ok, frame = self.capture.read()
		if not ok or frame is None:
			self.stream_finished = True
			return False

		self._set_frame(frame)
		return True

	def get_battery_level(self):
		if not self.use_tello:
			return -1

		now = time.time()
		if now - self.last_battery_update >= 5:
			try:
				self.battery_level = self.tello.get_battery()
				self.last_battery_update = now
			except Exception:
				pass
		return self.battery_level

	def get_source_mode(self):
		return "tello" if self.use_tello else "video"

	def get_battery_text(self):
		battery_level = self.get_battery_level()
		return "N/A" if battery_level < 0 else f"{battery_level}%"

	def get_tracking_metrics(self, bbox):
		if bbox is None:
			return {}

		x, y, w, h = bbox
		target_center_x = x + w / 2
		target_center_y = y + h / 2
		frame_center_x = FRAME_WIDTH / 2
		desired_center_y = FRAME_HEIGHT / 2 + self.target_y_offset

		return {
			"error_x": round(target_center_x - frame_center_x, 3),
			"error_y": round(target_center_y - desired_center_y, 3),
			"area_ratio": round((w * h) / self.reference_area, 6) if self.reference_area else 1.0,
		}

	def reset_tracking(self, stop_motion=False):
		self.tracker = None
		self.tracking_bbox = None
		self.reference_area = None
		self.tracking_active = False
		if stop_motion and self.flying:
			self.send_rc(0, 0, 0, 0)

	def send_rc(self, lr, fb, ud, yaw):
		command = (int(lr), int(fb), int(ud), int(yaw))
		if command == self.last_rc:
			return
		if self.use_tello and self.allow_flight:
			self.tello.send_rc_control(*command)
		self.last_rc = command

	def takeoff(self):
		if self.flying:
			return
		if self.use_tello and self.allow_flight:
			self.tello.takeoff()
		elif self.use_tello:
			print("观测模式：已模拟起飞状态，但未向真机发送起飞命令。")
		else:
			print("离线实验模式：已进入模拟飞行状态。")
		self.flying = True
		self.send_rc(0, 0, 0, 0)
		print("起飞成功")

	def land(self):
		if not self.flying:
			return
		self.send_rc(0, 0, 0, 0)
		if self.use_tello and self.allow_flight:
			self.tello.land()
		elif self.use_tello:
			print("观测模式：已模拟降落状态，但未向真机发送降落命令。")
		else:
			print("离线实验模式：已结束模拟飞行状态。")
		self.flying = False
		self.reset_tracking(stop_motion=False)
		print("降落成功")

	def start_tracking(self, bbox):
		if self.current_frame is None:
			return False

		x, y, w, h = bbox
		if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
			print("目标框太小，请重新框选。")
			return False

		tracker = create_tld_tracker()
		ok = tracker.init(self.current_frame.copy(), bbox)
		if ok is False:
			print("TLD 初始化失败，请重新框选。")
			return False

		self.tracker = tracker
		self.tracking_bbox = bbox
		self.reference_area = w * h
		self.tracking_active = True
		if not self.use_tello and self.video_pause_reason == "waiting_for_selection":
			self.video_paused = False
			self.video_pause_reason = ""
			print("已完成框选，开始播放视频。")
		print("目标已锁定，开始自动跟随。")
		return True

	def mouse_callback(self, event, x, y, flags, param):
		if self.current_frame is None:
			return

		if event == cv2.EVENT_LBUTTONDOWN:
			self.dragging = True
			self.drag_start = (x, y)
			self.drag_current = (x, y)
		elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
			self.drag_current = (x, y)
		elif event == cv2.EVENT_LBUTTONUP and self.dragging:
			self.dragging = False
			self.drag_current = (x, y)

			x0, y0 = self.drag_start
			x1, y1 = self.drag_current
			left = min(x0, x1)
			top = min(y0, y1)
			width = abs(x1 - x0)
			height = abs(y1 - y0)

			self.drag_start = None
			self.drag_current = None
			self.reset_tracking(stop_motion=True)
			self.start_tracking((left, top, width, height))

	def read_frame(self):
		if self.use_tello:
			frame = self.frame_reader.frame
			if frame is None:
				return False
			self._set_frame(frame)
			return True

		if self.video_paused:
			return self.current_frame is not None

		return self._read_next_video_frame()

	def compute_auto_rc(self, bbox):
		metrics = self.get_tracking_metrics(bbox)
		error_x = metrics.get("error_x", 0.0)
		error_y = metrics.get("error_y", 0.0)
		area_ratio = metrics.get("area_ratio", 1.0)
		area_error = self.target_area_ratio - area_ratio
		area_deadband = max(0.08, self.target_area_ratio * 0.12)

		yaw = 0
		ud = 0
		fb = 0

		if abs(error_x) > 35:
			yaw = clamp(error_x * 0.22, -MAX_YAW_SPEED, MAX_YAW_SPEED)

		if abs(error_y) > 30:
			ud = clamp(-error_y * 0.20, -MAX_UP_DOWN_SPEED, MAX_UP_DOWN_SPEED)

		if area_error > area_deadband:
			fb = clamp(area_error * 90, 10, MAX_FORWARD_BACK_SPEED)
		elif area_error < -area_deadband:
			fb = clamp(area_error * 90, -MAX_FORWARD_BACK_SPEED, -10)

		return (0, fb, ud, yaw), metrics

	def draw_overlay(self, manual_mode=False):
		cv2.line(self.display_frame, (FRAME_WIDTH // 2 - 15, FRAME_HEIGHT // 2),
				 (FRAME_WIDTH // 2 + 15, FRAME_HEIGHT // 2), (255, 255, 0), 1)
		cv2.line(self.display_frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2 - 15),
				 (FRAME_WIDTH // 2, FRAME_HEIGHT // 2 + 15), (255, 255, 0), 1)
		desired_center_y = int(FRAME_HEIGHT / 2 + self.target_y_offset)
		cv2.line(self.display_frame, (0, desired_center_y), (FRAME_WIDTH, desired_center_y), (0, 165, 255), 1)

		status = "TRACKING" if self.tracking_active else "IDLE"
		if manual_mode:
			status = "MANUAL"
		if not self.use_tello and self.video_paused:
			status = "PAUSED"

		lines = [
			f"Battery: {self.get_battery_text()}",
			f"Flight: {'ON' if self.flying else 'OFF'}",
			f"Mode: {status}",
			f"Source: {self.get_source_mode()}  Frame: {self.frame_index}",
			f"Target distance ratio: {self.target_area_ratio:.2f}  y-offset: {self.target_y_offset:.0f}px",
			"t takeoff | x land | r reset | q quit",
			"i/k forward/back | j/l left/right | w/s up/down | a/d turn",
			"Drag mouse to select target",
		]

		if not self.use_tello:
			lines.append("p pause/resume")
			if self.video_pause_reason == "waiting_for_selection":
				lines.append("Video paused: select a target to begin playback")

		for index, text in enumerate(lines):
			cv2.putText(
				self.display_frame,
				text,
				(10, 24 + index * 24),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(0, 255, 255) if index < 3 else (255, 255, 255),
				2,
			)

		if self.dragging and self.drag_start and self.drag_current:
			x0, y0 = self.drag_start
			x1, y1 = self.drag_current
			cv2.rectangle(self.display_frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

		if self.tracking_bbox:
			x, y, w, h = self.tracking_bbox
			cv2.rectangle(self.display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	def run(self):
		if self.use_tello:
			print("请先连接 Tello WiFi。")
		else:
			print("当前运行在离线视频实验模式。")
		print("操作说明:")
		print("  t: 起飞")
		print("  x: 降落")
		print("  鼠标拖拽: 选择目标")
		print("  r: 清除当前目标")
		print("  q: 退出程序")

		cv2.namedWindow(WINDOW_NAME)
		cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

		while True:
			loop_started_at = time.perf_counter()
			if not self.read_frame():
				if self.stream_finished:
					print("视频输入结束。")
					break
				continue

			manual_mode = False
			rc_command = (0, 0, 0, 0)
			metrics = {}
			tracking_ok = None
			events = []

			if self.tracking_active and self.tracker is not None:
				ok, bbox = self.tracker.update(self.current_frame)
				tracking_ok = ok
				if ok:
					self.tracking_bbox = tuple(int(v) for v in bbox)
					rc_command, metrics = self.compute_auto_rc(self.tracking_bbox)
				else:
					print("目标丢失，请重新框选。")
					self.target_lost_count += 1
					events.append("target_lost")
					self.reset_tracking(stop_motion=True)

			self.draw_overlay(manual_mode=manual_mode)
			cv2.imshow(WINDOW_NAME, self.display_frame)

			key = cv2.waitKey(1) & 0xFF
			should_quit = False
			if key == ord('q'):
				events.append("quit")
				should_quit = True
			elif key == ord('p') and not self.use_tello:
				self.video_paused = not self.video_paused
				self.video_pause_reason = "manual_pause" if self.video_paused else ""
				print("视频已暂停。" if self.video_paused else "视频已继续播放。")
				events.append("pause" if self.video_paused else "resume")
			elif key == ord('t'):
				self.takeoff()
				events.append("takeoff")
			elif key == ord('x'):
				self.land()
				events.append("land")
			elif key == ord('r'):
				self.reset_tracking(stop_motion=True)
				print("已清除当前目标。")
				events.append("reset_tracking")

			manual_commands = {
				ord('j'): (-MANUAL_SPEED, 0, 0, 0),
				ord('l'): (MANUAL_SPEED, 0, 0, 0),
				ord('i'): (0, MANUAL_SPEED, 0, 0),
				ord('k'): (0, -MANUAL_SPEED, 0, 0),
				ord('w'): (0, 0, MANUAL_SPEED, 0),
				ord('s'): (0, 0, -MANUAL_SPEED, 0),
				ord('a'): (0, 0, 0, -MANUAL_SPEED),
				ord('d'): (0, 0, 0, MANUAL_SPEED),
				ord(' '): (0, 0, 0, 0),
			}

			if key in manual_commands:
				manual_mode = True
				rc_command = manual_commands[key]
				metrics = self.get_tracking_metrics(self.tracking_bbox)
				events.append(f"manual_{chr(key)}")
				self.draw_overlay(manual_mode=manual_mode)
				cv2.imshow(WINDOW_NAME, self.display_frame)

			if self.flying:
				self.send_rc(*rc_command)

			processing_ms = (time.perf_counter() - loop_started_at) * 1000
			self.recorder.log_frame(
				frame_index=self.frame_index,
				source_mode=self.get_source_mode(),
				event=",".join(events),
				tracking_active=self.tracking_active,
				tracker_ok=tracking_ok,
				bbox=self.tracking_bbox,
				metrics=metrics,
				planned_command=rc_command,
				sent_command=self.last_rc,
				flying=self.flying,
				battery_level=self.get_battery_level(),
				processing_ms=processing_ms,
			)
			self.recorder.write_frame(self.display_frame)

			if not self.use_tello and not self.video_paused and self.video_frame_interval > 0:
				remaining = self.video_frame_interval - (time.perf_counter() - loop_started_at)
				if remaining > 0:
					time.sleep(remaining)

			if should_quit:
				break

	def close(self):
		try:
			if self.flying:
				self.send_rc(0, 0, 0, 0)
				if self.use_tello and self.allow_flight:
					self.tello.land()
		finally:
			try:
				if self.use_tello:
					self.tello.streamoff()
			except Exception:
				pass
			try:
				if self.use_tello:
					self.tello.end()
			except Exception:
				pass
			if self.capture is not None:
				self.capture.release()
			self.summary = self.recorder.close()
			cv2.destroyAllWindows()

		print(f"总帧数: {self.frame_index}")
		print(f"目标丢失次数: {self.target_lost_count}")
		print(f"实验结果目录: {self.recorder.run_dir}")
		print(f"统计摘要文件: {self.recorder.summary_path}")
		if self.summary is not None:
			print(f"平均处理耗时: {self.summary['average_processing_ms']} ms")
			print(f"最大处理耗时: {self.summary['max_processing_ms']} ms")
			print(f"平均横向误差: {self.summary['average_abs_error_x']} px")
			print(f"平均纵向误差: {self.summary['average_abs_error_y']} px")
			print(f"跟踪更新成功率: {self.summary['tracker_success_ratio']:.2%}")


def build_argument_parser():
	parser = argparse.ArgumentParser(description="Tello TLD 跟踪实验脚本")
	parser.add_argument(
		"--video-source",
		default=None,
		help="离线视频源，可填写视频文件路径或摄像头索引；留空则连接 Tello 真机。",
	)
	parser.add_argument(
		"--output-dir",
		default="experiment_runs",
		help="实验结果输出目录，默认保存在 experiment_runs 下。",
	)
	parser.add_argument(
		"--save-video",
		action="store_true",
		help="保存叠加了跟踪框和状态信息的输出视频。",
	)
	parser.add_argument(
		"--no-fly",
		action="store_true",
		help="连接真机但不发送起飞和 RC 指令，仅做观测和日志采集。",
	)
	parser.add_argument(
		"--target-area-ratio",
		type=float,
		default=1.0,
		help="相对初始框选面积的目标比例。1.0 表示保持框选时的距离，>1 更近，<1 更远。",
	)
	parser.add_argument(
		"--target-y-offset",
		type=float,
		default=0.0,
		help="目标相对画面中心的期望垂直偏移，单位像素。负值更高，正值更低。",
	)
	return parser


def main():
	args = build_argument_parser().parse_args()
	app = TelloTLDTracker(
		video_source=args.video_source,
		output_dir=args.output_dir,
		save_video=args.save_video,
		no_fly=args.no_fly,
		target_area_ratio=args.target_area_ratio,
		target_y_offset=args.target_y_offset,
	)
	try:
		app.connect()
		app.run()
	finally:
		app.close()
		print("程序已退出")


if __name__ == "__main__":
	main()
