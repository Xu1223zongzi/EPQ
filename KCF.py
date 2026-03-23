import time

import cv2
from djitellopy import Tello


WINDOW_NAME = "Tello KCF Tracker"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MANUAL_SPEED = 35
MAX_YAW_SPEED = 45
MAX_UP_DOWN_SPEED = 35
MAX_FORWARD_BACK_SPEED = 30
MIN_BOX_SIZE = 20


def clamp(value, low, high):
    return max(low, min(high, int(value)))


def create_kcf_tracker():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError("当前 OpenCV 不支持 KCF，请安装 opencv-contrib-python。")


class TelloKCFTracker:
    def __init__(self):
        self.tello = Tello()
        self.frame_reader = None
        self.current_frame = None
        self.display_frame = None
        self.battery_level = -1
        self.last_battery_update = 0.0

        self.tracker = None
        self.tracking_bbox = None
        self.reference_area = None
        self.tracking_active = False

        self.drag_start = None
        self.drag_current = None
        self.dragging = False

        self.flying = False
        self.last_rc = (0, 0, 0, 0)

    def connect(self):
        self.tello.connect()
        self.battery_level = self.tello.get_battery()
        self.last_battery_update = time.time()
        print(f"电池电量: {self.battery_level}%")
        self.tello.streamon()
        time.sleep(2)
        self.frame_reader = self.tello.get_frame_read()

    def get_battery_level(self):
        now = time.time()
        if now - self.last_battery_update >= 5:
            try:
                self.battery_level = self.tello.get_battery()
                self.last_battery_update = now
            except Exception:
                pass
        return self.battery_level

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
        self.tello.send_rc_control(*command)
        self.last_rc = command

    def takeoff(self):
        if self.flying:
            return
        self.tello.takeoff()
        self.flying = True
        self.send_rc(0, 0, 0, 0)
        print("起飞成功")

    def land(self):
        if not self.flying:
            return
        self.send_rc(0, 0, 0, 0)
        self.tello.land()
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

        tracker = create_kcf_tracker()
        ok = tracker.init(self.current_frame.copy(), bbox)
        if ok is False:
            print("KCF 初始化失败，请重新框选。")
            return False

        self.tracker = tracker
        self.tracking_bbox = bbox
        self.reference_area = w * h
        self.tracking_active = True
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
        frame = self.frame_reader.frame
        if frame is None:
            return False
        self.current_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        self.display_frame = self.current_frame.copy()
        return True

    def compute_auto_rc(self, bbox):
        x, y, w, h = bbox
        target_center_x = x + w / 2
        target_center_y = y + h / 2
        frame_center_x = FRAME_WIDTH / 2
        frame_center_y = FRAME_HEIGHT / 2

        error_x = target_center_x - frame_center_x
        error_y = target_center_y - frame_center_y
        area_ratio = (w * h) / self.reference_area if self.reference_area else 1.0

        yaw = 0
        ud = 0
        fb = 0

        if abs(error_x) > 35:
            yaw = clamp(error_x * 0.22, -MAX_YAW_SPEED, MAX_YAW_SPEED)

        if abs(error_y) > 30:
            ud = clamp(-error_y * 0.20, -MAX_UP_DOWN_SPEED, MAX_UP_DOWN_SPEED)

        if area_ratio < 0.85:
            fb = clamp((1.0 - area_ratio) * 90, 10, MAX_FORWARD_BACK_SPEED)
        elif area_ratio > 1.18:
            fb = clamp(-(area_ratio - 1.0) * 90, -MAX_FORWARD_BACK_SPEED, -10)

        return 0, fb, ud, yaw

    def draw_overlay(self, manual_mode=False):
        cv2.line(self.display_frame, (FRAME_WIDTH // 2 - 15, FRAME_HEIGHT // 2),
                 (FRAME_WIDTH // 2 + 15, FRAME_HEIGHT // 2), (255, 255, 0), 1)
        cv2.line(self.display_frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2 - 15),
                 (FRAME_WIDTH // 2, FRAME_HEIGHT // 2 + 15), (255, 255, 0), 1)

        status = "TRACKING" if self.tracking_active else "IDLE"
        if manual_mode:
            status = "MANUAL"

        lines = [
            f"Battery: {self.get_battery_level()}%",
            f"Flight: {'ON' if self.flying else 'OFF'}",
            f"Mode: {status}",
            "t takeoff | x land | r reset | q quit",
            "i/k forward/back | j/l left/right | w/s up/down | a/d turn",
            "Drag mouse to select target",
        ]

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
        print("请先连接 Tello WiFi。")
        print("操作说明:")
        print("  t: 起飞")
        print("  x: 降落")
        print("  鼠标拖拽: 选择目标")
        print("  r: 清除当前目标")
        print("  q: 退出程序")

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            if not self.read_frame():
                continue

            manual_mode = False
            rc_command = (0, 0, 0, 0)

            if self.tracking_active and self.tracker is not None:
                ok, bbox = self.tracker.update(self.current_frame)
                if ok:
                    self.tracking_bbox = tuple(int(v) for v in bbox)
                    rc_command = self.compute_auto_rc(self.tracking_bbox)
                else:
                    print("目标丢失，请重新框选。")
                    self.reset_tracking(stop_motion=True)

            self.draw_overlay(manual_mode=manual_mode)
            cv2.imshow(WINDOW_NAME, self.display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('t'):
                self.takeoff()
                continue
            if key == ord('x'):
                self.land()
                continue
            if key == ord('r'):
                self.reset_tracking(stop_motion=True)
                print("已清除当前目标。")
                continue

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
                self.draw_overlay(manual_mode=manual_mode)
                cv2.imshow(WINDOW_NAME, self.display_frame)

            if self.flying:
                self.send_rc(*rc_command)

    def close(self):
        try:
            if self.flying:
                self.send_rc(0, 0, 0, 0)
                self.tello.land()
        finally:
            try:
                self.tello.streamoff()
            except Exception:
                pass
            try:
                self.tello.end()
            except Exception:
                pass
            cv2.destroyAllWindows()


def main():
    app = TelloKCFTracker()
    try:
        app.connect()
        app.run()
    finally:
        app.close()
        print("程序已退出")


if __name__ == "__main__":
    main()