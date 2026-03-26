import cv2
import numpy as np
from djitellopy import Tello
import time
import KeyPressModule as kp
from Keyboardcontrol import getKeyboardInput, Camera_Width, Camera_Height, FONT, FONT_SCALE, FONT_COLOR, LINE_THICKNESS

# ----- 初始化 -----
drone = Tello()
drone.connect()
print(f"电量：{drone.get_battery()}%")
drone.streamon()
frame_read = drone.get_frame_read()
kp.init()

# 起飞并升高
drone.takeoff()
drone.move_up(80)
time.sleep(2)

# ----- 参数设置 -----
forward_speed = 20      # 自动前进速度 (cm/s)
rotate_speed = 60       # 自动旋转速度 (°/s)
threshold = 50000       # 自动避障阈值（中等风险）
danger_threshold = 80000  # 手动接管阈值（高风险，需根据环境调整）
manual_speed = 70       # 手动控制速度

prev_frame = None

# 状态显示
manual_active = False
override_active = False

print("开始飞行，手动模式下如遇危险将自动避障...")

def get_avoid_command(center, left, right, thr, fwd_spd, rot_spd):
    """根据区域变化生成避障指令（停止前进，向空侧旋转）"""
    if center > thr:
        # 前方有障碍，比较左右
        if left < right:
            yaw = -rot_spd   # 向左转
        else:
            yaw = rot_spd    # 向右转
        fb = 0
    else:
        # 前方安全，继续前进
        fb = fwd_spd
        yaw = 0
    return 0, fb, 0, yaw   # lr, fb, ud, yaw

try:
    while True:
        # 1. 获取当前帧
        frame = frame_read.frame
        if frame is None:
            continue
        frame = cv2.resize(frame, (Camera_Width, Camera_Height))

        # 2. 获取手动指令（但先不发送）
        lr_man, fb_man, ud_man, yaw_man, man_active = getKeyboardInput(drone, manual_speed, frame)

        # 3. 图像处理，计算区域变化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is not None:
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            h, w = thresh.shape
            left = thresh[:, 0:w//3]
            center = thresh[:, w//3:2*w//3]
            right = thresh[:, 2*w//3:w]

            left_change = np.sum(left == 255)
            center_change = np.sum(center == 255)
            right_change = np.sum(right == 255)

            # 可选：显示阈值图像用于调试
            cv2.imshow("Threshold", thresh)

            # 4. 判断是否进入手动接管模式（危险检测）
            #    危险阈值一般比自动阈值高，表示更紧急
            if center_change > danger_threshold:
                # 危险！自动接管
                override_active = True
                # 生成避障指令（忽略手动指令）
                lr, fb, ud, yaw = get_avoid_command(center_change, left_change, right_change,
                                                     danger_threshold, forward_speed, rotate_speed)
                mode_text = "OVERRIDE!"
                color = (0, 0, 255)  # 红色
            elif man_active:
                # 手动模式且无危险，使用手动指令
                override_active = False
                lr, fb, ud, yaw = lr_man, fb_man, ud_man, yaw_man
                mode_text = "MANUAL"
                color = (0, 255, 0)  # 绿色
            else:
                # 自动模式，使用避障逻辑
                override_active = False
                lr, fb, ud, yaw = get_avoid_command(center_change, left_change, right_change,
                                                     threshold, forward_speed, rotate_speed)
                mode_text = "AUTO"
                color = (255, 255, 0)  # 青色

            # 在画面右上角显示当前模式
            cv2.putText(frame, mode_text, (Camera_Width - 150, 30), FONT, FONT_SCALE, color, LINE_THICKNESS)

        else:
            # 第一帧，无上一帧，悬停
            lr, fb, ud, yaw = 0, 0, 0, 0
            mode_text = "INIT"
            cv2.putText(frame, mode_text, (Camera_Width - 150, 30), FONT, FONT_SCALE, (255,255,255), LINE_THICKNESS)

        # 更新上一帧
        prev_frame = gray

        # 5. 发送最终指令
        drone.send_rc_control(lr, fb, ud, yaw)

        # 6. 显示画面
        cv2.imshow("Drone Control Centre", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("程序被中断")

# ----- 结束飞行 -----
finally:
    print("停止飞行，准备降落...")
    drone.send_rc_control(0, 0, 0, 0)
    drone.land()
    drone.streamoff()
    cv2.destroyAllWindows()
    print("已降落，程序结束。")