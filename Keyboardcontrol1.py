import logging
import time
import cv2
from djitellopy import tello
import KeyPressModule as kp
from time import sleep

# ----- 可被其他模块导入的常量 -----
Camera_Width = 720
Camera_Height = 480
DetectRange = [6000, 11000]      # 人脸跟踪参数（可选保留）
PID_Parameter = [0.5, 0.0004, 0.4]
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 255)          # 红色
LINE_THICKNESS = 1


def getKeyboardInput(drone, speed, image):
    """
    检测键盘按键，计算RC控制指令，并在图像上叠加信息。
    :param drone: Tello对象（用于获取电量、高度）
    :param speed: 速度值（cm/s 或 °/s）
    :param image: 要叠加文字的图像帧
    :return: (lr, fb, ud, yaw, manual_active)
             manual_active 为 True 表示有手动移动指令，False 表示没有
    """
    lr, fb, ud, yaw = 0, 0, 0, 0
    manual_active = False   # 标记是否有任何移动按键被按下

    # 拍照（按 e 键）
    if kp.getKey("e"):
        filename = f'D:/snap-{time.strftime("%H%M%S")}.jpg'
        cv2.imwrite(filename, image)
        print(f"照片已保存：{filename}")

    # 起飞/降落
    if kp.getKey("UP"):
        drone.takeoff()
    elif kp.getKey("DOWN"):
        drone.land()

    # 左右平移
    if kp.getKey("j"):
        manual_active = True
        lr = -speed
    elif kp.getKey("l"):
        manual_active = True
        lr = speed

    # 前后飞行
    if kp.getKey("i"):
        manual_active = True
        fb = speed
    elif kp.getKey("k"):
        manual_active = True
        fb = -speed

    # 上升下降
    if kp.getKey("w"):
        manual_active = True
        ud = speed
    elif kp.getKey("s"):
        manual_active = True
        ud = -speed

    # 旋转
    if kp.getKey("a"):
        manual_active = True
        yaw = -speed
    elif kp.getKey("d"):
        manual_active = True
        yaw = speed

    # 在图像上叠加电池、高度信息
    info_text = f"battery: {drone.get_battery()}%  height: {drone.get_height()}cm  time: {time.strftime('%H:%M:%S')}"
    cv2.putText(image, info_text, (10, 20), FONT, FONT_SCALE, FONT_COLOR, LINE_THICKNESS)

    # 如果有手动指令，显示当前的指令值
    if manual_active:
        cmd_text = f"Manual: lr={lr} fb={fb} ud={ud} yaw={yaw}"
        cv2.putText(image, cmd_text, (10, 40), FONT, FONT_SCALE, FONT_COLOR, LINE_THICKNESS)

    return lr, fb, ud, yaw, manual_active


# ----- 以下是原主程序，仅在直接运行此文件时执行 -----
if __name__ == "__main__":
    # 初始化无人机
    Drone = tello.Tello()
    Drone.connect()
    Drone.streamon()
    Drone.LOGGER.setLevel(logging.ERROR)
    sleep(5)

    kp.init()
    print("键盘控制已启动。按 UP 起飞，DOWN 降落，i/j/k/l 控制方向，w/s 升降，a/d 旋转，e 拍照。")

    while True:
        original = Drone.get_frame_read().frame
        frame = cv2.resize(original, (Camera_Width, Camera_Height))

        # 调用函数获取指令，但这里仍然不发送，而是由主程序发送（为了保持一致性，我们在主程序中发送）
        lr, fb, ud, yaw, manual = getKeyboardInput(Drone, 70, frame)
        if manual:
            Drone.send_rc_control(lr, fb, ud, yaw)

        cv2.imshow("Drone Control Centre", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    Drone.streamoff()
    cv2.destroyAllWindows()