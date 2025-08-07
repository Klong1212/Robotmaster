# -*-coding:utf-8-*-

from robomaster import robot, camera
import time
import math
import cv2
import threading
import csv

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.last_error = 0; self.integral = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return 0
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        self.last_error = error
        self.last_time = current_time
        return max(min(output, 100), -100)

    def reset(self):
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

current_yaw_angle = 0.0
g_detected_markers = []
latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()

def sub_gimbal_angle_handler(angle_info):
    global current_yaw_angle
    current_yaw_angle = angle_info[1]

def on_detect_marker(marker_info):
    global g_detected_markers
    # Ensure marker_info is a list, even if None or empty
    if marker_info is None:
        g_detected_markers = []
    else:
        g_detected_markers = marker_info

def camera_thread_func(ep_camera):
    global latest_frame
    while not stop_event.is_set():
        try:
            img = ep_camera.read_cv2_image(strategy="fifo", timeout=0.5)
            if img is not None:
                with frame_lock:
                    latest_frame = img
        except Exception as e:
            print(f"[Camera Thread] Error: {e}")
        time.sleep(0.05)  # ~20 FPS

if __name__ == '__main__':
    print("Initializing...")

    DISTANCE_ROBOT_TO_TARGET_LINE = 1.0
    DISTANCE_CENTER_TO_SIDE_TARGET = 0.6
    angle_rad = math.atan(DISTANCE_CENTER_TO_SIDE_TARGET / DISTANCE_ROBOT_TO_TARGET_LINE)
    angle_deg = math.degrees(angle_rad)

    TARGET_YAW_LEFT = -round(angle_deg, 2)
    TARGET_YAW_RIGHT = round(angle_deg, 2)
    TARGET_YAW_CENTER = 0.0
    mission_sequence = [TARGET_YAW_LEFT, TARGET_YAW_CENTER, TARGET_YAW_RIGHT, TARGET_YAW_CENTER, TARGET_YAW_LEFT]

    log_data = []
    log_header = [
        'timestamp', 'stage', 'target_angle', 'current_angle',
        'angle_error', 'angle_pid_output', 'marker_error_x', 'marker_error_y',
        'vision_yaw_pid_output', 'vision_pitch_pid_output'
    ]

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_camera = ep_robot.camera
    ep_vision = ep_robot.vision

    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    print("Robot initialized. Ready for hybrid mission.")
    time.sleep(2)

    ep_gimbal.sub_angle(freq=10, callback=sub_gimbal_angle_handler)
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    cam_thread = threading.Thread(target=camera_thread_func, args=(ep_camera,))
    cam_thread.start()

    angle_pid = PIDController(kp=5, ki=0.1, kd=4)
    vision_yaw_pid = PIDController(kp=180, ki=5, kd=10)
    vision_pitch_pid = PIDController(kp=180, ki=5, kd=10)

    COARSE_AIM_TOLERANCE = 4.0
    FINE_AIM_TOLERANCE_X = 0.01
    FINE_AIM_TOLERANCE_Y = 0.01

    # --- เพิ่มตัวแปรสำหรับ rate limit ---
    last_gimbal_cmd_time = 0
    GIMBAL_CMD_INTERVAL = 0.1  # ส่งคำสั่งไม่เกินทุก 0.1 วินาที
    last_yaw_speed = None
    last_pitch_speed = None
    def safe_drive_speed(yaw_speed, pitch_speed):
        global last_gimbal_cmd_time, last_yaw_speed, last_pitch_speed
        now = time.time()
        # ส่งคำสั่งเฉพาะเมื่อถึงเวลา และค่ามีการเปลี่ยนแปลง
        if (now - last_gimbal_cmd_time > GIMBAL_CMD_INTERVAL) or \
           (yaw_speed != last_yaw_speed) or (pitch_speed != last_pitch_speed):
            ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
            last_gimbal_cmd_time = now
            last_yaw_speed = yaw_speed
            last_pitch_speed = pitch_speed
    start_time = time.time()
    for target_angle in mission_sequence:
        print(f"\n--- Mission: Target Angle {target_angle} degrees ---")
        angle_pid.reset()

        print("Stage 1: Coarse aiming...")
        while True:
            error = target_angle - current_yaw_angle
            yaw_speed = angle_pid.compute(error)

            timestamp = time.time() - start_time
            log_data.append([timestamp, 'coarse', target_angle, current_yaw_angle, error, yaw_speed, '', '', '', ''])

            if abs(error) < COARSE_AIM_TOLERANCE:
                print("  [Coarse] Aim complete.")
                safe_drive_speed(0, 0)
                break

            safe_drive_speed(yaw_speed, 0)
            time.sleep(0.05)

        print("Stage 2: Fine-tuning with vision...")
        vision_yaw_pid.reset()
        vision_pitch_pid.reset()
        fine_tune_start_time = time.time()

        while time.time() - fine_tune_start_time < 10:
            target_marker = None
            min_dist = float('inf')
            with frame_lock:
                for m in g_detected_markers:
                    # Ensure marker data is valid and has at least 2 elements
                    if isinstance(m, (list, tuple)) and len(m) >= 2:
                        dist = abs(m[0] - 0.5)
                        if dist < min_dist:
                            target_marker = m
                            min_dist = dist

            timestamp = time.time() - start_time

            if target_marker:
                error_x = target_marker[0] - 0.5
                error_y = 0.5 - target_marker[1]
                yaw_speed_vision = vision_yaw_pid.compute(error_x)
                pitch_speed_vision = vision_pitch_pid.compute(error_y)

                log_data.append([timestamp, 'fine_tuning', target_angle, current_yaw_angle, '', '', error_x, error_y, yaw_speed_vision, pitch_speed_vision])

                if abs(error_x) < FINE_AIM_TOLERANCE_X and abs(error_y) < FINE_AIM_TOLERANCE_Y:
                    print("  [Fine] Target locked.")
                    safe_drive_speed(0, 0)
                    time.sleep(0.3)
                    ep_blaster.fire(fire_type='ir', times=1)
                    print("  FIRE!")
                    time.sleep(0.7)
                    break

                safe_drive_speed(yaw_speed_vision, pitch_speed_vision)
            else:
                print("  [Fine] Marker not found.")
                safe_drive_speed(0, 0)
                log_data.append([timestamp, 'searching', target_angle, current_yaw_angle, '', '', '', '', '', ''])

            time.sleep(0.1)  # ลดความถี่เหลือ 10 Hz

        else:
            print("  [Fine] Could not lock target in time.")

    try:
        print("\n--- Mission Completed, Cleaning up ---")
        safe_drive_speed(0, 0)
        time.sleep(0.2)

        log_filename = "gimbal_log.csv"
        with open(log_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
            writer.writerows(log_data)
        print("Log saved.")

    finally:
        stop_event.set()
        cam_thread.join()
        cv2.destroyAllWindows()
        ep_gimbal.unsub_angle()
        ep_vision.unsub_detect_info(name="marker")
        ep_camera.stop_video_stream()
        ep_robot.close()
