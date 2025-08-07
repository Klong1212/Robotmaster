# -*-coding:utf-8-*-

import robomaster
from robomaster import robot, vision, camera
import time
import math
import cv2
import threading
import csv
import traceback

# --- (PID Controller Class ไม่เปลี่ยนแปลง) ---
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.last_error = 0
        self.integral = 0
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

# --- 2. Global Variables & Callback Functions ---
current_yaw_angle = 0.0
current_pitch_angle = 0.0
g_detected_markers = []
latest_frame = None
stop_event = threading.Event()

# ★ เพิ่ม Lock สำหรับป้องกัน Race Condition
data_lock = threading.Lock()

def sub_gimbal_angle_handler(angle_info):
    global current_yaw_angle, current_pitch_angle
    with data_lock:
        current_pitch_angle, current_yaw_angle = angle_info[0], angle_info[1]

def on_detect_marker(marker_info):
    global g_detected_markers
    with data_lock:
        g_detected_markers = marker_info

# --- 3. Camera Thread ---
def camera_thread_func(ep_camera):
    global latest_frame
    while not stop_event.is_set():
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is not None:
                with data_lock:
                    latest_frame = img.copy() # ใช้ copy() เพื่อความปลอดภัย
        except Exception:
            print("Camera thread error:", traceback.format_exc())
            time.sleep(1)

# --- 4. Main Program ---
if __name__ == '__main__':
    # ... (ส่วนตั้งค่าภารกิจและ Log header เหมือนเดิม) ...
    DISTANCE_ROBOT_TO_TARGET_LINE = 1.0
    DISTANCE_CENTER_TO_SIDE_TARGET = 0.6
    angle_rad = math.atan(DISTANCE_CENTER_TO_SIDE_TARGET / DISTANCE_ROBOT_TO_TARGET_LINE)
    angle_deg = math.degrees(angle_rad)
    TARGET_YAW_LEFT = -round(angle_deg, 2)
    TARGET_YAW_RIGHT = round(angle_deg, 2)
    TARGET_YAW_CENTER = 0.0
    mission_sequence = [TARGET_YAW_LEFT, TARGET_YAW_CENTER, TARGET_YAW_RIGHT, TARGET_YAW_CENTER,TARGET_YAW_LEFT]
    log_data = []
    log_header = [
        'timestamp', 'stage', 'target_yaw', 'current_yaw', 'current_pitch',
        'yaw_error', 'angle_pid_output', 'marker_error_x', 'marker_error_y',
        'vision_yaw_pid_output', 'vision_pitch_pid_output'
    ]

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    # ... (ส่วน initialize อุปกรณ์อื่นๆ เหมือนเดิม) ...
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster
    ep_camera = ep_robot.camera
    ep_vision = ep_robot.vision

    try:
        ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
        print("Robot initialized. Ready for mission.")
        time.sleep(2)

        # ... (ส่วน Subscribe และเริ่ม Thread เหมือนเดิม) ...
        ep_gimbal.sub_angle(freq=20, callback=sub_gimbal_angle_handler)
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
        ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
        cam_thread = threading.Thread(target=camera_thread_func, args=(ep_camera,))
        cam_thread.start()

        # ... (ส่วน PID Controllers และ Tolerances เหมือนเดิม) ...
        angle_pid = PIDController(kp=4.5, ki=0.1, kd=3.5)
        vision_yaw_pid = PIDController(kp=180, ki=5, kd=10)
        vision_pitch_pid = PIDController(kp=180, ki=5, kd=10)
        COARSE_AIM_TOLERANCE = 3.0
        FINE_AIM_TOLERANCE_X = 0.01
        FINE_AIM_TOLERANCE_Y = 0.01
        
        start_time = time.time()
        for target_yaw in mission_sequence:
            print(f"\n--- Mission: Target Yaw {target_yaw}° ---")

            # === Stage 1: Coarse Aiming ===
            angle_pid.reset()
            while True:
                with data_lock:
                    error = target_yaw - current_yaw_angle
                    # อ่านค่า frame ภายใน lock เพื่อแสดงผล
                    img = latest_frame

                if img is not None:
                    display_img = cv2.resize(img, (720, 480))
                    cv2.imshow("Live View", display_img)
                    cv2.waitKey(1)

                yaw_speed = angle_pid.compute(error)
                ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=0)
                
                timestamp = time.time() - start_time
                with data_lock:
                    log_row = [timestamp, 'coarse', target_yaw, current_yaw_angle, current_pitch_angle, error, yaw_speed, '', '', '', '']
                log_data.append(log_row)

                if abs(error) < COARSE_AIM_TOLERANCE:
                    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                    break
                time.sleep(0.01)

            # === Stage 2: Fine-Tuning ===
            vision_yaw_pid.reset()
            vision_pitch_pid.reset()
            fine_tune_start_time = time.time()
            
            target_locked = False
            while time.time() - fine_tune_start_time < 5:
                with data_lock:
                    local_markers = g_detected_markers
                    img = latest_frame
                
                if img is not None:
                    display_img = cv2.resize(img, (720, 480))
                    cv2.imshow("Live View", display_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt # อนุญาตให้กด q ออกได้
                
                target_marker = None
                if local_markers:
                    target_marker = min(local_markers, key=lambda m: abs(m[0] - 0.5))

                timestamp = time.time() - start_time
                
                if target_marker:
                    error_x = target_marker[0] - 0.5
                    error_y = 0.5 - target_marker[1]
                    yaw_speed = vision_yaw_pid.compute(error_x)
                    pitch_speed = vision_pitch_pid.compute(error_y)
                    ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=pitch_speed)
                    
                    with data_lock:
                        log_row = [timestamp, 'fine_tuning', target_yaw, current_yaw_angle, current_pitch_angle, '', error_x, error_y, yaw_speed, pitch_speed]
                    log_data.append(log_row)

                    if abs(error_x) < FINE_AIM_TOLERANCE_X and abs(error_y) < FINE_AIM_TOLERANCE_Y:
                        target_locked = True
                        break
                else:
                    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                    with data_lock:
                        log_row = [timestamp, 'searching', target_yaw, current_yaw_angle, current_pitch_angle, '', '', '', '', '']
                    log_data.append(log_row)
                time.sleep(0.02)

            # === Stage 3: Firing ===
            if target_locked:
                print("  [Fine] Target locked! FIRING!")
                ep_blaster.fire(fire_type='ir', times=1)
                time.sleep(1)
            else:
                print("  [Fine] Could not lock target in time.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        # ... (ส่วน cleanup และบันทึก CSV เหมือนเดิม) ...
        print("\n--- Mission End, Cleaning up ---")
        stop_event.set()
        cam_thread.join()
        cv2.destroyAllWindows()
        ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
        ep_gimbal.unsub_angle()
        ep_vision.unsub_detect_info(name="marker")
        ep_camera.stop_video_stream()
        
        log_filename = "gimbal_mission_log.csv"
        print(f"Saving log data to {log_filename}...")
        with open(log_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
            writer.writerows(log_data)
        print("Log data saved.")

        ep_robot.close()
        print("Cleanup complete.")