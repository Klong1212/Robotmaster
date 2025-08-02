# -*-coding:utf-8-*-
import robomaster
from robomaster import robot, vision, camera # <-- แก้ไข: เพิ่ม camera
import time
import math
import cv2
import threading
import csv

# (คลาส PIDController เหมือนเดิมทุกประการ)
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.last_error = 0; self.integral = 0
        self.last_time = time.time()
    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return 0
        p_term = self.kp * error; self.integral += error * dt
        i_term = self.ki * self.integral
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        self.last_error = error; self.last_time = current_time
        return max(min(output, 100), -100)
    def reset(self):
        self.last_error = 0; self.integral = 0
        self.last_time = time.time()

# --- Globals and Callbacks (เหมือนเดิม) ---
current_yaw_angle = 0.0
g_detected_markers = []
def sub_gimbal_angle_handler(angle_info):
    global current_yaw_angle
    current_yaw_angle = angle_info[1]
def on_detect_marker(marker_info):
    global g_detected_markers
    g_detected_markers = marker_info

latest_frame = None
stop_event = threading.Event()
def camera_thread_func(ep_camera):
    global latest_frame
    while not stop_event.is_set():
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            if img is not None:
                latest_frame = img
        except Exception as e:
            print(f"Camera thread error: {e}")
            time.sleep(0.1)

if __name__ == '__main__':
    # ... (ส่วนประกาศตัวแปรค่ามุมเหมือนเดิม)
    DISTANCE_ROBOT_TO_TARGET_LINE = 1.0; DISTANCE_CENTER_TO_SIDE_TARGET = 0.6
    angle_rad = math.atan(DISTANCE_CENTER_TO_SIDE_TARGET / DISTANCE_ROBOT_TO_TARGET_LINE)
    angle_deg = math.degrees(angle_rad)
    TARGET_YAW_LEFT = -round(angle_deg, 2); TARGET_YAW_RIGHT = round(angle_deg, 2); TARGET_YAW_CENTER = 0.0

    # --- เตรียมการสำหรับบันทึกข้อมูล (เหมือนเดิม) ---
    log_data = []
    log_header = [
        'timestamp', 'stage', 'target_angle', 'current_angle', 
        'angle_error', 'angle_pid_output', 'marker_error_x', 'marker_error_y', 
        'vision_yaw_pid_output', 'vision_pitch_pid_output'
    ]
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    # ... (ส่วน initialize หุ่นยนต์และ thread เหมือนเดิม)
    ep_gimbal = ep_robot.gimbal; ep_blaster = ep_robot.blaster
    ep_camera = ep_robot.camera; ep_vision = ep_robot.vision
    
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    print("Robot initialized. Ready for hybrid mission.")
    time.sleep(2)
    ep_gimbal.sub_angle(freq=20, callback=sub_gimbal_angle_handler)
    
    # --- แก้ไข: ปรับความละเอียดวิดีโอให้ชัดและถี่ขึ้น ---
    print("Starting 720p video stream...")
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    cam_thread = threading.Thread(target=camera_thread_func, args=(ep_camera,))
    cam_thread.start()
    
    # --- ตั้งค่า PID และลำดับภารกิจ (เหมือนเดิม) ---
    angle_pid = PIDController(kp=5, ki=0.1, kd=4)
    vision_yaw_pid = PIDController(kp=180, ki=5, kd=10)
    vision_pitch_pid = PIDController(kp=180, ki=5, kd=10)
    COARSE_AIM_TOLERANCE = 2.0
    FINE_AIM_TOLERANCE_X = 0.01
    FINE_AIM_TOLERANCE_Y = 0.01
    mission_sequence = [TARGET_YAW_LEFT, TARGET_YAW_CENTER, TARGET_YAW_RIGHT, TARGET_YAW_CENTER, TARGET_YAW_LEFT]
    coun=0
    
    try:
        start_time = time.time() # เวลาเริ่มต้นของภารกิจทั้งหมด
        for target_angle in mission_sequence:
            print(f"\n--- Mission: Target Angle {target_angle} degrees ---")
            
            # --- STAGE 1: COARSE AIMING ---
            print("Stage 1: Coarse aiming...")
            angle_pid.reset()
            while True:
                error = target_angle - current_yaw_angle
                print(f"  [Coarse] Target: {target_angle:.2f}, Current: {current_yaw_angle:.2f}, Error: {error:.2f}", end='\r')
                yaw_speed = angle_pid.compute(error)
                
                # บันทึกข้อมูลของ Stage 1
                timestamp = time.time() - start_time
                log_row = [timestamp, 'coarse', target_angle, current_yaw_angle, error, yaw_speed, '', '', '', '']
                log_data.append(log_row)

                if abs(error) < COARSE_AIM_TOLERANCE:
                    print("\n  [Coarse] Aim complete. Switching to fine-tuning.")
                    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                    break
                
                ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=0)
                time.sleep(0.01)

            # --- STAGE 2: FINE-TUNING ---
            print("Stage 2: Fine-tuning with vision...")
            vision_yaw_pid.reset()
            vision_pitch_pid.reset()
            fine_tune_start_time = time.time()
            
            while time.time() - fine_tune_start_time < 10:
                img = latest_frame
                if img is not None:
                    # ปรับขนาดภาพให้เล็กลงเพื่อการแสดงผล (ถ้าต้องการ) แต่ยังคงประมวลผลจากภาพเต็ม
                    display_img = cv2.resize(img, (960, 540)) 
                    cv2.imshow("Live View (720p)", display_img)
                    cv2.waitKey(1)
                
                target_marker = None
                min_dist_from_center = float('inf')
                for m in g_detected_markers:
                    dist_x = abs(m[0] - 0.5)
                    if dist_x < min_dist_from_center:
                        min_dist_from_center = dist_x
                        target_marker = m
                
                timestamp = time.time() - start_time
                if target_marker:
                    error_x = target_marker[0] - 0.5
                    error_y = 0.5 - target_marker[1]
                    print(f"  [Fine] Marker found. Error X: {error_x:.3f}, Y: {error_y:.3f}", end='\r')
                    
                    yaw_speed_vision = vision_yaw_pid.compute(error_x)
                    pitch_speed_vision = vision_pitch_pid.compute(error_y)

                    # บันทึกข้อมูลของ Stage 2 (เจอ Marker)
                    log_row = [timestamp, 'fine_tuning', target_angle, current_yaw_angle, '', '', error_x, error_y, yaw_speed_vision, pitch_speed_vision]
                    log_data.append(log_row)
                    
                    if abs(error_x) < FINE_AIM_TOLERANCE_X and abs(error_y) < FINE_AIM_TOLERANCE_Y:
                        print(f"\n  [Fine] Target locked!")
                        ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                        time.sleep(0.5)
                        ep_blaster.fire(fire_type='ir',times=1)
                        print("FIRE!")
                        time.sleep(1)
                        coun+=1
                        break 
                    
                    ep_gimbal.drive_speed(yaw_speed=yaw_speed_vision, pitch_speed=pitch_speed_vision)
                else:
                    print("  [Fine] Marker not found. Holding position...", end='\r')
                    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                    
                    # บันทึกข้อมูลของ Stage 2 (ไม่เจอ Marker)
                    log_row = [timestamp, 'searching', target_angle, current_yaw_angle, '', '', '', '', '', '']
                    log_data.append(log_row)

                time.sleep(0.01)
            else:
                print("\n  [Fine] Could not lock target in time.")

    finally:
        print("\n--- Mission Completed, Cleaning up ---")
        
        # เขียนข้อมูลที่บันทึกไว้ลงไฟล์ CSV
        log_filename = "gimbal_log_720p.csv"
        print(f"Saving log data to {log_filename}...")
        with open(log_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
            writer.writerows(log_data)
        print("Log data saved.")

        # ส่วน cleanup อื่นๆ
        stop_event.set()
        cam_thread.join()
        cv2.destroyAllWindows()
        ep_gimbal.unsub_angle()
        ep_vision.unsub_detect_info(name="marker")
        ep_camera.stop_video_stream()
        ep_robot.close()