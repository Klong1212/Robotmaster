# -*-coding:utf-8-*-
import robomaster
from robomaster import robot, vision
import time
import cv2

# คลาสสำหรับ PID Controller (เหมือนเดิม)
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
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

# --- เปลี่ยนมาใช้ List เพื่อเก็บ Marker ทั้งหมดที่เห็น ---
g_detected_markers = []

# Callback function เมื่อตรวจจับ Marker ได้
def on_detect_marker(marker_info):
    global g_detected_markers
    g_detected_markers = marker_info

if __name__ == '__main__':
    # ================== 1. ตั้งค่าและเชื่อมต่อ ==================
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster

    ep_camera.start_video_stream(display=False)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    print("Robot initialized. Ready for the mission.")
    time.sleep(1)

    # ================== 2. ตั้งค่า PID และลำดับภารกิจ ==================
    yaw_pid = PIDController(kp=180, ki=5, kd=10)
    AIM_TOLERANCE = 0.01

    # --- กำหนดลำดับการยิงเป้าโดยใช้ "ตำแหน่ง" ---
    mission_sequence = ["LEFT", "CENTER", "RIGHT", "CENTER", "LEFT"]
    
    # ================== 3. เริ่มปฏิบัติภารกิจ ==================
    for target_position in mission_sequence:
        print(f"\n--- Current Target: {target_position} ---")
        yaw_pid.reset()
        
        target_locked = False
        while not target_locked:
            # --- ตรรกะการเลือกเป้าหมาย ---
            # 1. เรียงลำดับ Marker ที่เห็นจากซ้ายไปขวา (ตามค่า x)
            sorted_markers = sorted(g_detected_markers, key=lambda m: m[0])
            
            # 2. เลือกเป้าหมายจากลำดับที่เรียงแล้ว
            selected_target = None
            if len(sorted_markers) >= 3: # ต้องเห็นอย่างน้อย 3 อัน
                if target_position == "LEFT":
                    selected_target = sorted_markers[0]
                elif target_position == "CENTER":
                    selected_target = sorted_markers[1]
                elif target_position == "RIGHT":
                    selected_target = sorted_markers[2]
            
            # --- ส่วนแสดงผลกล้อง ---

            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Exception as e:
                # ถ้าดึงภาพไม่สำเร็จ ให้ข้ามไปรอบถัดไป
                print(f"Could not read image from camera: {e}")
                continue

            if img is not None:
                cv2.imshow("Live View", img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # --- ส่วนควบคุม PID ---
            if selected_target:
                # selected_target[0] คือค่า x ของ Marker
                error_x = selected_target[0] - 0.5
                print(f"Target '{target_position}' found. Error: {error_x:.3f}", end='\r')

                if abs(error_x) < AIM_TOLERANCE:
                    print(f"\nTarget '{target_position}' is locked!")
                    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
                    time.sleep(0.5)
                    ep_blaster.fire(fire_type='ir', times=1)
                    print("FIRE!")
                    time.sleep(1)
                    target_locked = True # ออกจาก Loop เพื่อไปเป้าหมายถัดไป

                yaw_speed = yaw_pid.compute(error_x)
                ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=0)
            else:
                print(f"Searching for targets... (Found {len(sorted_markers)}/3)", end='\r')
                ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
            
            time.sleep(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # ================== 4. สิ้นสุดภารกิจ ==================
    print("\n--- Mission Completed ---")
    cv2.destroyAllWindows()
    ep_vision.unsub_detect_info(name="marker")
    ep_camera.stop_video_stream()
    ep_robot.close()