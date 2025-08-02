# -*-coding:utf-8-*-
import robomaster
from robomaster import robot, vision
import time
import cv2

# คลาสสำหรับ PID Controller
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

        if dt <= 0:
            return 0

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        # คำนวณ Output ทั้งหมด
        output = p_term + i_term + d_term

        self.last_error = error
        self.last_time = current_time
        
        # จำกัดความเร็วสูงสุดเพื่อไม่ให้ Gimbal หมุนเร็วเกินไป
        return max(min(output, 100), -100)
    
    def reset(self):
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()


# ตัวแปรสำหรับเก็บข้อมูล Marker ที่ตรวจจับได้
# เราจะใช้ dictionary เพื่อให้ง่ายต่อการค้นหา marker ด้วย 'info' (เช่น '1', '2')
detected_markers = {}

# Callback function เมื่อตรวจจับ Marker ได้
def on_detect_marker(marker_info):
    global detected_markers
    detected_markers.clear()
    for marker in marker_info:
        x, y, w, h, info = marker
        detected_markers[info] = {'x': x, 'y': y, 'w': w, 'h': h}
        # print(f"Detected Marker: {info} at x={x:.2f}")


if __name__ == '__main__':
    # ================== 1. ตั้งค่าและเชื่อมต่อ ==================
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_blaster = ep_robot.blaster

    # เริ่ม Video Stream และการตรวจจับ Marker
    ep_camera.start_video_stream(display=False)
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)
    
    # ตั้ง Gimbal ให้อยู่ตรงกลางก่อนเริ่ม
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    print("Robot initialized. Ready for the mission.")
    time.sleep(1)

    # ================== 2. ตั้งค่า PID และลำดับภารกิจ ==================
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★   นี่คือส่วนที่คุณต้อง "ปรับจูน (Tuning)"   ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    yaw_pid = PIDController(kp=180, ki=5, kd=10) # <<<<<<< ปรับค่า Kp, Ki, Kd ที่นี่
    AIM_TOLERANCE = 0.01  # ค่า Error ที่ยอมรับได้เพื่อจะถือว่าถึงเป้าหมายแล้ว (ยิ่งน้อยยิ่งแม่น)

    # ลำดับการยิงเป้า (ใช้ 'info' ของ Marker)
    mission_sequence = ["1", "2", "3","2","1"] # สมมติเป้าซ้าย=1, ขวา=2, กลาง=?
    
    # ================== 3. เริ่มปฏิบัติภารกิจ ==================
    for target_marker_id in mission_sequence:
        print(f"\n--- Current Target: Marker '{target_marker_id}' ---")
        yaw_pid.reset() # รีเซ็ตค่า PID ทุกครั้งที่เปลี่ยนเป้าหมายใหม่

        while True:
            # ดึงข้อมูล Marker ล่าสุด
            target_info = detected_markers.get(target_marker_id)

            if target_info:
                # คำนวณ Error: คือระยะห่างของ Marker จากกึ่งกลางของภาพในแนวนอน (0.5)
                error_x = target_info['x'] - 0.5

                print(f"Target '{target_marker_id}' found. Error: {error_x:.3f}")

                # ถ้าเข้าใกล้เป้าหมายมากพอแล้ว
                if abs(error_x) < AIM_TOLERANCE:
                    print(f"Target '{target_marker_id}' is locked!")
                    ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0) # หยุด Gimbal
                    time.sleep(0.5)
                    ep_blaster.fire(fire_type='ir', times=1) # ยิง LED (ถ้าจะยิงจริงเปลี่ยนเป็น 'ir')
                    print("FIRE!")
                    time.sleep(1) # รอหลังยิง
                    break # ไปยังเป้าหมายถัดไปใน mission_sequence

                # คำนวณความเร็วที่ต้องใช้จาก PID
                yaw_speed = yaw_pid.compute(error_x)
                
                # สั่งให้ Gimbal เคลื่อนที่ (Pitch speed เป็น 0)
                ep_gimbal.drive_speed(yaw_speed=yaw_speed, pitch_speed=0)

            else:
                # ถ้าไม่เจอ Marker เป้าหมาย ให้หยุดรอ
                print(f"Searching for target '{target_marker_id}'...")
                ep_gimbal.drive_speed(yaw_speed=0, pitch_speed=0)
            
            time.sleep(0.01) # หน่วงเวลาเล็กน้อยในลูป

    # ================== 4. สิ้นสุดภารกิจและคืนค่าทรัพยากร ==================
    print("\n--- Mission Completed ---")
    ep_vision.unsub_detect_info(name="marker")
    ep_camera.stop_video_stream()
    ep_robot.close()