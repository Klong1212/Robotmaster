# -*-coding:utf-8-*-
import robomaster
from robomaster import robot
import time
import math
import pandas as pd

# --- 1. PID Controller Class (ไม่เปลี่ยนแปลง) ---
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.001:  # ป้องกันการหารด้วยศูนย์
            dt = 0.001
        
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.last_error = error
        self.last_time = current_time
        return output

    def reset(self):
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

# --- 2. Global Variables & Callback Functions ---
robot_position = {'x': 0, 'y': 0}
robot_attitude = {'yaw': 0, 'pitch': 0, 'roll': 0}
data_log = []

def position_handler(data):
    x, y, z = data
    robot_position['x'] = x
    robot_position['y'] = y

def attitude_handler(data):
    yaw, pitch, roll = data
    robot_attitude['yaw'] = yaw

# --- 3. ฟังก์ชันควบคุมการเคลื่อนที่ ---
def pid_move_forward(ep_chassis, target_distance, kp=1.5, ki=0.05, kd=0.3):
    pid = PIDController(kp, ki, kd)
    start_pos = robot_position.copy()

    while True:
        distance_traveled = math.sqrt((robot_position['x'] - start_pos['x'])**2 + (robot_position['y'] - start_pos['y'])**2)
        error = target_distance - distance_traveled
        
        if abs(error) < 0.02: # Tolerance 2 ซม.
            ep_chassis.drive_speed(x=0, y=0, z=0)
            break
            
        control_speed = pid.compute(error)
        control_speed = max(min(control_speed, 0.7), -0.7)
        ep_chassis.drive_speed(x=control_speed, y=0, z=0)
        
        # ★ แก้ไข: เพิ่ม current_x และ current_y
        log_entry = {
            'timestamp': time.time(), 
            'action': 'move', 
            'target': target_distance,
            'current_x': robot_position['x'],
            'current_y': robot_position['y'],
            'current_distance': distance_traveled, 
            'error': error, 
            'output': control_speed
        }
        data_log.append(log_entry)
        
        time.sleep(0.02)

def pid_turn(ep_chassis, target_angle, kp=1.5, ki=0.05, kd=0.3):
    pid = PIDController(kp, ki, kd)
    
    while True:
        current_yaw = robot_attitude['yaw']
        error = target_angle - current_yaw
        
        if error > 180: error -= 360
        if error < -180: error += 360

        if abs(error) < 1.0: # Tolerance 1 องศา
            ep_chassis.drive_speed(x=0, y=0, z=0)
            break

        control_speed = pid.compute(error)
        control_speed = max(min(control_speed, 80), -80)
        ep_chassis.drive_speed(x=0, y=0, z=control_speed)
        
        # ★ แก้ไข: เพิ่ม current_x, current_y และเปลี่ยนชื่อ 'current'
        log_entry = {
            'timestamp': time.time(), 
            'action': 'turn', 
            'target': target_angle, 
            'current_x': robot_position['x'],
            'current_y': robot_position['y'],
            'current_yaw': current_yaw, 
            'error': error, 
            'output': control_speed
        }
        data_log.append(log_entry)
        
        time.sleep(0.02)

# --- 4. Main Program ---
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_chassis.sub_position(freq=50, callback=position_handler)
    ep_chassis.sub_attitude(freq=50, callback=attitude_handler)
    time.sleep(2)

    side_length = 0.6
    
    # === วนลูป 4 ครั้งเพื่อวาดสี่เหลี่ยม ===
    
    for i in range(4):
        print(f"\n--- Side {i+1}/4: Moving forward {side_length}m ---")
        pid_move_forward(ep_chassis, side_length)
        time.sleep(1)

        target_yaw = robot_attitude['yaw'] + 90
        if target_yaw > 180:
            target_yaw -= 360
        
        print(f"\n--- Side {i+1}/4: Turning to {target_yaw:.1f} degrees ---")
        pid_turn(ep_chassis, target_yaw)
        time.sleep(1)

    # --- จบการทำงานและบันทึกข้อมูล ---
    ep_chassis.unsub_position()
    ep_chassis.unsub_attitude()
    
    if data_log:
        df = pd.DataFrame(data_log)
        df.to_csv("square_pid_log_with_xy_15_001_08.csv", index=False)
        print("\n✅ Data saved to square_pid_log_with_xy_2.csv")

    ep_robot.close()