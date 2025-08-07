# -*-coding:utf-8-*-
import robomaster
from robomaster import robot
import time
import math
import pandas as pd

# --- 1. PID Controller Class ---
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.last_error = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.001:
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
robot_attitude = {'yaw': 0}
data_log = []

def position_handler(data):
    x, y, z = data
    robot_position['x'] = x
    robot_position['y'] = y

def attitude_handler(data):
    yaw, pitch, roll = data
    robot_attitude['yaw'] = yaw

def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

# --- 3. PID Move Forward ---
def pid_move_forward(ep_chassis, target_distance, kp=1.5, ki=0.05, kd=0.05):
    pid = PIDController(kp, ki, kd)
    pid.reset()
    start_pos = robot_position.copy()

    while True:
        distance_traveled = math.sqrt((robot_position['x'] - start_pos['x'])**2 + (robot_position['y'] - start_pos['y'])**2)
        error = target_distance - distance_traveled

        if abs(error) < 0.02:
            ep_chassis.drive_speed(x=0, y=0, z=0)
            break

        control_speed = pid.compute(error)
        control_speed = max(min(control_speed, 0.5), -0.5)
        ep_chassis.drive_speed(x=control_speed, y=0, z=0)

        log_entry = {
            'timestamp': time.time(),
            'action': 'move',
            'target': target_distance,
            'current_x': robot_position['x'],
            'current_y': robot_position['y'],
            'current_distance': distance_traveled,
            'error': error,
            'output': control_speed,
            'speed_x': control_speed,
            'speed_y': 0,
            'speed_z': 0
        }
        data_log.append(log_entry)

        time.sleep(0.02)

# --- 4. PID Turn ---
def pid_turn(ep_chassis, target_angle, kp=1.5, ki=0.05, kd=0.05):
    pid = PIDController(kp, ki, kd)
    pid.reset()

    while True:
        current_yaw = robot_attitude['yaw']
        error = target_angle - current_yaw
        if error > 180:
            error -= 360
        if error < -180:
            error += 360

        if abs(error) < 1.0:
            ep_chassis.drive_speed(x=0, y=0, z=0)
            break

        control_speed = pid.compute(error)
        control_speed = max(min(control_speed, 80), -80)
        ep_chassis.drive_speed(x=0, y=0, z=control_speed)

        log_entry = {
            'timestamp': time.time(),
            'action': 'turn',
            'target': target_angle,
            'current_x': robot_position['x'],
            'current_y': robot_position['y'],
            'current_yaw': current_yaw,
            'error': error,
            'output': control_speed,
            'speed_x': 0,
            'speed_y': 0,
            'speed_z': control_speed
        }
        data_log.append(log_entry)

        time.sleep(0.02)

# --- 5. Go to Specific Point (back to origin) Without Rotation ---
def pid_go_to_point(ep_chassis, target_x, target_y, threshold=0.02, kp=1.5, ki=0.05, kd=0.3):
    pid_x = PIDController(kp, ki, kd)
    pid_y = PIDController(kp, ki, kd)
    pid_x.reset()
    pid_y.reset()

    while True:
        dx = target_x - robot_position['x']
        dy = target_y - robot_position['y']
        distance = math.sqrt(dx**2 + dy**2)

        if distance < threshold:
            ep_chassis.drive_speed(x=0, y=0, z=0)
            break

        vx = pid_x.compute(dx)
        vy = pid_y.compute(dy)

        vx = max(min(vx, 0.7), -0.7)
        vy = max(min(vy, 0.7), -0.7)

        ep_chassis.drive_speed(x=vx, y=vy, z=0)

        log_entry = {
            'timestamp': time.time(),
            'action': 'go_to_origin',
            'target_x': target_x,
            'target_y': target_y,
            'current_x': robot_position['x'],
            'current_y': robot_position['y'],
            'error_x': dx,
            'error_y': dy,
            'output_x': vx,
            'output_y': vy,
            'speed_x': vx,
            'speed_y': vy,
            'speed_z': 0
        }
        data_log.append(log_entry)

        time.sleep(0.02)

# --- 6. Main Program ---
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_chassis.sub_position(freq=50, callback=position_handler)
    ep_chassis.sub_attitude(freq=50, callback=attitude_handler)
    time.sleep(2)

    side_length = 0.6
    initial_yaw = robot_attitude['yaw']

    for loop_round in range(3):  # เดิน 3 รอบ
        print(f"\n🚀 Starting loop round {loop_round + 1}/3")
        for i in range(4):
            print(f"\n--- Side {i+1}/4: Moving forward {side_length}m ---")
            pid_move_forward(ep_chassis, side_length)
            time.sleep(1)

            target_yaw = normalize_angle(initial_yaw + (i + 1 + loop_round * 4) * 90)
            print(f"\n--- Side {i+1}/4: Turning to {target_yaw:.1f} degrees ---")
            pid_turn(ep_chassis, target_yaw)
            time.sleep(1)

        print("\n↩️ Returning to origin (0,0) without rotating")
        pid_go_to_point(ep_chassis, 0, 0)
        time.sleep(1)

    ep_chassis.unsub_position()
    ep_chassis.unsub_attitude()

    if data_log:
        df = pd.DataFrame(data_log)
        df.to_csv("square_pid_log_3rounds.csv", index=False)
        print("\n✅ Data saved to square_pid_log_3rounds.csv")

    time.sleep(2)
    ep_robot.close()
