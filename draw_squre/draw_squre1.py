from robomaster import robot
import pandas as pd
import time
import math

data_log = []
start_time = time.time()
latest_data = {"position_x": 0, "position_y": 0, "position_z": 0}

# === PID Controller ===
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.last_error = 0

    def compute(self, target, current, dt):
        error = target - current
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def get_current_yaw(ep_robot):
    imu_data = ep_robot.sensor.imu.get_imu_data()
    return imu_data[0]  # ต้องเช็กจาก SDK ว่า index 0 คือ yaw จริงหรือไม่

def pid_turn(ep_robot, target_angle, kp=0.8, ki=0.0, kd=0.05):
    pid = PIDController(kp, ki, kd)
    while True:
        start_t = time.time()
        current_yaw = get_current_yaw(ep_robot)
        control = pid.compute(target_angle, current_yaw, time.time() - start_t)
        control = max(min(control, 60), -60)  # เพิ่มความเร็วหมุน
        if abs(target_angle - current_yaw) < 1.0:
            ep_robot.chassis.drive_speed(x=0, y=0, z=0, timeout=0.05)
            break
        ep_robot.chassis.drive_speed(x=0, y=0, z=control, timeout=0.05)

def pid_go(ep_robot, target_x, target_y, kp=0.8, ki=0.0, kd=0.05):
    pid_x = PIDController(kp, ki, kd)
    pid_y = PIDController(kp, ki, kd)
    while True:
        start_t = time.time()
        current_x = latest_data["position_x"]
        current_y = latest_data["position_y"]
        dt = time.time() - start_t

        control_x = pid_x.compute(target_x, current_x, dt)
        control_y = pid_y.compute(target_y, current_y, dt)

        # # เพิ่มขีดจำกัดความเร็ว
        # control_x = max(min(control_x, 1.0), -1.0)
        # control_y = max(min(control_y, 1.0), -1.0)

        if abs(target_x - current_x) < 0.02 and abs(target_y - current_y) < 0.02:
            ep_robot.chassis.drive_speed(x=0, y=0, z=0, timeout=0.05)
            break
        print(f"Control X: {control_x}, Control Y: {control_y}")  # Debugging
        ep_robot.chassis.drive_speed(x=control_x, y=control_y, z=0, timeout=0.05)

# === Logging ===
def position_handler(data):
    x, y, z = data
    latest_data.update({"position_x": x, "position_y": y, "position_z": z})


# === Main ===
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal

    # เริ่ม subscription
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_chassis.sub_position(freq=10, callback=position_handler)
    target_position=[[0,0.6],[0.6,0.6],[0.6,0],[0,0]]
    current_angle = 0.0
    for i in target_position:
        print(i)
        pid_go(ep_robot, i[0], i[1])
        current_angle += 90
        if current_angle >= 360:
            current_angle -= 360
        pid_turn(ep_robot, current_angle)

    ep_chassis.unsub_position()

    # Save CSV
    df = pd.DataFrame(data_log)
    df.to_csv("draw_square_pid.csv", index=False)
    print("✅ Data saved to draw_square_pid.csv")

    ep_robot.close()
