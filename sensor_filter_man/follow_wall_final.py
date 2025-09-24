import robomaster
from robomaster import robot
import time
import math

a_power = 1302.301937
b_power = -0.490022

def adc_to_distance(adc_value):
    if adc_value is None or adc_value <= 100:
        return 200.0
    try:
        distance = (a_power / adc_value) ** (1 / -b_power)
        return distance
    except (ValueError, TypeError):
        return 200.0

class PID:
    def __init__(self, kp, ki, kd, output_limit=80, integral_limit=50):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral, self.prev_error = 0.0, 0.0
        self.output_limit = output_limit
        self.integral_limit = integral_limit

    def reset(self):
        self.integral, self.prev_error = 0.0, 0.0

    def compute(self, error, dt):
        if dt <= 0: return 0.0
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(min(output, self.output_limit), -self.output_limit)
        self.prev_error = error
        return output

front_distance_cm = None
current_yaw = 0.0

def sub_tof_handler(sub_info):
    global front_distance_cm
    if sub_info and isinstance(sub_info[0], (int, float)):
        front_distance_cm = sub_info[0] / 10.0

def sub_attitude_handler(attitude_info):
    global current_yaw
    current_yaw, pitch, roll = attitude_info

def scan_front_wall(ep_gimbal, ep_sensor):
    print("Scanning front wall type...")
    scan_distances = []
    scan_angles = [30, 15, 0, -15, -30]
    for angle in scan_angles:
        ep_gimbal.moveto(pitch=0, yaw=angle, pitch_speed=350, yaw_speed=350).wait_for_completed()
        time.sleep(0.4)
        distances = []
        for _ in range(3):
            if front_distance_cm is not None:
                distances.append(front_distance_cm)
            time.sleep(0.1)
        avg_distance = sum(distances) / len(distances) if distances else 200.0
        scan_distances.append(avg_distance)
        print(f"  Angle {angle}°: {avg_distance:.1f}cm")
    center_dist = scan_distances[2]
    left_dist = min(scan_distances[3], scan_distances[4])
    right_dist = min(scan_distances[0], scan_distances[1])
    print(f"Analysis: Center={center_dist:.1f}, Left={left_dist:.1f}, Right={right_dist:.1f}")
    wall_threshold = 38.0
    side_diff_threshold = 15.0
    if center_dist < wall_threshold and abs(left_dist - right_dist) < side_diff_threshold:
        wall_type = "FRONT_WALL"
        min_angle = 0
        print("FRONT WALL")
    elif left_dist < wall_threshold and (right_dist - left_dist) > side_diff_threshold:
        wall_type = "LEFT_CURVE"
        min_angle = -20
        print(f"LEFT CURVE: Left={left_dist:.1f}, Right={right_dist:.1f}")
    elif right_dist < wall_threshold and (left_dist - right_dist) > side_diff_threshold:
        wall_type = "RIGHT_CURVE"
        min_angle = 20
        print(f"RIGHT CURVE: Left={left_dist:.1f}, Right={right_dist:.1f}")
    elif left_dist < wall_threshold and right_dist < wall_threshold and center_dist > wall_threshold + 10:
        wall_type = "NARROW_PASSAGE"
        min_angle = 0
        print(f"NARROW PASSAGE: Center={center_dist:.1f}")
    elif center_dist < wall_threshold and left_dist < wall_threshold and right_dist < wall_threshold:
        wall_type = "DEAD_END"
        min_angle = 0
        print("DEAD END")
    else:
        wall_type = "FRONT_WALL"
        min_angle = 0
        print("UNCLEAR: Defaulting to FRONT_WALL")
    print(f"Final result: {wall_type} (suggested turn angle: {min_angle}°)")
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(0.5)
    return wall_type, min_angle

def check_right_path_before_turn(ep_gimbal, ep_sensor):
    print("Checking right path before turning...")
    ep_gimbal.moveto(pitch=0, yaw=90, pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(0.5)
    distances = []
    for _ in range(5):
        if front_distance_cm is not None:
            distances.append(front_distance_cm)
        time.sleep(0.15)
    if distances:
        avg_right_distance = sum(distances) / len(distances)
    else:
        avg_right_distance = 30.0
    print(f"Right path distance: {avg_right_distance:.1f}cm")
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(0.5)
    dead_end_threshold = 50.0
    is_dead_end = avg_right_distance < dead_end_threshold
    if is_dead_end:
        print(f"RIGHT PATH BLOCKED: {avg_right_distance:.1f}cm < {dead_end_threshold}cm")
    else:
        print(f"RIGHT PATH CLEAR: {avg_right_distance:.1f}cm")
    return not is_dead_end, avg_right_distance

def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis
    sensor_adaptor = ep_robot.sensor_adaptor
    ep_sensor = ep_robot.sensor
    ep_gimbal = ep_robot.gimbal
    ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
    ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
    ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=100).wait_for_completed()
    desired_dist_cm = 12.0
    forward_speed = 0.30
    stop_dist_cm = 30.0
    pid_wall = PID(kp=2.0, ki=0.01, kd=0.8, output_limit=50)
    pid_align = PID(kp=1.5, ki=0.0, kd=0.5, output_limit=50)
    ep_sensor.sub_distance(freq=20, callback=sub_tof_handler)
    ep_chassis.sub_attitude(freq=20, callback=sub_attitude_handler)
    time.sleep(1)
    initial_yaw = current_yaw
    print(f"Initial yaw: {initial_yaw:.1f}°")
    start_time = time.time()
    last_time = start_time
    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            adc_front = sensor_adaptor.get_adc(id=2, port=1)
            adc_back = sensor_adaptor.get_adc(id=1, port=2)
            dist_front = adc_to_distance(adc_front)
            dist_back = adc_to_distance(adc_back)
            avg_wall_distance = (dist_front + dist_back) / 2.0
            if front_distance_cm is not None and front_distance_cm <= stop_dist_cm:
                print(f"FRONT OBSTACLE DETECTED: {front_distance_cm:.1f} cm")
                ep_chassis.drive_speed(x=0, y=0, z=0)
                time.sleep(1.0)
                wall_type, suggested_angle = scan_front_wall(ep_gimbal, ep_sensor)
                if wall_type == "LEFT_CURVE":
                    ep_chassis.move(z=suggested_angle, z_speed=45).wait_for_completed()
                    time.sleep(0.5)
                elif wall_type == "RIGHT_CURVE":
                    ep_chassis.move(z=suggested_angle, z_speed=45).wait_for_completed()
                    time.sleep(0.5)
                elif wall_type == "NARROW_PASSAGE":
                    ep_chassis.drive_speed(x=0.1, y=0, z=0)
                    time.sleep(1.0)
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                elif wall_type == "DEAD_END":
                    ep_chassis.move(z=-180, z_speed=45).wait_for_completed()
                    time.sleep(1.0)
                else:
                    can_turn_right, right_distance = check_right_path_before_turn(ep_gimbal, ep_sensor)
                    if can_turn_right:
                        ep_chassis.drive_speed(x=0, y=0, z=0)
                        time.sleep(0.5)
                        ep_chassis.move(z=-90, z_speed=45).wait_for_completed()
                        time.sleep(1)
                    else:
                        ep_chassis.drive_speed(x=0, y=0, z=0)
                        break
                ep_gimbal.moveto(pitch=0, yaw=0, pitch_speed=100, yaw_speed=100)
                time.sleep(1)
                if wall_type != "FRONT_WALL" or can_turn_right:
                    if front_distance_cm is not None and front_distance_cm <= stop_dist_cm:
                        ep_chassis.drive_speed(x=0, y=0, z=0)
                        break
                    else:
                        continue
            wall_error = avg_wall_distance - desired_dist_cm
            align_error = dist_front - dist_back
            if abs(align_error) > 0.5:
                wall_error_port1 = dist_front - desired_dist_cm
                correction_wall = pid_wall.compute(-wall_error_port1, dt)
                curve_multiplier = 4
                if front_distance_cm is not None and front_distance_cm < 20.0:
                    curve_multiplier = 6
                correction_align = pid_align.compute(-align_error, dt) * curve_multiplier
                wz = correction_wall + correction_align
            else:
                correction_wall = pid_wall.compute(-wall_error, dt)
                correction_align = pid_align.compute(-align_error, dt)
                wz = correction_wall + correction_align
            max_wz = 40.0
            if front_distance_cm is not None and front_distance_cm < 20.0 and abs(align_error) > 0.5:
                max_wz = 60.0
            if abs(wz) > max_wz:
                wz = wz * 0.6
            ep_chassis.drive_speed(x=forward_speed, y=0, z=wz)
            print(f"P1:{dist_front:.1f} | P2:{dist_back:.1f} | Diff:{align_error:.1f} | Wz:{wz:.1f} | ChassisYaw:{current_yaw:.1f}°")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stop by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        total_time = time.time() - start_time
        print(f"Total Time: {total_time:.2f} s")
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        ep_sensor.unsub_distance()
        ep_chassis.unsub_attitude()
        ep_robot.close()
        print("Robot disconnected")

if __name__ == '__main__':
    main()