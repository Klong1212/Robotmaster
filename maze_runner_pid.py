import time
import threading
import math

import robomaster
from robomaster import robot, vision

# NOTE: matplotlib ใช้สำหรับวาดภาพแผนที่เมื่อรันบนคอมพิวเตอร์
# หากรันบนหุ่นยนต์โดยตรง โค้ดจะยังคงทำงานและเซฟไฟล์ PNG ได้
import matplotlib.pyplot as plt

# ==============================================================================
# การตั้งค่าและค่าคงที่ (Constants)
# ==============================================================================
GRID_SIZE_M = 0.6
WALL_THRESHOLD_MM = 500
VISION_SCAN_DURATION_S = 0.5
GIMBAL_TURN_SPEED = 200

ORIENTATIONS = {0: "North", 1: "East", 2: "South", 3: "West"}
WALL_NAMES = {0: "North Wall", 1: "East Wall", 2: "South Wall", 3: "Westก Wall"}

# ==============================================================================
# คลาสจัดการข้อมูลและแผนที่ (Data Handlers & Map)
# ==============================================================================
class TofDataHandler:
    def __init__(self):
        self.distance = 0
        self._lock = threading.Lock()
    def update(self, sub_info):
        with self._lock:
            self.distance = sub_info[0]
    def get_distance(self):
        with self._lock:
            return self.distance

class VisionDataHandler:
    def __init__(self):
        self.markers = []
        self._lock = threading.Lock()
    def update(self, vision_info):
        with self._lock:
            self.markers = [info[0] for info in vision_info[1:]] if vision_info[0] > 0 else []
    def get_markers(self):
        with self._lock:
            return list(self.markers)
    def clear(self):
        with self._lock:
            self.markers = []

class PoseDataHandler:
    def __init__(self):
        self.pose = [0.0] * 6
        self._lock = threading.Lock()
    def update_position(self, pos_info):
        with self._lock:
            self.pose[0], self.pose[1], self.pose[2] = pos_info[0], pos_info[1], pos_info[2]
    def update_attitude(self, att_info):
        with self._lock:
            self.pose[3], self.pose[4], self.pose[5] = att_info[0], att_info[1], att_info[2]
    def get_pose(self):
        with self._lock:
            return tuple(self.pose)
    def set_xy(self, x_m, y_m):
        with self._lock:
            self.pose[0], self.pose[1] = float(x_m), float(y_m)
    def set_yaw(self, yaw_deg):
        with self._lock:
            self.pose[3] = float(yaw_deg)

class RobotMap:
    def __init__(self):
        self.graph = {}
        self.explored = set()
    def add_connection(self, pos1, pos2):
        if pos1 not in self.graph: self.graph[pos1] = set()
        if pos2 not in self.graph: self.graph[pos2] = set()
        if pos2 not in self.graph[pos1]:
            self.graph[pos1].add(pos2)
            self.graph[pos2].add(pos1)
            print(f"      Map: Added connection between {pos1} and {pos2}")
    def mark_explored(self, position):
        self.explored.add(position)
    def get_unexplored_neighbors(self, position):
        if position not in self.graph: return []
        return [n for n in self.graph.get(position, []) if n not in self.explored]
    def get_path(self, start, goal):
        if start == goal: return [start]
        queue = [(start, [start])]
        visited = {start}
        while queue:
            current, path = queue.pop(0)
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    if neighbor == goal: return path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-1.0, 1.0)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()
    def update(self, current_value):
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0: return 0.0
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._previous_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        if self.output_limits:
            output = max(self.output_limits[0], min(self.output_limits[1], output))
        self._previous_error = error
        self._last_time = current_time
        return output

# ==============================================================================
# คลาสหลักสำหรับควบคุมตรรกะของหุ่นยนต์ (MazeExplorer)
# ==============================================================================
class MazeExplorer:
    def __init__(self, ep_robot, tof_handler, vision_handler, pose_handler):
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_led = ep_robot.led
        self.ep_vision = ep_robot.vision
        self.ep_gimbal = ep_robot.gimbal
        self.tof_handler = tof_handler
        self.vision_handler = vision_handler
        self.pose_handler = pose_handler
        self.current_position = (0, 0)
        self.current_orientation = 0 # 0:N, 1:E, 2:S, 3:W
        self.internal_map = RobotMap()
        self.marker_map = {}
        self.visited_path = [self.current_position]
        print("Robot Explorer Initialized. Starting at (0,0), facing North.")
        self.ep_led.set_led(r=0, g=0, b=255)
        self.ep_gimbal.recenter().wait_for_completed()
        # Reset pose at the beginning
        self.pose_handler.set_xy(0.0, 0.0)
        self.pose_handler.set_yaw(0.0)

    # <--- แก้ไข: ฟังก์ชันนี้จะคืนค่าระยะทางที่วัดได้ ---
    def scan_surroundings_with_gimbal(self, previous_position=None):
        print(f"\nScanning surroundings at {self.current_position} with Gimbal...")
        self.ep_led.set_led(r=255, g=255, b=0, effect="breathing")
        self.internal_map.mark_explored(self.current_position)
        x, y = self.current_position
        
        wall_distances = {} # <--- เพิ่ม: สร้าง dict เพื่อเก็บระยะทาง

        direction_to_skip = -1
        if previous_position:
            print(f"   -> Path from {previous_position} is known. Adding connection automatically.")
            self.internal_map.add_connection(self.current_position, previous_position)
            direction_to_skip = (self.current_orientation + 2) % 4
            print(f"   -> Will skip physical scan for direction {ORIENTATIONS.get(direction_to_skip, 'N/A')}.")

        for scan_direction in range(4):
            if scan_direction == direction_to_skip:
                continue

            neighbor_pos = None
            if scan_direction == 0: neighbor_pos = (x, y + 1)
            elif scan_direction == 1: neighbor_pos = (x + 1, y)
            elif scan_direction == 2: neighbor_pos = (x, y - 1)
            elif scan_direction == 3: neighbor_pos = (x - 1, y)

            if neighbor_pos in self.internal_map.explored:
                print(f"   -> Neighbor {neighbor_pos} ({ORIENTATIONS[scan_direction]}) already explored. Inferring from map, skipping physical scan.")
                continue

            print(f"   Scanning new area in direction: {ORIENTATIONS[scan_direction]}...")
            angle_to_turn_gimbal = (scan_direction - self.current_orientation) * 90
            if angle_to_turn_gimbal > 180: angle_to_turn_gimbal -= 360
            if angle_to_turn_gimbal < -180: angle_to_turn_gimbal += 360

            self.ep_gimbal.moveto(yaw=angle_to_turn_gimbal, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(0.5)
            distance_mm = self.tof_handler.get_distance()
            print(f"         - ToF distance: {distance_mm} mm")
            
            wall_distances[f'{scan_direction}'] = distance_mm # <--- เพิ่ม: เก็บค่าที่วัดได้

            if distance_mm >= WALL_THRESHOLD_MM:
                self.internal_map.add_connection(self.current_position, neighbor_pos)

            self.vision_handler.clear()
            time.sleep(VISION_SCAN_DURATION_S)
            detected_markers = self.vision_handler.get_markers()
            print(detected_markers)
            if detected_markers and distance_mm < WALL_THRESHOLD_MM:
                for marker_name in detected_markers:
                    wall_name = WALL_NAMES.get(scan_direction, "Unknown")
                    finding = (self.current_position, wall_name)
                    if marker_name not in self.marker_map: self.marker_map[marker_name] = []
                    if finding not in self.marker_map[marker_name]:
                        self.marker_map[marker_name].append(finding)
                        print(f"         !!! Marker Found & Logged: '{marker_name}' at Grid {finding[0]} on the {finding[1]} !!!")

        self.ep_gimbal.recenter().wait_for_completed()
        print("Scan complete. Gimbal recentered.")
        return wall_distances # <--- แก้ไข: คืนค่า dict

    def decide_next_path(self):
        # (ฟังก์ชันนี้ไม่มีการเปลี่ยนแปลง)
        unexplored = self.internal_map.get_unexplored_neighbors(self.current_position)
        if unexplored:
            return [self.current_position, unexplored[0]]
        for pos in reversed(self.visited_path):
            if self.internal_map.get_unexplored_neighbors(pos):
                print(f"No new paths here. Backtracking to find an unexplored path from {pos}...")
                return self.internal_map.get_path(self.current_position, pos)
        return None

    def move_forward_pid(self, distance_m, speed_limit=2.5):
        # (ฟังก์ชันนี้ไม่มีการเปลี่ยนแปลง)
        print(f"   PID Move: Moving forward {distance_m}m.")
        pid = PIDController(Kp=2.5, Ki=0.1, Kd=0.8, setpoint=distance_m, output_limits=(-speed_limit, speed_limit))
        start_x, start_y, _, _, _, _ = self.pose_handler.get_pose()
        while True:
            curr_x, curr_y, _, _, _, _ = self.pose_handler.get_pose()
            dist_traveled = math.hypot(curr_x - start_x, curr_y - start_y)
            if abs(distance_m - dist_traveled) < 0.01: break
            vx_speed = pid.update(dist_traveled)
            self.ep_chassis.drive_speed(x=vx_speed, y=0, z=0, timeout=0.1)
            time.sleep(0.01)
        self.ep_chassis.drive_speed(0, 0, 0)
        print("   PID Move: Completed.")


    def turn_pid(self, target_angle, speed_limit=180):
        # (ฟังก์ชันนี้ไม่มีการเปลี่ยนแปลง)
        print(f"   PID Turn: Turning to {target_angle} degrees.")
        pid = PIDController(Kp=1.8, Ki=0.1, Kd=0.8, setpoint=0, output_limits=(-speed_limit, speed_limit))
        while True:
            
            _, _, _, current_yaw, _, _ = self.pose_handler.get_pose()
            error = target_angle - current_yaw
            print(current_yaw)
            if error > 180: error -= 360
            if error < -180: error += 360
            if abs(error) < 0.5: break
            vz_speed = pid.update(-error)
            self.ep_chassis.drive_speed(x=0, y=0, z=vz_speed, timeout=0.1)
            time.sleep(0.01)
        self.ep_gimbal.recenter().wait_for_completed()
        self.ep_chassis.drive_speed(0, 0, 0)
        print("   PID Turn: Completed.")
    def recenter_robot(self, wall_distances):
        print("   -> Recenter using ToF data...")
        x_offset_mm = 0
        y_offset_mm = 0
        
        # คำนวณค่า offset แกน X (East-West)
        if '1' in wall_distances and '3' in wall_distances: # 1:East, 3:West
            dist_e = wall_distances['1']
            dist_w = wall_distances['3']
            # ค่าผิดพลาดคือครึ่งหนึ่งของผลต่างระยะทาง
            error_mm = (dist_w - dist_e) / 2.0
            # ต้องเคลื่อนที่ไปในทิศทางของแกน Y ของหุ่นยนต์ เพื่อแก้แกน X ของโลก
            # (เมื่อหุ่นยนต์หันหน้าทิศเหนือ (0 deg))
            if self.current_orientation == 0:
                y_offset_mm = error_mm
            elif self.current_orientation == 2:
                y_offset_mm = -error_mm
            print(f"      X-Axis Correction: Move by {y_offset_mm:.1f} mm")

        # คำนวณค่า offset แกน Y (North-South)
        if '0' in wall_distances and '2' in wall_distances: # 0:North, 2:South
            dist_n = wall_distances['0']
            dist_s = wall_distances['2']
            error_mm = (dist_s - dist_n) / 2.0
            # ต้องเคลื่อนที่ไปในทิศทางของแกน X ของหุ่นยนต์ เพื่อแก้แกน Y ของโลก
            # (เมื่อหุ่นยนต์หันหน้าทิศเหนือ (0 deg))
            if self.current_orientation == 0:
                x_offset_mm = error_mm
            elif self.current_orientation == 2:
                x_offset_mm = -error_mm
            print(f"      Y-Axis Correction: Move by {x_offset_mm:.1f} mm")

        # ถ้ามีค่า offset ที่ต้องแก้ไข ให้เคลื่อนที่เล็กน้อย
        # แปลงจาก mm เป็น m
        x_move_m = x_offset_mm / 1000.0
        y_move_m = y_offset_mm / 1000.0
        
        if abs(x_move_m) > 0.005 or abs(y_move_m) > 0.005:
            print(f"   Applying offset movement: x={x_move_m:.3f}m, y={y_move_m:.3f}m")
            self.ep_chassis.move(x=x_move_m, y=y_move_m, z=0, xy_speed=0.2).wait_for_completed()
            time.sleep(0.5)
        else:
            print("   Robot is well-centered. No adjustment needed.")
    def execute_path(self, path):
        # (ฟังก์ชันนี้ไม่มีการเปลี่ยนแปลง)
        if not path or len(path) < 2: return
        print(f"Executing path with PID: {path}")
        self.ep_led.set_led(r=0, g=0, b=255)
        for i in range(len(path) - 1):
            start_node, end_node = path[i], path[i+1]
            dx, dy = end_node[0] - start_node[0], end_node[1] - start_node[1]
        
            target_orientation = -1
            if dx == 0 and dy == 1: target_orientation = 0
            elif dx == 1 and dy == 0: target_orientation = 1
            elif dx == 0 and dy == -1: target_orientation = 2
            elif dx == -1 and dy == 0: target_orientation = 3

            target_angle = 0
            if target_orientation == 1: target_angle = 90
            elif target_orientation == 2: target_angle = 180
            elif target_orientation == 3: target_angle = -90

            self.turn_pid(target_angle)
            self.current_orientation = target_orientation
            time.sleep(0.2)
            self.move_forward_pid(GRID_SIZE_M)
            self.current_position = end_node
            self.visited_path.append(self.current_position)

            self.pose_handler.set_xy(end_node[0] * GRID_SIZE_M, end_node[1] * GRID_SIZE_M)
            self.pose_handler.set_yaw(target_angle)
            time.sleep(0.2)
    # เพิ่มฟังก์ชันนี้เข้าไปในคลาส MazeExplorer
    def correct_orientation_with_wall(self, wall_distances):
        """
        ปรับมุมของหุ่นยนต์ให้ตั้งฉากกับกำแพงด้านหน้าโดยใช้ ToF sensor
        """
        # ทิศทางด้านหน้าของหุ่นยนต์ ณ ปัจจุบัน (0:N, 1:E, 2:S, 3:W)
        front_direction = self.current_orientation
        
        # ตรวจสอบว่ามีกำแพงอยู่ด้านหน้าหรือไม่ ถ้าไม่มีก็ไม่ต้องทำอะไร
        if str(front_direction) not in wall_distances or wall_distances[str(front_direction)] >= WALL_THRESHOLD_MM:
            print("   -> Orientation Correction: No wall in front to calibrate with.")
            return

        print("   -> Correcting orientation using front wall...")
        self.ep_led.set_led(r=139, g=0, b=255, effect="breathing") # สีม่วงเพื่อบ่งบอกว่ากำลังปรับมุม

        # พารามิเตอร์การกวาด
        SWEEP_ANGLE_DEG = 5  # กวาด +/- 5 องศาจากตำแหน่งปัจจุบัน
        SWEEP_SPEED_DPS = 20 # ความเร็วในการกวาด (องศาต่อวินาที)

        min_dist_mm = float('inf')
        angle_at_min_dist = 0
        
        # เริ่มกวาดจากซ้ายไปขวา
        self.ep_gimbal.moveto(yaw=-SWEEP_ANGLE_DEG, pitch=0, yaw_speed=SWEEP_SPEED_DPS).wait_for_completed()
        time.sleep(0.5)
        
        # เริ่มการกวาดจริง
        self.ep_gimbal.moveto(yaw=SWEEP_ANGLE_DEG, pitch=0, yaw_speed=SWEEP_SPEED_DPS).wait_for_start()

        start_time = time.time()
        while time.time() - start_time < (2 * SWEEP_ANGLE_DEG / SWEEP_SPEED_DPS) + 0.5:
            try:
                # อ่านค่ามุมของ gimbal และระยะทางจาก ToF
                gimbal_yaw, _ = self.ep_gimbal.get_attitude(mode="relative_to_chassis")
                current_dist = self.tof_handler.get_distance()
                
                if current_dist < min_dist_mm:
                    min_dist_mm = current_dist
                    angle_at_min_dist = gimbal_yaw # มุมที่ให้ระยะทางสั้นที่สุด
                
                time.sleep(0.01)
            except Exception as e:
                # อาจเกิดข้อผิดพลาดระหว่าง gimbal กำลังเคลื่อนที่
                pass

        self.ep_gimbal.recenter().wait_for_completed()
        
        # มุมที่ผิดพลาดคือมุมที่ gimbal ต้องหันไปเพื่อให้ตั้งฉาก
        error_angle = angle_at_min_dist
        print(f"      - Min distance {min_dist_mm}mm found at gimbal angle {error_angle:.2f}°")
        
        # ถ้าค่าผิดพลาดน้อยมาก ก็ไม่ต้องแก้ไข
        if abs(error_angle) < 0.5:
            print("      - Orientation is acceptable. No correction needed.")
            self.ep_led.set_led(r=0, g=0, b=255) # กลับไปสีน้ำเงินปกติ
            return
            
        # คำนวณมุมเป้าหมายใหม่ของ Chassis
        _, _, _, current_chassis_yaw, _, _ = self.pose_handler.get_pose()
        correction_angle = current_chassis_yaw + error_angle
        
        # ทำให้มุมอยู่ในช่วง -180 ถึง 180
        if correction_angle > 180: correction_angle -= 360
        if correction_angle < -180: correction_angle += 360

        print(f"      - Chassis yaw error is {error_angle:.2f}°. Correcting...")
        
        # ใช้ PID turn เพื่อหมุนแก้ไข
        self.turn_pid(correction_angle)

        # *** ขั้นตอนสำคัญ: รีเซ็ตค่า Yaw ใหม่หลังจากแก้ไขแล้ว ***
        # เราต้องบอกหุ่นยนต์ว่า "ตอนนี้แหละคือมุมที่ถูกต้อง"
        # เช่น ถ้าก่อนหน้านี้หุ่นยนต์หันไปทิศเหนือ (0) เราก็จะรีเซ็ตค่า yaw เป็น 0
        target_yaw = 0
        if self.current_orientation == 1: target_yaw = 90
        elif self.current_orientation == 2: target_yaw = 180
        elif self.current_orientation == 3: target_yaw = -90
        
        self.pose_handler.set_yaw(target_yaw)
        print(f"      - Orientation corrected. Yaw reset to {target_yaw}°.")
        self.ep_led.set_led(r=0, g=0, b=255) # กลับไปสีน้ำเงินปกติ
    # <--- แก้ไข: ปรับปรุงลำดับการทำงานใน `run_mission` ---
    def run_mission(self):
        start_time = time.time()
        time_limit_seconds = 600
        print(f"Mission started! Time limit: {time_limit_seconds} seconds.")

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit_seconds:
                print(f"\n--- TIME'S UP! ({int(elapsed_time)}s elapsed) ---")
                self.ep_led.set_led(r=255, g=193, b=7, effect="flash")
                break

            if self.current_position not in self.internal_map.explored:
                previous_pos = self.visited_path[-2] if len(self.visited_path) > 1 else None
                
                # Step 1: สแกนและรับค่าระยะทางกลับมา
                wall_distances=self.scan_surroundings_with_gimbal(previous_position=previous_pos)
                
                # Step 2: จัดตำแหน่งกลางโดยใช้ข้อมูลจากการสแกน
            
                self.recenter_robot(wall_distances)
                self.correct_orientation_with_wall(wall_distances)
            else:
                print(f"\nPosition {self.current_position} already explored. Skipping scan.")
            # Step 3: ตัดสินใจเลือกเส้นทางต่อไป
            path_to_execute = self.decide_next_path()

            if not path_to_execute:
                print("\n--- MISSION COMPLETE! All areas explored. ---")
                self.ep_led.set_led(r=0, g=255, b=0, effect="on")
                break

            # Step 4: เดินทางตามเส้นทาง
            self.execute_path(path_to_execute)
            
        print("\n--- Final Marker Map ---")
        if self.marker_map:
            for name, findings in sorted(self.marker_map.items()):
                print(f"   Marker '{name}':")
                for details in findings:
                    print(f"         - Found at Grid={details[0]}, Wall={details[1]}")
        else:
            print("   No markers were logged.")

        plot_map_and_path(self.internal_map.graph, self.visited_path)

def plot_map_and_path(graph, visited_path, filename='maze_map_pid.png'):
    plt.figure(figsize=(8, 8))
    plt.title('Robot Map & Traversed Path (PID)')
    for node, neighbors in graph.items():
        x1, y1 = node
        for nb in neighbors:
            x2, y2 = nb
            plt.plot([x1, x2], [y1, y2], color='lightblue', zorder=1)
    if not graph:
        print("Plotting skipped: Map is empty.")
    else:
        xs, ys = [n[0] for n in graph.keys()], [n[1] for n in graph.keys()]
        plt.scatter(xs, ys, s=50, color='blue', zorder=2, label='Map Nodes')
    if visited_path:
        px, py = [p[0] for p in visited_path], [p[1] for p in visited_path]
        plt.plot(px, py, color='red', linewidth=2, zorder=3, label='Robot Path')
        plt.scatter(px[0], py[0], s=150, color='green', marker='o', zorder=4, label='Start')
        plt.scatter(px[-1], py[-1], s=150, color='purple', marker='X', zorder=4, label='End')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, dpi=150)
    print(f"Map saved to '{filename}'")
    plt.close()

# ==============================================================================
# ส่วนหลักของโปรแกรม (Main Execution)
# ==============================================================================
if __name__ == '__main__':
    ep_robot = None
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        print("Robot connected.")
        tof_handler = TofDataHandler()
        vision_handler = VisionDataHandler()
        pose_handler = PoseDataHandler()
        ep_robot.reset_robot_mode()
        ep_robot.sensor.sub_distance(freq=10, callback=tof_handler.update)
        ep_robot.vision.sub_detect_info(name="marker", callback=vision_handler.update)
        ep_robot.chassis.sub_position(freq=20, callback=pose_handler.update_position)
        ep_robot.chassis.sub_attitude(freq=20, callback=pose_handler.update_attitude)

        print("Subscribed to all required sensors.")
        time.sleep(2)
        explorer = MazeExplorer(ep_robot, tof_handler, vision_handler, pose_handler)
        explorer.run_mission()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ep_robot:
            ep_robot.sensor.unsub_distance()
            ep_robot.vision.unsub_detect_info(name="marker")
            ep_robot.chassis.unsub_position()
            ep_robot.chassis.unsub_attitude()
            ep_robot.close()
            print("break")