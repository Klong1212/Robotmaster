import time
import threading
import math
import csv
from datetime import datetime
from collections import deque  # <<< เพิ่ม import นี้ไว้ด้านบน
import robomaster
from robomaster import robot, vision

# NOTE: matplotlib ใช้สำหรับวาดภาพแผนที่เมื่อรันบนคอมพิวเตอร์
# หากรันบนหุ่นยนต์โดยตรง โค้ดจะยังคงทำงานและเซฟไฟล์ PNG ได้
import matplotlib.pyplot as plt
import statistics
# ==============================================================================
# การตั้งค่าและค่าคงที่ (Constants)
# ==============================================================================
GRID_SIZE_M = 0.635
WALL_THRESHOLD_MM = 500
VISION_SCAN_DURATION_S = 1
GIMBAL_TURN_SPEED = 450

ORIENTATIONS = {0: "North", 1: "East", 2: "South", 3: "West"}
WALL_NAMES = {0: "North Wall", 1: "East Wall", 2: "South Wall", 3: "West Wall"}  # <<< CHANGED (พิมพ์ตก)

# ==============================================================================
# คลาสจัดการข้อมูลและแผนที่ (Data Handlers & Map)
# ==============================================================================
class TofDataHandler:
    def __init__(self, window_size=5):
        self._lock = threading.Lock()
        self.raw_distance = 0.0            # ค่าดิบล่าสุด (mm)
        self._window = deque(maxlen=window_size)
        self._median = 0.0                 # ค่า median ปัจจุบัน

    def update(self, sub_info):
        d = float(sub_info[0]) if sub_info else 0.0
        with self._lock:
            self.raw_distance = d
            self._window.append(d)
            self._median = statistics.median(self._window)

    def get_distance(self):
        # คืนค่าแบบกรองแล้ว (median)
        with self._lock:
            return self._median if self._window else self.raw_distance

    def get_raw_distance(self):
        # ถ้าต้องการค่า ToF ดิบ
        with self._lock:
            return self.raw_distance

class VisionDataHandler:
    def __init__(self):
        self.markers = []
        self._lock = threading.Lock()
        self._sample_logged = False  # log โครงสร้างครั้งแรกครั้งเดียว

    def update(self, vision_info):
        with self._lock:
            # vision_info เป็น list ของ detection tuples
            # รูปแบบพบบ่อย: (x, y, w, h, label) หรือ (x, y, w, h, label, ... )
            self.markers.clear()
            if vision_info:
                if not self._sample_logged:
                    print("[Vision raw] ->", vision_info)
                    self._sample_logged = True
                for t in vision_info:
                    if not isinstance(t, (list, tuple)) or len(t) < 5:
                        continue
                    label = t[4]
                    # เผื่อบางเวอร์ชันส่งเป็น int id หรือ str ชื่อ
                self.markers.append(str(label))

    def get_markers(self):
        with self._lock:
            return list(self.markers)

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
        self.blocked = set()
    def add_connection(self, pos1, pos2):
        if pos1 not in self.graph: self.graph[pos1] = set()
        if pos2 not in self.graph: self.graph[pos2] = set()
        if pos2 not in self.graph[pos1]:
            self.graph[pos1].add(pos2)
            self.graph[pos2].add(pos1)

    def add_blocked(self, pos1, pos2):
        edge = tuple(sorted([pos1, pos2]))
        self.blocked.add(edge)
    def mark_explored(self, position):
        self.explored.add(position)
    def get_unexplored_neighbors(self, position):
        if position not in self.graph: return []
        return [n for n in self.graph.get(position, []) if n not in self.explored]
    def get_path(self, start, goal):
        if start == goal: return [start]
        queue = [(start, [start])]
        visited = {start}
        print("queue: ",queue)
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
        self.step_counter = 0  # <<< นับจำนวนช่องที่เดิน เพื่อ trig ทุกๆ 2 ช่อง
        self.ep_led.set_led(r=0, g=0, b=255)
        
        # Reset pose at the beginning
        self.pose_handler.set_xy(0.0, 0.0)
        self.pose_handler.set_yaw(0.0)

        # --- Logs for CSV (เดิม) ---
        self.scan_log = []      # {'ts','grid_x','grid_y','N_mm','E_mm','S_mm','W_mm'}
        self.wall_log = set()
        self.marker_log = []    # {'ts','name','grid_x','grid_y','wall'}
        self.path_log = []      # {'ts','grid_x','grid_y','yaw'}

        # --- เพิ่ม: บันทึกเซ็นเซอร์แบบต่อเนื่อง (Continuous Logging) ---
        self.continuous_sensor_log = []   # เก็บ dict ต่อเนื่อง
        self._logging_thread = None
        self._logging_thread_active = False

    # -------------------- Continuous Sensor Logging (NEW) --------------------
    def _continuous_log_worker(self, frequency_hz=10):
        print(f"[Logging Thread] Started. Logging at {frequency_hz} Hz.")
        period = 1.0 / max(1, frequency_hz)
        while self._logging_thread_active:
            t0 = time.time()
            try:
                x, y, z, yaw, pitch, roll = self.pose_handler.get_pose()
                tof = self.tof_handler.get_distance()
                markers = self.vision_handler.get_markers()
                self.continuous_sensor_log.append({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "pos_x_m": x, "pos_y_m": y, "pos_z_m": z,
                    "att_yaw_deg": yaw, "att_pitch_deg": pitch, "att_roll_deg": roll,
                    "tof_distance_mm": tof,
                    "detected_markers": ",".join(markers) if markers else ""
                })
            except Exception as e:
                print(f"[Logging Thread] Error: {e}")
            # คุมความถี่
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
        print("[Logging Thread] Stopped.")

    def start_continuous_logging(self, frequency_hz=10):
        if not self._logging_thread_active:
            self._logging_thread_active = True
            self._logging_thread = threading.Thread(
                target=self._continuous_log_worker, kwargs={"frequency_hz": frequency_hz}, daemon=True
            )
            self._logging_thread.start()

    def stop_continuous_logging(self):
        if self._logging_thread_active:
            self._logging_thread_active = False
            if self._logging_thread:
                self._logging_thread.join(timeout=2.0)

    # ------------------------------------------------------------------------
    # บันทึก CSV (ขยายให้รวม continuous_sensor_log ด้วย)
    def save_csv_logs(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) continuous sensor log
        if self.continuous_sensor_log:
            try:
                with open(f"continuous_sensor_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                    fieldnames = ["timestamp", "pos_x_m", "pos_y_m", "pos_z_m",
                                  "att_yaw_deg", "att_pitch_deg", "att_roll_deg",
                                  "tof_distance_mm", "detected_markers"]
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(self.continuous_sensor_log)
                print(f"[CSV] continuous_sensor_log_{ts}.csv saved")
            except Exception as e:
                print(f"[CSV] Error saving continuous log: {e}")

        # 2) scan per grid
        if self.scan_log:
            with open(f"scan_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["ts","grid_x","grid_y","N_mm","E_mm","S_mm","W_mm"])
                w.writeheader()
                for r in self.scan_log: w.writerow(r)

        # 3) closed walls
        if self.wall_log:
            with open(f"walls_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["cell1_x","cell1_y","cell2_x","cell2_y"])
                for (a,b) in sorted(self.wall_log):
                    w.writerow([a[0],a[1],b[0],b[1]])

        # 4) markers
        if self.marker_log:
            with open(f"marker_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["ts","name","grid_x","grid_y","wall"])
                w.writeheader()
                for r in self.marker_log: w.writerow(r)

        # 5) visited path
        if self.visited_path:
            with open(f"path_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["step","grid_x","grid_y"])
                for i, (gx,gy) in enumerate(self.visited_path):
                    w.writerow([i,gx,gy])

    # ==================== จากนี้ “ตรรกะ/การเคลื่อนที่เดิม” ====================
    # <--- ฟังก์ชันนี้จะคืนค่าระยะทางที่วัดได้ --->
    def scan_surroundings_with_gimbal(self, previous_position=None):
        print(f"\nScanning surroundings at {self.current_position} with Gimbal...")
        self.ep_led.set_led(r=255, g=255, b=0, effect="breathing")
        self.internal_map.mark_explored(self.current_position)
        x, y = self.current_position
        
        wall_distances = {} # เก็บระยะทาง

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

            print(f"   Scanning new area in direction: {ORIENTATIONS[scan_direction]}...")
            angle_to_turn_gimbal = (scan_direction - self.current_orientation) * 90
            if angle_to_turn_gimbal > 180: angle_to_turn_gimbal -= 360
            if angle_to_turn_gimbal < -180: angle_to_turn_gimbal += 360

            self.ep_gimbal.moveto(yaw=angle_to_turn_gimbal, pitch=0, yaw_speed=GIMBAL_TURN_SPEED,pitch_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(0.5)  # ให้เวลาหุ่นยนต์ปรับตำแหน่งก่อนวัด
            self.ep_gimbal.moveto(yaw=angle_to_turn_gimbal, pitch=-15, yaw_speed=GIMBAL_TURN_SPEED,pitch_speed=GIMBAL_TURN_SPEED).wait_for_completed()


            time.sleep(0.5)
            distance_mm = self.tof_handler.get_distance()
            print(f"         - ToF distance: {distance_mm} mm")
            wall_distances[f'{scan_direction}'] = distance_mm

            if distance_mm >= WALL_THRESHOLD_MM:
                self.internal_map.add_connection(self.current_position, neighbor_pos)
            else:
                print(f"           - Wall detected at {distance_mm}mm. Preparing to scan for markers.")
                # --- เตรียมเข้าใกล้เพื่อสแกน แล้วกลับ ---
                move_dist_m = (distance_mm / 1000.0) - 0.20
                move_dist_m_y = (distance_mm / 1000.0) - 0.20
                relative_direction = (scan_direction - self.current_orientation + 4) % 4
                self.internal_map.add_blocked(self.current_position, neighbor_pos)
                self.wall_log.add(tuple(sorted([self.current_position, neighbor_pos])))  # <<< เก็บสำหรับ CSV

                move_x, move_y = 0.0, 0.0
                if relative_direction == 0:   # หน้า
                    move_x = move_dist_m_y
                elif relative_direction == 1:  # ขวา
                    move_y = move_dist_m      # y+ = ซ้าย, y- = ขวา (ตาม SDK)
                elif relative_direction == 2:  # หลัง
                    move_x = -move_dist_m_y
                elif relative_direction == 3:  # ซ้าย
                    move_y = -move_dist_m

                if abs(move_dist_m) > 0.01:
                    print(f"             Adjusting position: move x={move_x:.2f}m, y={move_y:.2f}m.")
                    self.ep_chassis.move(x=move_x, y=move_y, z=0, xy_speed=1.5).wait_for_completed()
                    
                else:
                    print("             Position is good, no adjustment needed for scan.")
                time.sleep(0.2)

                for i in range(0, 60, 10):
                    time.sleep(0.2)
                    self.ep_gimbal.moveto(yaw=angle_to_turn_gimbal-30+i, pitch=-15, yaw_speed=GIMBAL_TURN_SPEED,pitch_speed=GIMBAL_TURN_SPEED).wait_for_completed()
                    dm = self.vision_handler.get_markers()
                    print(dm)
                    if dm:
                        detected_markers = dm
                        break

                if detected_markers and distance_mm < 400:
                    print(f"             Markers detected: {detected_markers}")
                    for marker_name in detected_markers:
                        wall_name = WALL_NAMES.get(scan_direction, "Unknown Wall")
                        finding = (self.current_position, wall_name)
                        if marker_name not in self.marker_map: self.marker_map[marker_name] = []
                        if finding not in self.marker_map[marker_name]:
                            self.marker_map[marker_name].append(finding)
                            self.marker_log.append({
                                "ts": datetime.now().isoformat(timespec="seconds"),
                                "name": marker_name,
                                "grid_x": self.current_position[0],
                                "grid_y": self.current_position[1],
                                "wall": wall_name
                            })
                            print(f"             !!! Marker LOGGED: '{marker_name}' at Grid {finding[0]} on the {finding[1]} !!!")

        self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        print("Scan complete. Gimbal recentered.")

        # เก็บ scan_log ต่อกริด
        def pick(d, k): 
            return d.get(k, None)
        self.scan_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "grid_x": x, "grid_y": y,
            "N_mm": pick(wall_distances, '0'),
            "E_mm": pick(wall_distances, '1'),
            "S_mm": pick(wall_distances, '2'),
            "W_mm": pick(wall_distances, '3')
        })

        return wall_distances

    def decide_next_path(self):
        unexplored = self.internal_map.get_unexplored_neighbors(self.current_position)
        if unexplored:
            return [self.current_position, unexplored[0]]
        for pos in reversed(self.visited_path):
            if self.internal_map.get_unexplored_neighbors(pos):
                print(f"No new paths here. Backtracking to find an unexplored path from {pos}...")

                return self.internal_map.get_path(self.current_position, pos)
        return None

    def periodic_wall_clearance_adjust(self, target_clearance_m=0.18):
        """
        ใช้ ToF ด้านหน้าตาม orientation ปัจจุบัน ถ้ามีกำแพง (dist < threshold)
        จะขยับเข้า/ออกให้เข้าใกล้ target_clearance_m
        """
        
                    
        for scan_direction in range(4):



            print(f"   Scanning new area in direction: {ORIENTATIONS[scan_direction]}...")
            angle_to_turn_gimbal = (scan_direction - self.current_orientation) * 90
            if angle_to_turn_gimbal > 180: angle_to_turn_gimbal -= 360
            if angle_to_turn_gimbal < -180: angle_to_turn_gimbal += 360

            self.ep_gimbal.moveto(yaw=angle_to_turn_gimbal, pitch=0, yaw_speed=GIMBAL_TURN_SPEED,pitch_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(0.5)  # ให้เวลาหุ่นยนต์ปรับตำแหน่งก่อนวัด
            self.ep_gimbal.moveto(yaw=angle_to_turn_gimbal, pitch=-15, yaw_speed=GIMBAL_TURN_SPEED,pitch_speed=GIMBAL_TURN_SPEED).wait_for_completed()


            time.sleep(0.5)
            distance_mm = self.tof_handler.get_distance()
            print(f"         - ToF distance: {distance_mm} mm")

            if distance_mm < WALL_THRESHOLD_MM:
                print(f"           - Wall detected at {distance_mm}mm. Preparing to scan for markers.")
                # --- เตรียมเข้าใกล้เพื่อสแกน แล้วกลับ ---
                move_dist_m = (distance_mm / 1000.0) - 0.18
                move_dist_m_y = (distance_mm / 1000.0) - 0.18
                relative_direction = (scan_direction - self.current_orientation + 4) % 4

                move_x, move_y = 0.0, 0.0
                if relative_direction == 0:   # หน้า
                    move_x = move_dist_m_y
                elif relative_direction == 1:  # ขวา
                    move_y = move_dist_m      # y+ = ซ้าย, y- = ขวา (ตาม SDK)
                elif relative_direction == 2:  # หลัง
                    move_x = -move_dist_m_y
                elif relative_direction == 3:  # ซ้าย
                    move_y = -move_dist_m

                if abs(move_dist_m) > 0.05:
                    print(f"             Adjusting position: move x={move_x:.2f}m, y={move_y:.2f}m.")
                    self.ep_chassis.move(x=move_x, y=move_y, z=0, xy_speed=0.3).wait_for_completed()
                    
                else:
                    print("             Position is good, no adjustment needed for scan.")
                time.sleep(0.2)


                self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                
    def move_forward_pid(self, distance_m, speed_limit=5):
        print(f"   PID Move: Moving forward {distance_m}m.")
        pid = PIDController(Kp=2, Ki=0.2, Kd=0.25, setpoint=distance_m, output_limits=(-speed_limit, speed_limit))
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

        # log path step
        gx, gy = self.current_position
        _, _, _, yaw, _, _ = self.pose_handler.get_pose()
        self.path_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "grid_x": gx, "grid_y": gy, "yaw": yaw
        })

    def turn_pid(self, target_angle, speed_limit=180):
        print(f"   PID Turn: Turning to {target_angle} degrees.")
        pid = PIDController(Kp=2.5, Ki=0, Kd=0.05, setpoint=0, output_limits=(-speed_limit, speed_limit))
        while True:
            _, _, _, current_yaw, _, _ = self.pose_handler.get_pose()
            error = target_angle - current_yaw
            if error > 180: error -= 360
            if error < -180: error += 360
            if abs(error) < 1.5: break
            vz_speed = pid.update(-error)
            self.ep_chassis.drive_speed(x=0, y=0, z=vz_speed, timeout=0.1)
            time.sleep(0.01)
        self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        self.ep_chassis.drive_speed(0, 0, 0)
        print("   PID Turn: Completed.")

    def execute_path(self, path):
        if not path or len(path) < 2: return
        print(f"Executing path with PID: {path}")
        self.ep_led.set_led(r=0, g=0, b=255)
        num=1
        for i in range(len(path) - 1):
            print(self.current_position in self.internal_map.explored , num % 2)
            if self.current_position in self.internal_map.explored and num % 2 == 0:

                self.periodic_wall_clearance_adjust(target_clearance_m=0.18)
                print(f"\nPosition {self.current_position} already explored. Skipping scan.")
            num+=1
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
            print("end++++++++++++++++++++++++++++++++++++++++++++++++++")

    # <--- ลำดับการทำงานใน run_mission (เดิม) --->
    def run_mission(self):
        start_time = time.time()
        time_limit_seconds = 600
        print(f"Mission started! Time limit: {time_limit_seconds} seconds.")
        
        self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit_seconds:
                print(f"\n--- TIME'S UP! ({int(elapsed_time)}s elapsed) ---")
                self.ep_led.set_led(r=255, g=193, b=7, effect="flash")
                break

            
            if self.current_position not in self.internal_map.explored:

                previous_pos = self.visited_path[-2] if len(self.visited_path) > 1 else None
                self.scan_surroundings_with_gimbal(previous_position=previous_pos)
            path_to_execute = self.decide_next_path()
            if not path_to_execute:
                print("\n--- MISSION COMPLETE! All areas explored. ---")
                self.ep_led.set_led(r=0, g=255, b=0, effect="on")
                break
            self.execute_path(path_to_execute)
            
            
        print("\n--- Final Marker Map ---")
        if self.marker_map:
            for name, findings in sorted(self.marker_map.items()):
                print(f"   Marker '{name}':")
                for details in findings:
                    print(f"         - Found at Grid={details[0]}, Wall={details[1]}")
        else:
            print("   No markers were logged.")

        plot_map_with_walls(
            self.internal_map.graph,
            self.internal_map.blocked,
            self.visited_path,
            self.marker_map,
            filename="maze_map.png"
        )

        # เดิมมีอยู่แล้ว: บันทึก CSV ของระบบ mapping
        self.save_csv_logs()


def plot_map_with_walls(graph, blocked, path, marker_map, filename="maze_map.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.title("Robot Map with Walls and Path")

    # วาด “ทางเปิด” จาก graph (เส้นบาง)
    drawn = set()
    for a, nbs in graph.items():
        for b in nbs:
            edge = tuple(sorted([a, b]))
            if edge in drawn: 
                continue
            drawn.add(edge)
            x1, y1 = a
            x2, y2 = b
            plt.plot([x1, x2], [y1, y2], linewidth=1.2, alpha=0.5, zorder=1)

    # วาด “กำแพงปิด” จาก blocked (เส้นหนา)
    for edge in blocked:
        (x1, y1), (x2, y2) = edge
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and abs(dy) == 1:
            # เซลล์เดียวกันคอลัมน์ (เหนือ-ใต้) -> กำแพงแนวนอนที่กึ่งกลาง
            y_mid = (y1 + y2) / 2.0
            plt.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], linewidth=3.0, color="k", zorder=3)
        elif dy == 0 and abs(dx) == 1:
            # เซลล์เดียวกันแถว (ซ้าย-ขวา) -> กำแพงแนวตั้งที่กึ่งกลาง
            x_mid = (x1 + x2) / 2.0
            plt.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], linewidth=3.0, color="k", zorder=3)

    # โหนด
    if graph:
        xs, ys = zip(*graph.keys())
        plt.scatter(xs, ys, s=36, color="tab:blue", zorder=2, label="Nodes")

    # เส้นทางที่เดิน
    if path:
        px, py = zip(*path)
        plt.plot(px, py, linewidth=2, color="tab:red", zorder=4, label="Path")
        plt.scatter(px[0], py[0], s=120, color="green", marker='o', zorder=5, label='Start')
        plt.scatter(px[-1], py[-1], s=120, color="purple", marker='X', zorder=5, label='End')

    # วาง Marker “บนกำแพง” ของกริดที่พบ  # <<< CHANGED
    for name, hits in (marker_map or {}).items():
        for (gx, gy), wall in hits:
            # คำนวณตำแหน่งกลางกำแพงของกริด (gx, gy)
            if wall == "North Wall":
                mx, my = gx, gy + 0.4
            elif wall == "East Wall":
                mx, my = gx + 0.4, gy
            elif wall == "South Wall":
                mx, my = gx, gy - 0.4
            elif wall == "West Wall":
                mx, my = gx - 0.4, gy
            else:
                mx, my = gx, gy  # เผื่อกรณีพิเศษ

            plt.scatter([mx], [my], s=60, marker='o', color="red", zorder=6)  # วางบนผนัง
            plt.text(mx + 0.06, my + 0.06, f"{name}", fontsize=8, zorder=7)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(filename, dpi=150)
    print(f"Map saved to '{filename}'")
    plt.close()


# ==============================================================================
# ส่วนหลักของโปรแกรม (Main Execution)
# ==============================================================================
if __name__ == '__main__':
    ep_robot = None
    explorer = None
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        print("Robot connected.")
        tof_handler = TofDataHandler()
        vision_handler = VisionDataHandler()
        ep_vision = ep_robot.vision

        pose_handler = PoseDataHandler()
        ep_robot.sensor.sub_distance(freq=10, callback=tof_handler.update)
        ep_robot.chassis.sub_position(freq=10, callback=pose_handler.update_position)
        ep_robot.chassis.sub_attitude(freq=10, callback=pose_handler.update_attitude)
        ep_robot.vision.sub_detect_info(name="marker", callback=vision_handler.update)

        print("Subscribed to all required sensors.")
        explorer = MazeExplorer(ep_robot, tof_handler, vision_handler, pose_handler)

        # <<< เริ่มบันทึกเซนเซอร์แบบต่อเนื่องทันทีหลัง subscribe >>>
        print("Starting continuous sensor logging...")
        explorer.start_continuous_logging(frequency_hz=10)

        time.sleep(2)  # รอให้ค่าเริ่มต้นนิ่งนิดนึง
        explorer.run_mission()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if explorer:
            # หยุด logging และบันทึกทุกอย่างให้เรียบร้อย (กันพลาดกรณี error)
            print("Stopping continuous sensor logging...")
            explorer.stop_continuous_logging()
            explorer.save_csv_logs()

        if ep_robot:
            ep_robot.sensor.unsub_distance()
            ep_robot.vision.unsub_detect_info(name="marker")
            ep_robot.chassis.unsub_position()
            ep_robot.chassis.unsub_attitude()
            ep_robot.close()
            print("break")
