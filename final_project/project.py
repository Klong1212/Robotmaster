"""
=================================================================================
MAZE EXPLORATION ROBOT WITH IR SENSORS AND REAL-TIME MAPPING
=================================================================================
Description: หุ่นยนต์สำรวจเขาวงกต โดยใช้ ToF sensor ด้านหน้า และ IR sensors ด้านข้าง
            พร้อมแสดงแผนที่แบบ real-time และไปยัง goal หลังสำรวจครบ

Features:
- ToF sensor สำหรับวัดระยะด้านหน้า
- IR digital sensors สำหรับตรวจจับกำแพงด้านซ้าย/ขวา
- Real-time mapping แสดงแผนที่ขณะสำรวจ
- Goal-oriented navigation หลังสำรวจครบ
- CSV logging และ state saving/loading

Hardware Requirements:
- RoboMaster EP/S1
- ToF sensor (built-in)
- IR digital sensors x2 (port 1: ซ้าย, port 2: ขวา)

Author: [Your Name]
Date: October 2025
=================================================================================
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================
import time
import threading
import math
import csv
from datetime import datetime
from collections import deque
import json
import os
import sys
import statistics

import robomaster
from robomaster import robot, vision

# Matplotlib for real-time plotting and map visualization
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Robot Movement and Grid Settings
GRID_SIZE_M = 0.635                    # ขนาดกริดเป็นเมตร
WALL_THRESHOLD_MM = 450                # เกณฑ์ระยะทางที่ถือว่ามีกำแพง (มม.)

# Gimbal and Vision Settings
GIMBAL_TURN_SPEED = 360                # ความเร็วหมุนกิมบอล (องศา/วินาที)
CAMERA_HORIZONTAL_FOV = 96             # มุมมองกล้องแนวนอน (องศา)
GIMBAL_SWEEP_OFFSETS = [0, -30, +30]   # มุมกวาดหา marker (องศา)
VISION_SCAN_DURATION_S = 1             # ระยะเวลาสแกน vision (วินาที)

# Direction Mappings
ORIENTATIONS = {
    0: "North", 
    1: "East", 
    2: "South", 
    3: "West"
}

WALL_NAMES = {
    0: "North Wall", 
    1: "East Wall", 
    2: "South Wall", 
    3: "West Wall"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def wrap_angle_deg(angle_deg: float) -> float:
    """ห่อมุมให้อยู่ในช่วง (-180, 180]"""
    while angle_deg <= -180.0:
        angle_deg += 360.0
    while angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg

def sub_angle(a_deg: float, b_deg: float) -> float:
    """คืนค่า (a - b) แบบ wrap ในช่วง (-180, 180]"""
    return wrap_angle_deg(a_deg - b_deg)


# =============================================================================
# DATA HANDLERS AND SENSOR MANAGEMENT
# =============================================================================

class TofDataHandler:
    """จัดการข้อมูลจาก ToF sensor ด้วย median filtering"""
    
    def __init__(self, window_size=3):
        self._lock = threading.Lock()
        self.raw_distance = 0.0
        self._window = deque(maxlen=window_size)
        self._median = 0.0

    def update(self, sub_info):
        """อัปเดตค่า ToF จาก callback"""
        d = float(sub_info[0]) if sub_info else 0.0
        with self._lock:
            self.raw_distance = d
            self._window.append(d)
            self._median = statistics.median(self._window)

    def get_distance(self):
        """คืนค่าระยะทางที่กรองแล้ว (median)"""
        with self._lock:
            return self._median if self._window else self.raw_distance

    def get_raw_distance(self):
        """คืนค่า ToF ดิบ"""
        with self._lock:
            return self.raw_distance
        with self._lock:
            return self.raw_distance


class VisionDataHandler:
    """จัดการข้อมูลจากกล้อง สำหรับ marker detection"""
    
    def __init__(self):
        self.markers = []                 # [(label, x)]
        self._lock = threading.Lock()
        self._sample_logged = False       # log โครงสร้างครั้งแรกครั้งเดียว

    def update(self, vision_info):
        """อัปเดตข้อมูล marker จาก callback"""
        with self._lock:
            self.markers.clear()
            if vision_info:
                if not self._sample_logged:
                    print("[Vision raw] ->", vision_info)
                    self._sample_logged = True
                for t in vision_info:
                    # รูปแบบ: (x, y, w, h, label, ...)
                    if not isinstance(t, (list, tuple)) or len(t) < 5:
                        continue
                    x = float(t[0])  # 0..1 (ซ้าย..ขวา)
                    label = str(t[4])
                    self.markers.append((label, x))

    def get_markers(self):
        """คืนรายการ markers ปัจจุบัน"""
        with self._lock:
            return list(self.markers)


class GimbalDataHandler:
    """จัดการข้อมูลมุมกิมบอล (yaw, pitch)"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, angle_info):
        """อัปเดตมุมกิมบอลจาก callback"""
        with self._lock:
            self.yaw = float(angle_info[0])
            self.pitch = float(angle_info[1])

    def get_yaw_pitch(self):
        """คืนค่ามุม (yaw, pitch)"""
        with self._lock:
            return (self.yaw, self.pitch)


class PoseDataHandler:
    """จัดการข้อมูลตำแหน่งและทิศทางของหุ่นยนต์"""
    
    def __init__(self):
        self.pose = [0.0] * 6  # [x, y, z, yaw, pitch, roll]
        self._lock = threading.Lock()
        
    def update_position(self, pos_info):
        """อัปเดตตำแหน่ง (x, y, z)"""
        with self._lock:
            self.pose[0], self.pose[1], self.pose[2] = pos_info[0], pos_info[1], pos_info[2]
            
    def update_attitude(self, att_info):
        """อัปเดตทิศทาง (yaw, pitch, roll)"""
        with self._lock:
            self.pose[3], self.pose[4], self.pose[5] = att_info[0], att_info[1], att_info[2]
            
    def get_pose(self):
        """คืนค่า pose ทั้งหมด"""
        with self._lock:
            return tuple(self.pose)
            
    def set_xy(self, x_m, y_m):
        """กำหนดตำแหน่ง x, y"""
        with self._lock:
            self.pose[0], self.pose[1] = float(x_m), float(y_m)
            
    def set_yaw(self, yaw_deg):
        """กำหนดทิศทาง yaw"""
        with self._lock:
            self.pose[3] = float(yaw_deg)


# =============================================================================
# MAP AND NAVIGATION CLASSES
# =============================================================================

class RobotMap:
    """จัดการแผนที่และเส้นทางของหุ่นยนต์"""
    
    def __init__(self):
        self.graph = {}           # กราฟเชื่อมต่อระหว่างช่อง
        self.explored = set()     # ช่องที่สำรวจแล้ว
        self.blocked = set()      # ขอบที่ปิดกั้น (กำแพง)
        
    def add_connection(self, pos1, pos2):
        """เพิ่มการเชื่อมต่อระหว่างสองช่อง"""
        if pos1 not in self.graph: 
            self.graph[pos1] = set()
        if pos2 not in self.graph: 
            self.graph[pos2] = set()
        if pos2 not in self.graph[pos1]:
            self.graph[pos1].add(pos2)
            self.graph[pos2].add(pos1)

    def add_blocked(self, pos1, pos2):
        """เพิ่มขอบที่ปิดกั้น (กำแพง)"""
        edge = tuple(sorted([pos1, pos2]))
        self.blocked.add(edge)
        
    def mark_explored(self, position):
        """ทำเครื่องหมายช่องที่สำรวจแล้ว"""
        self.explored.add(position)
        
    def get_unexplored_neighbors(self, position):
        """หาช่องใกล้เคียงที่ยังไม่ได้สำรวจ"""
        if position not in self.graph: 
            return []
        return [n for n in self.graph.get(position, []) if n not in self.explored]
        
    def get_path(self, start, goal):
        """หาเส้นทางจาก start ไป goal ด้วย BFS"""
        if start == goal: 
            return [start]
        queue = [(start, [start])]
        visited = {start}
        print("queue: ", queue)
        while queue:
            current, path = queue.pop(0)
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    if neighbor == goal: 
                        return path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None


class PIDController:
    """ตัวควบคุม PID สำหรับการเคลื่อนที่"""
    
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-1.0, 1.0)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()
        
    def update(self, current_value):
        """อัปเดตผลลัพธ์ PID"""
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0: 
            return 0.0
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._previous_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        if self.output_limits:
            output = max(self.output_limits[0], min(self.output_limits[1], output))
        self._previous_error = error
        self._last_time = current_time
        return output


# =============================================================================
# MAIN MAZE EXPLORER CLASS
# =============================================================================

class MazeExplorer:
    """
    คลาสหลักสำหรับควบคุมหุ่นยนต์สำรวจเขาวงกต
    
    Features:
    - สำรวจเขาวงกตแบบอัตโนมัติ
    - ใช้ ToF + IR sensors
    - Real-time mapping
    - Goal-oriented navigation
    - CSV logging
    """
    
    def __init__(self, ep_robot, tof_handler, vision_handler, gimbal_handler, pose_handler):
        """
        Initialize MazeExplorer
        
        Args:
            ep_robot: RoboMaster robot instance
            tof_handler: ToF sensor data handler
            vision_handler: Vision/marker detection handler
            gimbal_handler: Gimbal angle handler
            pose_handler: Robot pose handler
        """
        # Robot hardware interfaces
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_led = ep_robot.led
        self.ep_vision = ep_robot.vision
        self.ep_gimbal = ep_robot.gimbal
        self.ep_adaptor = ep_robot.sensor_adaptor
        
        # Data handlers
        self.tof_handler = tof_handler
        self.vision_handler = vision_handler
        self.gimbal_handler = gimbal_handler
        self.pose_handler = pose_handler
        
        # Navigation state
        self.visited_for_check = []
        self.current_position = (0, 0)
        self.current_orientation = 0  # 0:N, 1:E, 2:S, 3:W
        self.internal_map = RobotMap()
        self.visited_path = [self.current_position]
        
        # Exploration settings
        self.border = (4, 4)  # ขนาดพื้นที่สำรวจ (กริด)
        self.step_counter = 0  # นับจำนวนช่องที่เดิน
        
        # Goal settings
        self.goal_position = (3, 3)  # จุดปลายทางสุดท้าย
        self.exploration_complete = False  # flag สำรวจครบหรือยัง
        
        # Marker detection (ถ้าต้องการใช้)
        self.marker_map = {}  # {name: [{grid_pos:(x,y), wall:str, offset_m:float}]}
        
        # Real-time plotting
        self.real_time_plot = True
        self.fig = None
        self.ax = None
        
        # Initialize robot
        self.ep_led.set_led(r=0, g=0, b=255)
        self.pose_handler.set_xy(0.0, 0.0)
        self.pose_handler.set_yaw(0.0)
        
        # Logging
        self.scan_log = []      # สแกนแต่ละกริด
        self.wall_log = set()   # กำแพงที่พบ
        self.marker_log = []    # marker ที่พบ
        self.path_log = []      # เส้นทางที่เดิน
        self.continuous_sensor_log = []  # บันทึกเซ็นเซอร์ต่อเนื่อง
        
        # Logging thread
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
                markers = [m[0] for m in self.vision_handler.get_markers()]
                self.continuous_sensor_log.append({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "pos_x_m": x, "pos_y_m": y, "pos_z_m": z,
                    "att_yaw_deg": yaw, "att_pitch_deg": pitch, "att_roll_deg": roll,
                    "tof_distance_mm": tof,
                    "detected_markers": ",".join(markers) if markers else ""
                })
            except Exception as e:
                print(f"[Logging Thread] Error: {e}")
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
    # บันทึก CSV (รวม continuous_sensor_log และ marker offset)
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

        # 4) markers (มี offset_m)
        if self.marker_log:
            with open(f"marker_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["ts","name","grid_x","grid_y","wall","offset_m"])
                w.writeheader()
                for r in self.marker_log: w.writerow(r)

        # 5) visited path
        if self.visited_path:
            with open(f"path_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["step","grid_x","grid_y"])
                for i, (gx,gy) in enumerate(self.visited_path):
                    w.writerow([i,gx,gy])

    # ========================================================================
    # Sensor Scanning Functions
    # ========================================================================
    
    def _read_ir_sensors(self):
        """อ่านข้อมูลจาก IR sensors และ invert ค่า"""
        ir_left_raw = self.ep_adaptor.get_io(id=1, port=1)   # IR ซ้าย
        ir_right_raw = self.ep_adaptor.get_io(id=1, port=2)  # IR ขวา
        
        # Invert ค่า boolean (0->True, 1->False)
        wall_left = not ir_left_raw
        wall_right = not ir_right_raw
        
        print(f"   IR sensors: Left={wall_left} (raw:{ir_left_raw}), Right={wall_right} (raw:{ir_right_raw})")
        return wall_left, wall_right
    
    def _get_neighbor_position(self, scan_direction):
        """คำนวณตำแหน่งเพื่อนบ้านตามทิศทาง"""
        x, y = self.current_position
        if scan_direction == 0: return (x, y + 1)      # North
        elif scan_direction == 1: return (x + 1, y)    # East  
        elif scan_direction == 2: return (x, y - 1)    # South
        elif scan_direction == 3: return (x - 1, y)    # West
        return None
    
    def _detect_wall_in_direction(self, scan_direction, wall_left, wall_right):
        """ตรวจจับกำแพงในทิศทางที่กำหนด"""
        relative_direction = (scan_direction - self.current_orientation + 4) % 4
        
        if relative_direction == 0:  # ด้านหน้า -> ใช้ ToF
            distance_mm = self.tof_handler.get_distance()
            has_wall = distance_mm < WALL_THRESHOLD_MM
            print(f"         - ToF Front: {distance_mm:.1f} mm, Wall: {has_wall}")
            return has_wall, distance_mm
            
        elif relative_direction == 1:  # ด้านขวา -> ใช้ IR ขวา
            distance_mm = 100.0 if wall_right else 1000.0
            print(f"         - IR Right: Wall={wall_right}")
            return wall_right, distance_mm
            
        elif relative_direction == 3:  # ด้านซ้าย -> ใช้ IR ซ้าย
            distance_mm = 100.0 if wall_left else 1000.0
            print(f"         - IR Left: Wall={wall_left}")
            return wall_left, distance_mm
            
        else:  # ด้านหลัง -> สมมติไม่มีกำแพง
            distance_mm = 1000.0
            print(f"         - Back: Assuming no wall")
            return False, distance_mm
    
    def scan_surroundings_with_gimbal(self, previous_position=None):
        """
        สแกนหากำแพงรอบๆ โดยใช้ ToF (หน้า) และ IR sensors (ซ้าย-ขวา)
        
        Args:
            previous_position: ตำแหน่งก่อนหน้า (ไม่ต้องสแกนทิศนั้น)
            
        Returns:
            dict: ระยะทางที่วัดได้ในแต่ละทิศ
        """
        print(f"\nScanning surroundings at {self.current_position} with ToF and IR sensors...")
        self.ep_led.set_led(r=255, g=255, b=0, effect="breathing")
        self.internal_map.mark_explored(self.current_position)
        
        wall_distances = {}
        
        # กำหนดทิศที่ไม่ต้องสแกน (ทิศที่มาจากตำแหน่งก่อนหน้า)
        direction_to_skip = -1
        if previous_position:
            print(f"   -> Path from {previous_position} is known. Adding connection automatically.")
            self.internal_map.add_connection(self.current_position, previous_position)
            direction_to_skip = (self.current_orientation + 2) % 4
            print(f"   -> Will skip physical scan for direction {ORIENTATIONS.get(direction_to_skip, 'N/A')}.")

        # อ่านข้อมูล IR sensors ครั้งเดียว
        wall_left, wall_right = self._read_ir_sensors()

        # สแกนทั้ง 4 ทิศ
        for scan_direction in range(4):
            if scan_direction == direction_to_skip:
                continue
                
            print(f"   Scanning direction: {ORIENTATIONS[scan_direction]}...")
            
            neighbor_pos = self._get_neighbor_position(scan_direction)
            has_wall, distance_mm = self._detect_wall_in_direction(scan_direction, wall_left, wall_right)
            
            wall_distances[f'{scan_direction}'] = distance_mm

            # ประมวลผลผลการตรวจจับ
            if not has_wall:
                # ไม่มีกำแพง -> เป็นทางเปิด
                if (neighbor_pos[0] >= 0 and neighbor_pos[1] >= 0 and 
                    neighbor_pos[0] < self.border[0] and neighbor_pos[1] < self.border[1]):
                    self.internal_map.add_connection(self.current_position, neighbor_pos)
                    print(f"           - Open path recorded towards {neighbor_pos}.")
            else:
                # มีกำแพง -> บล็อกทาง
                print(f"           - Wall detected. Blocking path to {neighbor_pos}.")
                self.internal_map.add_blocked(self.current_position, neighbor_pos)
                self.wall_log.add(tuple(sorted([self.current_position, neighbor_pos])))

        # รีเซ็นเตอร์กิมบอล
        self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        print("Scan complete using ToF and IR sensors.")

        # บันทึกผลการสแกน
        self._save_scan_results(wall_distances)
        return wall_distances
    
    def _save_scan_results(self, wall_distances):
        """บันทึกผลการสแกนลงใน scan_log"""
        x, y = self.current_position
        self.scan_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "grid_x": x, "grid_y": y,
            "N_mm": wall_distances.get('0', None),
            "E_mm": wall_distances.get('1', None),
            "S_mm": wall_distances.get('2', None),
            "W_mm": wall_distances.get('3', None)
        })

    # ========================================================================
    # Goal and Navigation Functions
    # ========================================================================
    
    def set_goal(self, goal_x, goal_y):
        """กำหนดจุดปลายทาง (goal)"""
        self.goal_position = (goal_x, goal_y)
        print(f"Goal set to: {self.goal_position}")

    def decide_next_path(self):
        """
        ตัดสินใจหาเส้นทางถัดไป
        - หากยังสำรวจไม่ครบ: หาจุดใหม่ที่ยังไม่ได้สำรวจ
        - หากสำรวจครบแล้ว: ไปยัง goal
        
        Returns:
            list: เส้นทางที่ต้องเดิน หรือ None หากเสร็จแล้ว
        """
        # ตรวจสอบว่าสำรวจครบทุกจุดแล้วหรือยัง
        if not self.exploration_complete:
            # ยังสำรวจไม่ครบ - หาจุดใหม่ที่ยังไม่ได้สำรวจ
            unexplored = self.internal_map.get_unexplored_neighbors(self.current_position)
            if unexplored:
                return [self.current_position, unexplored[0]]
            
            # ไม่มีจุดใหม่รอบๆ - ย้อนกลับหาจุดที่มีทางใหม่
            for pos in reversed(self.visited_path):
                if self.internal_map.get_unexplored_neighbors(pos):
                    print(f"Backtracking to find unexplored path from {pos}...")
                    return self.internal_map.get_path(self.current_position, pos)
            
            # ไม่มีจุดใหม่แล้ว - สำรวจครบแล้ว
            print(f"\n=== EXPLORATION COMPLETE! ===")
            print(f"All areas explored. Now heading to goal: {self.goal_position}")
            self.exploration_complete = True
            
            # หาเส้นทางไป goal
            if self.current_position != self.goal_position:
                path_to_goal = self.internal_map.get_path(self.current_position, self.goal_position)
                if path_to_goal:
                    return path_to_goal
                else:
                    print(f"Warning: Cannot find path to goal {self.goal_position}!")
                    return None
            else:
                print(f"Already at goal position!")
                return None
        else:
            # สำรวจครบแล้ว และอยู่ที่ goal แล้ว
            return None

    # ========================================================================
    # Real-time Plotting Functions
    # ========================================================================
    
    def init_real_time_plot(self):
        """เริ่มต้น real-time plotting"""
        if not self.real_time_plot:
            return
            
        try:
            plt.ion()  # เปิด interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_title("Robot Map - Real Time")
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim(-1, self.border[0])
            self.ax.set_ylim(-1, self.border[1])
            plt.show(block=False)
            print("Real-time plotting initialized")
        except Exception as e:
            print(f"Failed to initialize real-time plot: {e}")
            self.real_time_plot = False

    def update_real_time_plot(self):
        """อัปเดต real-time plot"""
        if not self.real_time_plot or self.fig is None:
            return
            
        try:
            # ล้างแผนที่เก่า
            self.ax.clear()
            self.ax.set_title(f"Robot Map - Real Time | Position: {self.current_position}")
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim(-1, self.border[0])
            self.ax.set_ylim(-1, self.border[1])

            # วาด "ทางเปิด" จาก graph (เส้นบาง)
            drawn = set()
            for a, nbs in self.internal_map.graph.items():
                for b in nbs:
                    edge = tuple(sorted([a, b]))
                    if edge in drawn:
                        continue
                    drawn.add(edge)
                    x1, y1 = a
                    x2, y2 = b
                    self.ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1.2, alpha=0.5, zorder=1)

            # วาด "กำแพงปิด" จาก blocked (เส้นหนา)
            for edge in self.internal_map.blocked:
                (x1, y1), (x2, y2) = edge
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and abs(dy) == 1:
                    # แนวเดียวกันคอลัมน์ (เหนือ-ใต้) -> กำแพงแนวนอนที่กึ่งกลาง
                    y_mid = (y1 + y2) / 2.0
                    self.ax.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], 'k-', linewidth=3.0, zorder=3)
                elif dy == 0 and abs(dx) == 1:
                    # แนวเดียวกันแถว (ซ้าย-ขวา) -> กำแพงแนวตั้งที่กึ่งกลาง
                    x_mid = (x1 + x2) / 2.0
                    self.ax.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], 'k-', linewidth=3.0, zorder=3)

            # วาดโหนดที่สำรวจแล้ว
            if self.internal_map.graph:
                explored_nodes = list(self.internal_map.explored)
                if explored_nodes:
                    ex, ey = zip(*explored_nodes)
                    self.ax.scatter(ex, ey, s=50, c='lightblue', marker='s', zorder=2, label='Explored')

            # วาดเส้นทางที่เดิน
            if len(self.visited_path) > 1:
                px, py = zip(*self.visited_path)
                self.ax.plot(px, py, 'r-', linewidth=2, zorder=4, label='Path')

            # วาดตำแหน่งปัจจุบัน
            cx, cy = self.current_position
            self.ax.scatter([cx], [cy], s=150, c='red', marker='o', zorder=5, label='Current')

            # วaด Goal
            gx, gy = self.goal_position
            self.ax.scatter([gx], [gy], s=120, c='gold', marker='*', zorder=5, label='Goal')

            # วาดจุดเริ่มต้น
            if self.visited_path:
                sx, sy = self.visited_path[0]
                self.ax.scatter([sx], [sy], s=120, c='green', marker='o', zorder=5, label='Start')

            self.ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
            
            # อัปเดตแผนที่
            plt.draw()
            plt.pause(0.1)  # หน่วงเวลาเล็กน้อยให้เห็นการเปลี่ยนแปลง
            
        except Exception as e:
            print(f"Error updating real-time plot: {e}")

    def close_real_time_plot(self):
        """ปิด real-time plotting"""
        if self.real_time_plot and self.fig is not None:
            try:
                plt.ioff()  # ปิด interactive mode
                plt.close(self.fig)
                print("Real-time plotting closed")
            except Exception as e:
                print(f"Error closing real-time plot: {e}")

    # ========================================================================
    # Sensor Adjustment Functions
    # ========================================================================
    
    def periodic_wall_clearance_adjust(self, target_clearance_m=0.175):
        """
        ปรับตำแหน่งหุ่นโดยใช้ข้อมูลจากเซ็นเซอร์ ToF และ IR
        เพื่อให้อยู่ห่างจากกำแพงในระยะที่เหมาะสม
        """
        print("   Checking wall clearances using ToF and IR sensors...")
        
        # อ่านค่าจากเซ็นเซอร์
        try:
            # ToF หน้า
            front_distance_mm = self.tof_handler.get_distance()
            
            # IR ซ้าย/ขวา
            ir_left = self.ep_adaptor.get_adc(id=1, port=1)
            ir_right = self.ep_adaptor.get_adc(id=1, port=2)
            
            wall_left = ir_left > 0.5
            wall_right = ir_right > 0.5
            
            print(f"   Sensors: Front={front_distance_mm:.1f}mm, Left={wall_left}, Right={wall_right}")
            
            # ปรับตำแหน่งถ้าใกล้กำแพงหน้าเกินไป
            if front_distance_mm < WALL_THRESHOLD_MM:
                move_dist_m = (front_distance_mm / 1000.0) - target_clearance_m
                if abs(move_dist_m) > 0.025:
                    print(f"   Adjusting distance from front wall: {move_dist_m:.2f}m")
                    self.ep_chassis.move(x=move_dist_m, y=0, z=0, xy_speed=0.3).wait_for_completed()
                    time.sleep(0.2)
            
            self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
            
        except Exception as e:
            print(f"Error in wall clearance adjust: {e}")
            self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)

    # ========================================================================
    # Movement and PID Control Functions
    # ========================================================================
    
    def move_forward_pid(self, distance_m, speed_limit=0.2):
        """
        ใช้ PID controller เพื่อเคลื่อนที่ไปข้างหน้าระยะทางที่กำหนด
        
        Args:
            distance_m: ระยะทางที่ต้องการเดิน (เมตร)
            speed_limit: ความเร็วสูงสุด (เมตร/วินาที)
        """
        print(f"   PID Move: Moving forward {distance_m}m.")
        pid = PIDController(Kp=2, Ki=0.02, Kd=0.1, setpoint=distance_m, output_limits=(-speed_limit, speed_limit))
        start_x, start_y, _, _, _, _ = self.pose_handler.get_pose()
        self
        while True:
            curr_x, curr_y, _, _, _, _ = self.pose_handler.get_pose()
            dist_traveled = math.hypot(curr_x - start_x, curr_y - start_y)
            if abs(distance_m - dist_traveled) < 0.01: break
            if  self.tof_handler.get_distance()<=280:
                self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.1)
                break
            vx_speed = pid.update(dist_traveled)
            self.ep_chassis.drive_speed(x=vx_speed, y=0, z=0, timeout=0.1)
            time.sleep(0.01)

        self.ep_chassis.drive_speed(0, 0, 0)
        print("   PID Move: Completed.")

        gx, gy = self.current_position
        _, _, _, yaw, _, _ = self.pose_handler.get_pose()
        self.path_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "grid_x": gx, "grid_y": gy, "yaw": yaw
        })

    def turn_pid(self, target_angle, speed_limit=90):
        """
        ใช้ PID controller เพื่อหมุนไปยังมุมที่กำหนด
        
        Args:
            target_angle: มุมเป้าหมาย (องศา)
            speed_limit: ความเร็วการหมุนสูงสุด (องศา/วินาที)
        """
        print(f"   PID Turn: Turning to {target_angle} degrees.")
        pid = PIDController(Kp=2.5, Ki=0, Kd=0.15, setpoint=0, output_limits=(-speed_limit, speed_limit))
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
        """
        ดำเนินการเดินตามเส้นทางที่กำหนดด้วย PID control
        
        Args:
            path: รายการของตำแหน่งที่ต้องเดิน [(x1,y1), (x2,y2), ...]
        """
        if not path or len(path) < 2: return
        print(f"Executing path with PID: {path}")
        self.ep_led.set_led(r=0, g=0, b=255)
        for i in range(len(path) - 1):
            if self.current_position in self.visited_for_check:
                self.periodic_wall_clearance_adjust(target_clearance_m=0.18)
                print(f"\nPosition {self.current_position} already explored. Skipping scan.")
            self.visited_for_check.append(self.current_position)
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

    # ========================================================================
    # Main Mission Control
    # ========================================================================
    
    def run_mission(self):
        """
        ดำเนินการภารกิจหลัก:
        1. สำรวจพื้นที่ทั้งหมดก่อน
        2. ไปยังจุดปลายทาง (goal) เมื่อสำรวจครบแล้ว
        3. แสดงแผนที่แบบ real-time
        4. บันทึกข้อมูลเซ็นเซอร์และเส้นทาง
        """
        start_time = time.time()
        time_limit_seconds = 600
        print(f"Mission started! Time limit: {time_limit_seconds} seconds.")
        self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()

        # เริ่มต้น real-time plotting
        self.init_real_time_plot()
        self.update_real_time_plot()  # แสดงแผนที่เริ่มต้น

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit_seconds:
                print(f"\n--- TIME'S UP! ({int(elapsed_time)}s elapsed) ---")
                self.ep_led.set_led(r=255, g=193, b=7, effect="flash")
                break


            if self.current_position not in self.internal_map.explored:
                previous_pos = self.visited_path[-2] if len(self.visited_path) > 1 else None
                self.scan_surroundings_with_gimbal(previous_position=previous_pos)
                # อัปเดตแผนที่หลังสแกน
                self.update_real_time_plot()

            path_to_execute = self.decide_next_path()
            if not path_to_execute:
                if self.exploration_complete and self.current_position == self.goal_position:
                    print(f"\n--- MISSION COMPLETE! Reached goal at {self.goal_position} ---")
                    self.ep_led.set_led(r=0, g=255, b=0, effect="on")
                else:
                    print(f"\n--- EXPLORATION COMPLETE! All areas explored ---")
                    self.ep_led.set_led(r=255, g=165, b=0, effect="on")  # orange
                break
            self.execute_path(path_to_execute)
            # อัปเดตแผนที่หลังเคลื่อนที่
            self.update_real_time_plot()

        print("\n--- Final Marker Map ---")
        if self.marker_map:
            for name, findings in sorted(self.marker_map.items()):
                print(f"   Marker '{name}':")
                for details in findings:
                    print(f"         - Found at Grid={details['grid_pos']}, Wall={details['wall']}, Offset={details['offset_m']} m")
        else:
            print("   No markers were logged.")

        # ปิด real-time plot และบันทึกแผนที่สุดท้าย
        self.close_real_time_plot()
        
        plot_map_with_walls(
            self.internal_map.graph,
            self.internal_map.blocked,
            self.visited_path,
            self.marker_map,
            filename="maze_map.png"
        )

        self.save_csv_logs()

# ==============================================================================
# Map Visualization Functions
# ==============================================================================

def plot_map_with_walls(graph, blocked, path, marker_map, filename="maze_map.png"):
    """
    วาดแผนที่สุดท้ายพร้อมกำแพง เส้นทาง และ markers
    
    Args:
        graph: กราฟของเส้นทางที่เดินได้
        blocked: เส้นทางที่ถูกกำแพงปิด
        path: เส้นทางที่หุ่นเดิน
        marker_map: markers ที่พบ
        filename: ชื่อไฟล์ที่ต้องการบันทึก
    """
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
            # แนวเดียวกันคอลัมน์ (เหนือ-ใต้) -> กำแพงแนวนอนที่กึ่งกลาง
            y_mid = (y1 + y2) / 2.0
            plt.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], linewidth=3.0, color="k", zorder=3)
        elif dy == 0 and abs(dx) == 1:
            # แนวเดียวกันแถว (ซ้าย-ขวา) -> กำแพงแนวตั้งที่กึ่งกลาง
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
        
    # วาง Marker “ตามตำแหน่งจริงบนกำแพง” ด้วย offset_m (ซ้าย=ลบ, ขวา=บวก)
    for name, hits in (marker_map or {}).items():
        for finding in hits:
            (gx, gy) = finding['grid_pos']
            wall = finding['wall']
            offset = finding['offset_m'] / GRID_SIZE_M  # เป็นหน่วย "จำนวนกริด"

            if wall == "North Wall":
                # ซ้าย(ลบ) -> x ลดลง: mx = gx + offset
                mx, my = gx + offset, gy + 0.4
            elif wall == "East Wall":
                # ซ้าย(ลบ) ของผนังหันขวา = เหนือ -> y เพิ่ม: my = gy - offset
                mx, my = gx + 0.4, gy - offset
            elif wall == "South Wall":
                # ซ้าย(ลบ) ของผนังหันลง = ตะวันออก -> x เพิ่ม: mx = gx - offset
                mx, my = gx - offset, gy - 0.4
            elif wall == "West Wall":
                # ซ้าย(ลบ) ของผนังหันซ้าย = ใต้ -> y ลด: my = gy + offset
                mx, my = gx - 0.4, gy + offset
            else:
                mx, my = gx, gy

            plt.scatter([mx], [my], s=60, marker='*', color="magenta", zorder=6)
            plt.text(mx + 0.06, my + 0.06, f"{name}", fontsize=8, zorder=7)


    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(filename, dpi=150)
    print(f"Map saved to '{filename}'")
    plt.close()


# ==============================================================================
# Main Program Execution
# ==============================================================================

def main():
    """
    ฟังก์ชันหลักของโปรแกรม
    - เชื่อมต่อกับหุ่นยนต์ RoboMaster
    - กำหนดการทำงานของเซ็นเซอร์
    - เริ่มต้นระบบสำรวจ maze
    - จัดการการหยุดทำงานอย่างปลอดภัย
    """
    ep_robot = None
    explorer = None
    
    try:
        # เชื่อมต่อหุ่นยนต์
        print("Connecting to RoboMaster robot...")
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        print("Robot connected successfully.")

        # สร้าง data handlers สำหรับเซ็นเซอร์ต่างๆ
        tof_handler = TofDataHandler()
        vision_handler = VisionDataHandler()
        gimbal_handler = GimbalDataHandler()
        pose_handler = PoseDataHandler()

        # สมัครรับข้อมูลจากเซ็นเซอร์
        print("Setting up sensor subscriptions...")
        ep_robot.sensor.sub_distance(freq=10, callback=tof_handler.update)
        ep_robot.chassis.sub_position(freq=10, callback=pose_handler.update_position)
        ep_robot.chassis.sub_attitude(freq=10, callback=pose_handler.update_attitude)
        ep_robot.vision.sub_detect_info(name="marker", callback=vision_handler.update)
        ep_robot.gimbal.sub_angle(freq=20, callback=gimbal_handler.update)
        print("All sensor subscriptions completed.")

        # สร้าง maze explorer
        explorer = MazeExplorer(ep_robot, tof_handler, vision_handler, gimbal_handler, pose_handler)
        
        # กำหนดการตั้งค่าภารกิจ
        explorer.set_goal(3, 0)  # กำหนด goal position
        print(f"Mission objective: Explore all accessible areas first, then navigate to goal {explorer.goal_position}")
        print("Real-time map visualization: ENABLED")
        
        # เริ่มบันทึกข้อมูลเซ็นเซอร์
        print("Starting continuous sensor data logging...")
        explorer.start_continuous_logging(frequency_hz=10)

        # รอให้ระบบเซ็นเซอร์เสถียร
        time.sleep(2)
        
        # เริ่มปฏิบัติภารกิจ
        print("\n" + "="*50)
        print("🤖 STARTING MAZE EXPLORATION MISSION 🤖")
        print("="*50)
        explorer.run_mission()

    except Exception as e:
        print(f"❌ An error occurred during mission: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # การหยุดทำงานอย่างปลอดภัย
        print("\n" + "="*50)
        print("🔧 CLEANUP AND SHUTDOWN SEQUENCE 🔧")
        print("="*50)
        
        if explorer:
            print("Stopping sensor logging and saving data...")
            explorer.stop_continuous_logging()
            explorer.save_csv_logs()
            explorer.close_real_time_plot()

        if ep_robot:
            print("Unsubscribing from sensors...")
            try:
                ep_robot.sensor.unsub_distance()
                ep_robot.vision.unsub_detect_info(name="marker")
                ep_robot.gimbal.unsub_angle()
                ep_robot.chassis.unsub_position()
                ep_robot.chassis.unsub_attitude()
                ep_robot.close()
                print("Robot connection closed safely.")
            except Exception as e:
                print(f"Warning during robot cleanup: {e}")
        
        print("✅ Mission completed and cleanup finished.")

if __name__ == '__main__':
    main()
