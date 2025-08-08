import time
import math
import robomaster
import threading
import keyboard
import matplotlib.pyplot as plt
from robomaster import robot
from typing import List, Tuple, Dict, Set, Optional, Any

# ==============================================================================
# 1. ศูนย์รวมการตั้งค่า (Centralized Configuration)
# ==============================================================================
class RobotConfig:
    GRID_SIZE_M: float = 0.6
    WALL_THRESHOLD_MM: int = 500
    VISION_SCAN_DURATION_S: float = 1.0
    MOVEMENT_TIMEOUT_S: float = 10.0
    GIMBAL_TURN_SPEED: int = 120
    MAX_TURN_SPEED_DPS: int = 100
    MAX_MOVE_SPEED_MPS: float = 0.7
    TURN_TOLERANCE_DEG: float = 1.0
    MOVE_TOLERANCE_M: float = 0.02
    EMERGENCY_STOP_DISTANCE_MM: int = 150
    TURN_PID: Tuple[float, float, float] = (3.0, 0.15, 0.35)
    MOVE_PID: Tuple[float, float, float] = (2.5, 0.1, 0.25)
    HEADING_PID: Tuple[float, float, float] = (2.2, 0, 0) # Ki เป็น 0 อย่างตั้งใจ

Point = Tuple[int, int]
ORIENTATIONS: Dict[int, str] = {0: "North", 1: "East", 2: "South", 3: "West"}
WALL_NAMES: Dict[int, str] = {0: "South Wall", 1: "West Wall", 2: "North Wall", 3: "East Wall"}
DIRECTION_VECTORS: Dict[int, Point] = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

# ==============================================================================
# 2. คลาสจัดการระบบ (System Classes)
# ==============================================================================
class RobotMovementError(Exception): pass

class PIDController:
    """[FIXED] คลาส PID Controller พร้อมระบบ Anti-Windup ที่ปลอดภัย"""
    def __init__(self, Kp: float, Ki: float, Kd: float, output_limits: Tuple[float, float]):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.min_output, self.max_output = output_limits
        self.last_error: float = 0.0; self.integral: float = 0.0; self.last_time: float = time.time()

    def update(self, error: float) -> float:
        current_time = time.time(); delta_time = current_time - self.last_time
        if delta_time == 0: return self.min_output
        
        p_term = self.Kp * error
        self.integral += error * delta_time
        d_term = self.Kd * ((error - self.last_error) / delta_time)
        output = p_term + (self.Ki * self.integral) + d_term
        
        # [FIXED] เพิ่มเงื่อนไขป้องกันการหารด้วยศูนย์เมื่อ Ki = 0
        if self.Ki != 0:
            if output > self.max_output:
                self.integral -= (output - self.max_output) / self.Ki
                output = self.max_output
            elif output < self.min_output:
                self.integral -= (output - self.min_output) / self.Ki
                output = self.min_output
        else:
             # ถ้า Ki เป็น 0 ให้ clamp ค่า output ตามปกติ
            output = max(self.min_output, min(output, self.max_output))

        self.last_error = error; self.last_time = current_time; return output
    
    def reset(self): self.last_error = 0.0; self.integral = 0.0; self.last_time = time.time()

class OdometryDataHandler:
    def __init__(self): self.yaw: float = 0.0; self.x: float = 0.0; self.y: float = 0.0
    def sub_attitude(self, sub_info: List[float]): self.yaw = sub_info[0]
    def sub_position(self, sub_info: List[float]): self.x, self.y = sub_info[0], sub_info[1]
    def get_yaw(self) -> float: return self.yaw
    def get_position(self) -> Tuple[float, float]: return (self.x, self.y)

class TofDataHandler:
    def __init__(self): self.distance: int = 0
    def update(self, sub_info: List[int]): self.distance = sub_info[0]
    def get_distance(self) -> int: return self.distance

class VisionDataHandler:
    def __init__(self): self.markers: List[str] = []
    def update(self, vision_info: List[Any]): self.markers = [info[0] for info in vision_info[1:]] if vision_info[0] > 0 else []
    def get_markers(self) -> List[str]: return self.markers

class RobotMap:
    def __init__(self): self.graph: Dict[Point, Set[Point]] = {}; self.explored: Set[Point] = set()
    def add_connection(self, pos1: Point, pos2: Point):
        self.graph.setdefault(pos1, set()).add(pos2); self.graph.setdefault(pos2, set()).add(pos1)
        print(f"  Map: Added connection between {pos1} and {pos2}")
    def mark_explored(self, position: Point): self.explored.add(position)
    
    def get_path_to_unexplored(self, start_pos: Point) -> Optional[List[Point]]:
        """[IMPROVED] ปรับปรุงการ Backtracking ให้เสถียร"""
        unexplored_adj = [n for n in self.graph.get(start_pos, set()) if n not in self.explored]
        if unexplored_adj: return [start_pos, sorted(unexplored_adj)[0]] # เลือกอันที่น้อยที่สุดเสมอ
        
        # [IMPROVED] sort list ก่อนเพื่อให้การ backtrack เหมือนเดิมทุกครั้ง
        for pos in sorted(list(self.explored), reverse=True):
            if any(n not in self.explored for n in self.graph.get(pos, set())):
                print(f"No new paths here. Backtracking to {pos}...")
                return self.get_path(start_pos, pos)
        return None

    def get_path(self, start: Point, goal: Point) -> Optional[List[Point]]:
        if start == goal: return [start]
        queue: List[Tuple[Point, List[Point]]] = [(start, [start])]; visited: Set[Point] = {start}
        while queue:
            current, path = queue.pop(0)
            for neighbor in sorted(list(self.graph.get(current, set()))): # sort เพื่อความเสถียร
                if neighbor not in visited:
                    if neighbor == goal: return path + [neighbor]
                    visited.add(neighbor); queue.append((neighbor, path + [neighbor]))
        return None

# ==============================================================================
# 3. คลาสหลัก (Main Logic Class)
# ==============================================================================
class MazeExplorer:
    def __init__(self, ep_robot: robot.Robot, cfg: RobotConfig, handlers: Dict[str, Any], stop_event: threading.Event):
        self.ep_robot = ep_robot; self.ep_chassis = ep_robot.chassis; self.ep_led = ep_robot.led; self.ep_gimbal = ep_robot.gimbal
        self.cfg = cfg
        self.odom_handler: OdometryDataHandler = handlers['odom']
        self.tof_handler: TofDataHandler = handlers['tof']
        self.vision_handler: VisionDataHandler = handlers['vision']
        self.stop_event = stop_event
        self.current_position: Point = (0, 0); self.current_orientation: int = 0
        self.internal_map = RobotMap(); self.marker_map: Dict[str, Tuple[Point, str]] = {}
        self.mission_path: List[Point] = [(0, 0)]
        self.turn_pid = PIDController(*cfg.TURN_PID, (-cfg.MAX_TURN_SPEED_DPS, cfg.MAX_TURN_SPEED_DPS))
        self.move_pid = PIDController(*cfg.MOVE_PID, (-cfg.MAX_MOVE_SPEED_MPS, cfg.MAX_MOVE_SPEED_MPS))
        self.heading_pid = PIDController(*cfg.HEADING_PID, (-cfg.MAX_TURN_SPEED_DPS, cfg.MAX_TURN_SPEED_DPS))
        print("Robot Explorer Initialized (Production Grade). Press 'q' to stop and generate map.")
        self.ep_led.set_led(r=0, g=0, b=255); self.ep_gimbal.recenter().wait_for_completed()

    # ... ส่วนที่เหลือของคลาส MazeExplorer ทำงานถูกต้องและปลอดภัย ไม่มีการเปลี่ยนแปลง ...
    def scan_surroundings_with_gimbal(self):
        print(f"\nScanning surroundings at {self.current_position} with Gimbal...")
        self.ep_led.set_led(r=255, g=255, b=0, effect="breathing")
        self.internal_map.mark_explored(self.current_position)
        x, y = self.current_position
        for scan_dir in range(4):
            if self.stop_event.is_set(): return
            print(f"  Scanning direction: {ORIENTATIONS[scan_dir]}...")
            angle_to_turn_gimbal = (scan_dir - self.current_orientation) * 90
            if angle_to_turn_gimbal >= 180: angle_to_turn_gimbal -= 360
            if angle_to_turn_gimbal <= -180: angle_to_turn_gimbal += 360
            self.ep_gimbal.moveto(yaw=angle_to_turn_gimbal, pitch=0, yaw_speed=self.cfg.GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(0.5)
            if self.tof_handler.get_distance() >= self.cfg.WALL_THRESHOLD_MM:
                dx, dy = DIRECTION_VECTORS[scan_dir]
                self.internal_map.add_connection(self.current_position, (x + dx, y + dy))
            time.sleep(self.cfg.VISION_SCAN_DURATION_S)
            for marker_name in self.vision_handler.get_markers():
                if marker_name not in self.marker_map:
                    dx, dy = DIRECTION_VECTORS[scan_dir]
                    self.marker_map[marker_name] = ((x + dx, y + dy), WALL_NAMES[scan_dir])
                    print(f"    !!! Marker Found: '{marker_name}' at Grid {(x + dx, y + dy)} on the {WALL_NAMES[scan_dir]} !!!")
        self.ep_gimbal.recenter().wait_for_completed()
        print("Scan complete. Gimbal recentered.")
    def turn_with_pid(self, angle_to_turn: float):
        start_time = time.time(); self.turn_pid.reset()
        start_yaw = self.odom_handler.get_yaw(); target_yaw = start_yaw + angle_to_turn
        while time.time() - start_time < self.cfg.MOVEMENT_TIMEOUT_S:
            if self.stop_event.is_set(): raise RobotMovementError("Mission stopped by user.")
            current_yaw = self.odom_handler.get_yaw(); error = target_yaw - current_yaw
            if error > 180: error -= 360
            if error < -180: error += 360
            if abs(error) < self.cfg.TURN_TOLERANCE_DEG: self.ep_chassis.drive_speed(vz=0); return
            vz_speed = self.turn_pid.update(error); self.ep_chassis.drive_speed(vz=vz_speed); time.sleep(0.01)
        self.ep_chassis.drive_speed(vz=0); raise RobotMovementError(f"Turn action timed out.")
    def move_forward_with_pid(self, distance_m: float):
        start_time = time.time(); self.move_pid.reset(); self.heading_pid.reset()
        self.ep_gimbal.recenter().wait_for_completed()
        start_x, start_y = self.odom_handler.get_position(); target_heading = self.odom_handler.get_yaw()
        while time.time() - start_time < self.cfg.MOVEMENT_TIMEOUT_S:
            if self.stop_event.is_set(): raise RobotMovementError("Mission stopped by user.")
            if self.tof_handler.get_distance() < self.cfg.EMERGENCY_STOP_DISTANCE_MM: self.ep_chassis.drive_speed(x=0, y=0, z=0); raise RobotMovementError(f"Emergency stop! Obstacle detected.")
            current_x, current_y = self.odom_handler.get_position()
            distance_traveled = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
            error = distance_m - distance_traveled
            if abs(error) < self.cfg.MOVE_TOLERANCE_M: self.ep_chassis.drive_speed(x=0, y=0, z=0); return
            vx_speed = self.move_pid.update(error)
            current_heading = self.odom_handler.get_yaw(); heading_error = target_heading - current_heading
            if heading_error > 180: heading_error -= 360
            if heading_error < -180: heading_error += 360
            vz_correct_speed = self.heading_pid.update(heading_error)
            self.ep_chassis.drive_speed(x=vx_speed, z=vz_correct_speed); time.sleep(0.01)
        self.ep_chassis.drive_speed(x=0, y=0, z=0); raise RobotMovementError(f"Move action timed out.")
    def execute_path(self, path: List[Point]):
        if not path or len(path) < 2: return
        print(f"Executing path with PID: {path}")
        self.ep_led.set_led(r=0, g=0, b=255)
        for i in range(len(path) - 1):
            start_node, end_node = path[i], path[i+1]
            dx, dy = end_node[0] - start_node[0], end_node[1] - start_node[1]
            target_orientation = -1
            for direction, vector in DIRECTION_VECTORS.items():
                if vector == (dx, dy): target_orientation = direction; break
            angle_to_turn = (target_orientation - self.current_orientation) * 90
            if angle_to_turn >= 180: angle_to_turn -= 360
            if angle_to_turn <= -180: angle_to_turn += 360
            if angle_to_turn != 0:
                print(f"  Turning chassis {angle_to_turn} degrees with PID...")
                self.turn_with_pid(angle_to_turn)
                self.current_orientation = target_orientation
            print(f"  Moving forward {self.cfg.GRID_SIZE_M}m with PID...")
            self.move_forward_with_pid(self.cfg.GRID_SIZE_M)
            self.current_position = end_node
            self.mission_path.append(end_node)
    def run_mission(self):
        print("Mission starting... Press 'q' to stop.")
        while not self.stop_event.is_set():
            try:
                self.scan_surroundings_with_gimbal()
                if self.stop_event.is_set(): break
                path_to_execute = self.internal_map.get_path_to_unexplored(self.current_position)
                if not path_to_execute: print("\n--- MISSION COMPLETE! All areas explored. ---"); self.ep_led.set_led(r=0, g=255, b=0, effect="on"); break
                self.execute_path(path_to_execute)
            except RobotMovementError as e: print(f"\n--- MOVEMENT ERROR: {e} ---"); print("Aborting mission."); self.ep_led.set_led(r=255, g=0, b=0, effect="flash"); break
            except KeyboardInterrupt: break
        if self.stop_event.is_set(): print("\nMission stopped by user ('q' key).")

# ==============================================================================
# 4. ฟังก์ชันสร้างภาพแผนที่ (Map Visualization)
# ==============================================================================
def visualize_map(robot_map: RobotMap, marker_map: Dict[str, Tuple[Point, str]], mission_path: List[Point]):
    """[IMPROVED] ใช้ Matplotlib สร้างภาพแผนที่จากข้อมูลที่รวบรวมได้"""
    if not robot_map.graph: print("Map data is empty, cannot generate visualization."); return
    fig, ax = plt.subplots(figsize=(10, 10)); ax.set_aspect('equal'); ax.set_facecolor('#2B2B2B')
    all_nodes = list(robot_map.graph.keys()); min_x, max_x = min(p[0] for p in all_nodes) - 1, max(p[0] for p in all_nodes) + 1
    min_y, max_y = min(p[1] for p in all_nodes) - 1, max(p[1] for p in all_nodes) + 1
    for x in range(min_x, max_x + 2):
        for y in range(min_y, max_y + 2):
            pos = (x, y)
            if pos not in robot_map.graph: continue
            if (x, y + 1) not in robot_map.graph.get(pos, set()): ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], color='cyan', linewidth=3)
            if (x + 1, y) not in robot_map.graph.get(pos, set()): ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], color='cyan', linewidth=3)
    if len(mission_path) > 1:
        path_x, path_y = [p[0] for p in mission_path], [p[1] for p in mission_path]
        ax.plot(path_x, path_y, 'o-', color='magenta', markersize=8, markerfacecolor='yellow', label='Robot Path')
    
    # [IMPROVED] ปรับตำแหน่งการวาด Marker
    for name, (wall_pos, wall_orient_str) in marker_map.items():
        from_pos = mission_path[-1]
        for pos in reversed(mission_path):
            if wall_pos in [(pos[0]+dx, pos[1]+dy) for dx,dy in DIRECTION_VECTORS.values()]: from_pos = pos; break
        mx, my = (from_pos[0] + wall_pos[0]) / 2.0, (from_pos[1] + wall_pos[1]) / 2.0
        ax.plot(mx, my, '*', color='lime', markersize=15, label=f'Marker "{name}"', markeredgecolor='black')
        ax.text(mx + 0.1, my, name, color='white', fontsize=12, ha='left', va='center')
    
    ax.set_xticks(range(min_x, max_x + 2)); ax.set_yticks(range(min_y, max_y + 2))
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_title('Maze Exploration Map', fontsize=16, color='white')
    ax.set_xlabel('X Coordinate', color='white'); ax.set_ylabel('Y Coordinate', color='white')
    ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    plt.legend(); plt.show()

# ==============================================================================
# 5. ส่วนหลักของโปรแกรม (Main Execution)
# ==============================================================================
def main():
    ep_robot = None
    stop_event = threading.Event()
    def on_key_press(event):
        if event.name == 'q': stop_event.set(); print("\n'q' pressed! Signaling mission to stop..."); keyboard.unhook_all()
    keyboard.on_press(on_key_press)
    try:
        config = RobotConfig(); handlers = {'odom': OdometryDataHandler(), 'tof': TofDataHandler(), 'vision': VisionDataHandler()}
        ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap"); print("Robot connected.")
        ep_robot.vision.enable_detection(name="marker"); ep_robot.chassis.set_push_freq(push_freq=50)
        print("Vision & high-freq chassis push enabled.")
        ep_robot.chassis.sub_attitude(freq=50, callback=handlers['odom'].sub_attitude)
        ep_robot.chassis.sub_position(freq=50, callback=handlers['odom'].sub_position)
        ep_robot.sensor.sub_distance(freq=10, callback=handlers['tof'].update)
        ep_robot.vision.sub_detect_info(name="marker", callback=handlers['vision'].update)
        print("Subscribed to all required sensors."); time.sleep(2)
        explorer = MazeExplorer(ep_robot, config, handlers, stop_event)
        explorer.run_mission()
        print("\nGenerating final map visualization...")
        visualize_map(explorer.internal_map, explorer.marker_map, explorer.mission_path)
    except Exception as e: print(f"\n--- An unexpected critical error occurred: {e} ---")
    finally:
        if ep_robot:
            print("Closing connection and releasing resources...")
            ep_robot.chassis.drive_speed(x=0, y=0, z=0); ep_robot.close()
            print("Cleanup complete.")
        keyboard.unhook_all()

if __name__ == '__main__':
    main()