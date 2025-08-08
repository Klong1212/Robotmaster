import time
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set, Optional, Any

# ==============================================================================
# 1. ศูนย์รวมการตั้งค่า (Centralized Configuration)
# ==============================================================================
class RobotConfig:
    GRID_SIZE_M: float = 0.6
    WALL_THRESHOLD_MM: int = 500
    TURN_TOLERANCE_DEG: float = 1.0
    MOVE_TOLERANCE_M: float = 0.02
    TURN_PID: Tuple[float, float, float] = (3.0, 0.15, 0.35)
    MOVE_PID: Tuple[float, float, float] = (2.5, 0.1, 0.25)
    HEADING_PID: Tuple[float, float, float] = (2.2, 0, 0)
    SIM_TIME_STEP_S: float = 0.01 
    MAX_TURN_SPEED_DPS: int = 200
    MAX_MOVE_SPEED_MPS: float = 1.0

Point = Tuple[int, int]
ORIENTATIONS: Dict[int, str] = {0: "North", 1: "East", 2: "South", 3: "West"}
DIRECTION_VECTORS: Dict[int, Point] = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

# ==============================================================================
# 2. คลาสสำหรับจำลองสภาพแวดล้อม
# ==============================================================================
class SimulatedMaze:
    def __init__(self):
        connections = {
            (0,0): {(1,0)},
            (1,0): {(2,0)},
            (2,0): {(2,1)},
            (2,1): {(2,2), (1,1)},
            (1,1): {(1,2)},
            (1,2): {(0,2)},
            (2,2): {(2,1)},
        }
        self.true_map: Dict[Point, Set[Point]] = {}
        for pos, neighbors in connections.items():
            self.true_map.setdefault(pos, set()).update(neighbors)
            for neighbor in neighbors:
                self.true_map.setdefault(neighbor, set()).add(pos)

        self.true_markers: Dict[Point, str] = {(2,-1): "A", (3,1): "B", (0,3): "C"}

    def check_wall(self, from_pos: Point, direction: int) -> bool:
        dx, dy = DIRECTION_VECTORS[direction]
        to_pos = (from_pos[0] + dx, from_pos[1] + dy)
        return from_pos not in self.true_map or to_pos not in self.true_map.get(from_pos, set())

    def check_markers(self, from_pos: Point, direction: int) -> List[str]:
        dx, dy = DIRECTION_VECTORS[direction]
        wall_pos = (from_pos[0] + dx, from_pos[1] + dy)
        return [self.true_markers[wall_pos]] if wall_pos in self.true_markers else []

class SimulatedRobot:
    def __init__(self, maze: SimulatedMaze, cfg: RobotConfig):
        self.maze, self.cfg = maze, cfg
        self.x: float = 0.0
        self.y: float = 0.0
        self.yaw: float = 90.0
        self.gimbal_yaw: float = 0.0

    def _get_world_scan_direction(self) -> int:
        """[FIXED] คำนวณทิศที่เซ็นเซอร์หันไปในโลกจริงให้ถูกต้อง"""
        # Yaw ของ Gimbal ใน Robomaster SDK เป็นแบบตามเข็มนาฬิกา (Clockwise)
        # Yaw ของโลกจำลองเราเป็นแบบทวนเข็ม (Counter-clockwise) จึงต้องใช้การลบ
        world_scan_yaw = (self.yaw - self.gimbal_yaw) % 360
        
        # แปลงมุมองศา เป็น direction (0=N, 1=E, 2=S, 3=W)
        # เพิ่ม 45 องศาเพื่อเลื่อนช่วงการแบ่งให้ง่ายขึ้น (0-89.9 -> East, 90-179.9 -> North)
        dir_index = round((world_scan_yaw + 45) / 90) % 4
        
        # Mapping จาก index ที่ได้ ไปยัง direction ของ Explorer
        # 0->E(1), 1->N(0), 2->W(3), 3->S(2)
        remap = {0: 1, 1: 0, 2: 3, 3: 2}
        return remap[dir_index]

    def get_tof_distance(self) -> int:
        scan_dir = self._get_world_scan_direction()
        has_wall = self.maze.check_wall((round(self.x), round(self.y)), scan_dir)
        return 100 if has_wall else 1000

    def get_vision_markers(self) -> List[str]:
        scan_dir = self._get_world_scan_direction()
        return self.maze.check_markers((round(self.x), round(self.y)), scan_dir)
    
    def get_odometry(self) -> Tuple[float, float, float]: return self.x, self.y, self.yaw
    def move_gimbal(self, yaw: float): self.gimbal_yaw = yaw
    def recenter_gimbal(self): self.gimbal_yaw = 0
    def update_state(self, vx: float, vz: float, delta_time: float):
        self.yaw = (self.yaw + vz * delta_time) % 360
        rad = math.radians(self.yaw)
        self.x += vx * math.cos(rad) * delta_time
        self.y += vx * math.sin(rad) * delta_time

# ... (คลาส PIDController, RobotMap เหมือนเดิมทุกประการ) ...
class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float, output_limits: Tuple[float, float]):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd; self.min_output, self.max_output = output_limits
        self.last_error: float = 0.0; self.integral: float = 0.0; self.last_time: float = time.time()
    def update(self, error: float) -> float:
        current_time = time.time(); delta_time = current_time - self.last_time
        if delta_time == 0: return self.min_output
        p_term = self.Kp * error; self.integral += error * delta_time
        d_term = self.Kd * ((error - self.last_error) / delta_time)
        output = p_term + (self.Ki * self.integral) + d_term
        if self.Ki != 0:
            if output > self.max_output: self.integral -= (output - self.max_output) / self.Ki; output = self.max_output
            elif output < self.min_output: self.integral -= (output - self.min_output) / self.Ki; output = self.min_output
        self.last_error = error; self.last_time = current_time; return output
    def reset(self): self.last_error = 0.0; self.integral = 0.0; self.last_time = time.time()
class RobotMap:
    def __init__(self): self.graph: Dict[Point, Set[Point]] = {}; self.explored: Set[Point] = set()
    def add_connection(self, pos1: Point, pos2: Point): self.graph.setdefault(pos1, set()).add(pos2); self.graph.setdefault(pos2, set()).add(pos1); print(f"  Map: Added connection between {pos1} and {pos2}")
    def mark_explored(self, position: Point): self.explored.add(position)
    def get_path_to_unexplored(self, start_pos: Point) -> Optional[List[Point]]:
        unexplored_adj = [n for n in self.graph.get(start_pos, set()) if n not in self.explored]
        if unexplored_adj: return [start_pos, unexplored_adj[0]]
        for pos in sorted(list(self.explored), reverse=True):
            if any(n not in self.explored for n in self.graph.get(pos, set())):
                print(f"No new paths here. Backtracking to {pos}..."); return self.get_path(start_pos, pos)
        return None
    def get_path(self, start: Point, goal: Point) -> Optional[List[Point]]:
        if start == goal: return [start]
        queue: List[Tuple[Point, List[Point]]] = [(start, [start])]; visited: Set[Point] = {start}
        while queue:
            current, path = queue.pop(0)
            for neighbor in self.graph.get(current, set()):
                if neighbor not in visited:
                    if neighbor == goal: return path + [neighbor]
                    visited.add(neighbor); queue.append((neighbor, path + [neighbor]))
        return None

# ==============================================================================
# 4. คลาสหลัก (Main Logic Class) - ฉบับ Simulation
# ==============================================================================
class MazeExplorer:
    def __init__(self, sim_robot: SimulatedRobot, cfg: RobotConfig):
        self.sim_robot, self.cfg = sim_robot, cfg
        self.current_position: Point = (0, 0); self.current_orientation: int = 0
        self.internal_map = RobotMap(); self.marker_map: Dict[str, Any] = {}
        self.mission_path: List[Point] = [(0, 0)]
        self.turn_pid = PIDController(*cfg.TURN_PID, (-cfg.MAX_TURN_SPEED_DPS, cfg.MAX_TURN_SPEED_DPS))
        self.move_pid = PIDController(*cfg.MOVE_PID, (-cfg.MAX_MOVE_SPEED_MPS, cfg.MAX_MOVE_SPEED_MPS))
        self.heading_pid = PIDController(*cfg.HEADING_PID, (-cfg.MAX_TURN_SPEED_DPS, cfg.MAX_TURN_SPEED_DPS))
        print("Maze Explorer Brain Initialized for Simulation.")
    
    def scan_surroundings(self):
        print(f"\nScanning surroundings at {self.current_position}...")
        self.internal_map.mark_explored(self.current_position)
        x, y = self.current_position
        for scan_dir in range(4):
            print(f"  Scanning direction: {ORIENTATIONS[scan_dir]}...")
            angle_to_turn_gimbal = (scan_dir - self.current_orientation) * 90
            if angle_to_turn_gimbal >= 180: angle_to_turn_gimbal -= 360
            if angle_to_turn_gimbal <= -180: angle_to_turn_gimbal += 360
            self.sim_robot.move_gimbal(angle_to_turn_gimbal)
            
            if self.sim_robot.get_tof_distance() >= self.cfg.WALL_THRESHOLD_MM:
                dx, dy = DIRECTION_VECTORS[scan_dir]
                self.internal_map.add_connection(self.current_position, (x + dx, y + dy))

            for marker_name in self.sim_robot.get_vision_markers():
                if marker_name not in self.marker_map:
                    dx, dy = DIRECTION_VECTORS[scan_dir]
                    self.marker_map[marker_name] = ((x + dx, y + dy), ORIENTATIONS[scan_dir])
                    print(f"    !!! Marker Found: '{marker_name}' !!!")
        self.sim_robot.recenter_gimbal()
        print("Scan complete.")

    def turn_with_pid(self, angle_to_turn: float):
        self.turn_pid.reset()
        _, _, start_yaw = self.sim_robot.get_odometry()
        target_yaw = start_yaw + angle_to_turn
        while True:
            _, _, current_yaw = self.sim_robot.get_odometry()
            error = target_yaw - current_yaw
            if error > 180: error -= 360
            if error < -180: error += 360
            if abs(error) < self.cfg.TURN_TOLERANCE_DEG: break
            vz_speed = self.turn_pid.update(error)
            self.sim_robot.update_state(vx=0, vz=vz_speed, delta_time=self.cfg.SIM_TIME_STEP_S)
            time.sleep(self.cfg.SIM_TIME_STEP_S)

    def move_forward_with_pid(self, distance_m: float):
        self.move_pid.reset(); self.heading_pid.reset()
        start_x, start_y, target_heading = self.sim_robot.get_odometry()
        while True:
            current_x, current_y, current_heading = self.sim_robot.get_odometry()
            distance_traveled = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
            error = distance_m - distance_traveled
            if abs(error) < self.cfg.MOVE_TOLERANCE_M: break
            vx_speed = self.move_pid.update(error)
            heading_error = target_heading - current_heading
            if heading_error > 180: heading_error -= 360
            if heading_error < -180: heading_error += 360
            vz_correct_speed = self.heading_pid.update(heading_error)
            self.sim_robot.update_state(vx=vx_speed, vz=vz_correct_speed, delta_time=self.cfg.SIM_TIME_STEP_S)
            time.sleep(self.cfg.SIM_TIME_STEP_S)

    def execute_path(self, path: List[Point]):
        print(f"Executing path: {path}")
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
                print(f"  Turning chassis {angle_to_turn} degrees to face {ORIENTATIONS[target_orientation]}...")
                self.turn_with_pid(angle_to_turn)
                self.current_orientation = target_orientation
            print(f"  Moving forward {self.cfg.GRID_SIZE_M}m...")
            self.move_forward_with_pid(self.cfg.GRID_SIZE_M)
            self.sim_robot.x, self.sim_robot.y = float(end_node[0]), float(end_node[1])
            self.current_position = end_node
            self.mission_path.append(end_node)
            print(f"  Arrived at {end_node}")

    def run_mission(self):
        print("--- Simulation Starting ---")
        while True:
            self.scan_surroundings()
            path_to_execute = self.internal_map.get_path_to_unexplored(self.current_position)
            if not path_to_execute:
                print("\n--- SIMULATION COMPLETE! All areas explored. ---")
                break
            self.execute_path(path_to_execute)

# ==============================================================================
# 5. ฟังก์ชันสร้างภาพแผนที่ (Map Visualization)
# ==============================================================================
def visualize_map(robot_map: RobotMap, marker_map: Dict[str, Any], mission_path: List[Point]):
    # ... (ส่วนนี้ทำงานถูกต้อง ไม่ต้องแก้ไข) ...
    if not robot_map.graph: print("Map data is empty."); return
    fig, ax = plt.subplots(figsize=(10, 10)); ax.set_aspect('equal'); ax.set_facecolor('#2B2B2B')
    all_nodes = list(robot_map.graph.keys()); min_x, max_x = min(p[0] for p in all_nodes) - 1, max(p[0] for p in all_nodes) + 1
    min_y, max_y = min(p[1] for p in all_nodes) - 1, max(p[1] for p in all_nodes) + 1
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x, y) not in robot_map.graph: continue
            if (x, y + 1) not in robot_map.graph.get((x,y), set()): ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], color='cyan', linewidth=3)
            if (x + 1, y) not in robot_map.graph.get((x,y), set()): ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], color='cyan', linewidth=3)
            if (x, y - 1) not in robot_map.graph.get((x,y), set()): ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color='cyan', linewidth=3)
            if (x - 1, y) not in robot_map.graph.get((x,y), set()): ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color='cyan', linewidth=3)
    if len(mission_path) > 1:
        path_x, path_y = [p[0] for p in mission_path], [p[1] for p in mission_path]
        ax.plot(path_x, path_y, 'o-', color='magenta', markersize=8, markerfacecolor='yellow', label='Robot Path')
    for name, (wall_pos, wall_orient_str) in marker_map.items():
        from_pos = mission_path[-1]
        for pos in reversed(mission_path):
            if wall_pos in [(pos[0]+dx, pos[1]+dy) for dx,dy in DIRECTION_VECTORS.values()]: from_pos = pos; break
        mx, my = (from_pos[0] + wall_pos[0]) / 2, (from_pos[1] + wall_pos[1]) / 2
        ax.plot(mx, my, '*', color='lime', markersize=15, label=f'Marker "{name}"', markeredgecolor='black')
        ax.text(mx + 0.1, my, name, color='white', fontsize=12, ha='left', va='center')
    ax.set_xticks(range(min_x, max_x + 2)); ax.set_yticks(range(min_y, max_y + 2))
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_title('Maze Exploration Map (Simulation)', fontsize=16, color='white')
    ax.set_xlabel('X Coordinate', color='white'); ax.set_ylabel('Y Coordinate', color='white')
    ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    plt.legend(); plt.show()

# ==============================================================================
# 6. ส่วนหลักของโปรแกรม (Main Execution)
# ==============================================================================
if __name__ == '__main__':
    config = RobotConfig()
    sim_maze = SimulatedMaze()
    sim_robot = SimulatedRobot(sim_maze, config)
    explorer = MazeExplorer(sim_robot, config)
    
    input("Press Enter to start the simulation...")
    explorer.run_mission()

    print("\nGenerating final map visualization...")
    visualize_map(explorer.internal_map, explorer.marker_map, explorer.mission_path)