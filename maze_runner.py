import time
import robomaster
from robomaster import robot

# ==============================================================================
# การตั้งค่าและค่าคงที่ (Constants)
# ==============================================================================
GRID_SIZE_M = 0.6          # ขนาดของช่องในเขาวงกต (เมตร)
WALL_THRESHOLD_MM = 500    # ระยะที่ถือว่าเป็นกำแพง (มิลลิเมตร) < 50 ซม.
VISION_SCAN_DURATION_S = 1.0 # เวลาที่ใช้ในการหยุดมองหา Marker (วินาที)
GIMBAL_TURN_SPEED = 120    # ความเร็วในการหมุน Gimbal (องศา/วินาที)
TURN_SPEED = 45            # ความเร็วในการหมุนตัว (องศา/วินาที)
MOVE_SPEED = 0.5           # ความเร็วในการเคลื่อนที่ (เมตร/วินาที)

# ทิศทาง: 0=North (+y), 1=East (+x), 2=South (-y), 3=West (-x)
ORIENTATIONS = {0: "North", 1: "East", 2: "South", 3: "West"}
# ชื่อเรียกกำแพงของ Grid ที่อยู่ "ข้างหน้า" ตามทิศที่สแกน
# เช่น ถ้าสแกนไปทิศเหนือ (North) จะไปเจอกำแพงด้านใต้ (South Wall) ของ Grid ถัดไป
WALL_NAMES = {0: "South Wall", 1: "West Wall", 2: "North Wall", 3: "East Wall"}

# ==============================================================================
# คลาสจัดการข้อมูลและแผนที่ (Data Handlers & Map)
# ==============================================================================
class TofDataHandler:
    def __init__(self):
        self.distance = 0
    def update(self, sub_info):
        self.distance = sub_info[0]
    def get_distance(self):
        return self.distance

class VisionDataHandler:
    def __init__(self):
        self.markers = []
    def update(self, vision_info):
        # vision_info[0] คือจำนวนที่เจอ, vision_info[1:] คือข้อมูลของแต่ละอัน
        self.markers = [info[0] for info in vision_info[1:]] if vision_info[0] > 0 else []
    def get_markers(self):
        return self.markers

class RobotMap:
    def __init__(self):
        self.graph = {}
        self.explored = set()
    def add_connection(self, pos1, pos2):
        if pos1 not in self.graph: self.graph[pos1] = set()
        if pos2 not in self.graph: self.graph[pos2] = set()
        self.graph[pos1].add(pos2)
        self.graph[pos2].add(pos1)
        print(f"    Map: Added connection between {pos1} and {pos2}")
    def mark_explored(self, position):
        self.explored.add(position)
    def get_unexplored_neighbors(self, position):
        if position not in self.graph: return []
        return [n for n in self.graph.get(position, set()) if n not in self.explored]
    def get_path(self, start, goal):
        if start == goal: return [start]
        queue = [(start, [start])]
        visited = {start}
        while queue:
            current, path = queue.pop(0)
            for neighbor in self.graph.get(current, set()):
                if neighbor not in visited:
                    if neighbor == goal: return path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

# ==============================================================================
# คลาสหลักสำหรับควบคุมตรรกะของหุ่นยนต์ (MazeExplorer)
# ==============================================================================
class MazeExplorer:
    def __init__(self, ep_robot, tof_handler, vision_handler):
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_led = ep_robot.led
        self.ep_vision = ep_robot.vision
        self.ep_gimbal = ep_robot.gimbal

        self.tof_handler = tof_handler
        self.vision_handler = vision_handler

        self.current_position = (0, 0)
        self.current_orientation = 0
        self.internal_map = RobotMap()
        # [IMPROVEMENT] เปลี่ยน marker_map ให้เก็บ List ของตำแหน่งที่พบ
        # เพื่อรองรับกรณีเจอ Marker ชื่อเดียวกันหลายที่
        self.marker_map = {}

        print("Robot Explorer Initialized. Starting at (0,0), facing North.")
        self.ep_led.set_led(r=0, g=0, b=255)
        self.ep_gimbal.recenter().wait_for_completed()

    def scan_surroundings_with_gimbal(self):
        print(f"\nScanning surroundings at {self.current_position} with Gimbal...")
        self.ep_led.set_led(r=255, g=255, b=0, effect="breathing")
        self.internal_map.mark_explored(self.current_position)
        x, y = self.current_position

        for scan_direction in range(4):
            print(f"  Scanning direction: {ORIENTATIONS[scan_direction]}...")
            
            angle_to_turn_gimbal = (scan_direction - self.current_orientation) * 90
            # แปลงมุมให้อยู่ในช่วง [-180, 180] ตามที่ gimbal.moveto ต้องการ
            if angle_to_turn_gimbal > 180: angle_to_turn_gimbal -= 360
            if angle_to_turn_gimbal < -180: angle_to_turn_gimbal += 360

            self.ep_gimbal.moveto(
                yaw=angle_to_turn_gimbal, pitch=0, yaw_speed=GIMBAL_TURN_SPEED
            ).wait_for_completed()

            # --- 1. Wall Scanning ---
            time.sleep(0.5) # รอให้ค่า ToF นิ่ง
            distance_mm = self.tof_handler.get_distance()
            print(f"    - ToF distance: {distance_mm} mm")

            if distance_mm >= WALL_THRESHOLD_MM:
                if scan_direction == 0:   # North
                    self.internal_map.add_connection((x, y), (x, y + 1))
                elif scan_direction == 1: # East
                    self.internal_map.add_connection((x, y), (x + 1, y))
                elif scan_direction == 2: # South
                    self.internal_map.add_connection((x, y), (x, y - 1))
                elif scan_direction == 3: # West
                    self.internal_map.add_connection((x, y), (x - 1, y))

            # --- 2. Marker Scanning ---
            # [CRITICAL FIX] เคลียร์ข้อมูล vision เก่าก่อนจะอ่านค่าใหม่
            # เพื่อป้องกันการนำข้อมูลจากทิศทางเก่ามาใช้ในทิศทางปัจจุบัน
            self.vision_handler.markers.clear()
            time.sleep(VISION_SCAN_DURATION_S)
            detected_markers = self.vision_handler.get_markers()

            if detected_markers:
                for marker_name in detected_markers:
                    wall_grid_pos, wall_direction_name = None, "Unknown"
                    if scan_direction == 0:   # North
                        wall_grid_pos, wall_direction_name = (x, y + 1), WALL_NAMES[0]
                    elif scan_direction == 1: # East
                        wall_grid_pos, wall_direction_name = (x + 1, y), WALL_NAMES[1]
                    elif scan_direction == 2: # South
                        wall_grid_pos, wall_direction_name = (x, y - 1), WALL_NAMES[2]
                    elif scan_direction == 3: # West
                        wall_grid_pos, wall_direction_name = (x - 1, y), WALL_NAMES[3]
                    
                    finding = (wall_grid_pos, wall_direction_name)
                    
                    # [IMPROVEMENT] เพิ่มตำแหน่งที่พบเข้าไปใน List
                    if marker_name not in self.marker_map:
                        self.marker_map[marker_name] = []
                    
                    # ตรวจสอบว่าเคยบันทึกตำแหน่งนี้สำหรับ Marker นี้แล้วหรือยัง
                    if finding not in self.marker_map[marker_name]:
                        self.marker_map[marker_name].append(finding)
                        print(f"    !!! Marker Found & Logged: '{marker_name}' at Grid {finding[0]} on the {finding[1]} !!!")

        self.ep_gimbal.recenter().wait_for_completed()
        print("Scan complete. Gimbal recentered.")

    def decide_next_path(self):
        unexplored = self.internal_map.get_unexplored_neighbors(self.current_position)
        if unexplored:
            return [self.current_position, unexplored[0]]
        
        # Backtracking: หา Grid ที่เคยไปแล้วแต่ยังมีทางที่ยังไม่ได้สำรวจ
        for pos in reversed(list(self.internal_map.explored)):
            if self.internal_map.get_unexplored_neighbors(pos):
                print(f"No new paths here. Backtracking to find an unexplored path from {pos}...")
                return self.internal_map.get_path(self.current_position, pos)
        
        return None

    def execute_path(self, path):
        if not path or len(path) < 2: return
        
        print(f"Executing path: {path}")
        self.ep_led.set_led(r=0, g=0, b=255)
        
        for i in range(len(path) - 1):
            start_node, end_node = path[i], path[i+1]
            dx, dy = end_node[0] - start_node[0], end_node[1] - start_node[1]
            
            target_orientation = -1
            if dx == 0 and dy == 1: target_orientation = 0   # North
            elif dx == 1 and dy == 0: target_orientation = 1   # East
            elif dx == 0 and dy == -1: target_orientation = 2  # South
            elif dx == -1 and dy == 0: target_orientation = 3  # West
            
            angle_to_turn = (target_orientation - self.current_orientation) * 90
            if angle_to_turn > 180: angle_to_turn -= 360
            if angle_to_turn < -180: angle_to_turn += 360
            
            if angle_to_turn != 0:
                print(f"  Turning chassis {angle_to_turn} degrees to face {ORIENTATIONS[target_orientation]}.")
                self.ep_chassis.move(z=angle_to_turn, z_speed=TURN_SPEED).wait_for_completed()
                self.current_orientation = target_orientation
            
            print(f"  Moving forward {GRID_SIZE_M}m to {end_node}.")
            self.ep_chassis.move(x=GRID_SIZE_M, xy_speed=MOVE_SPEED).wait_for_completed()
            self.current_position = end_node
            time.sleep(0.5)

    def run_mission(self):
        start_time = time.time()
        time_limit_seconds = 240
        print(f"Mission started! Time limit: {time_limit_seconds} seconds.")

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit_seconds:
                print(f"\n--- TIME'S UP! ({int(elapsed_time)}s elapsed) ---")
                self.ep_led.set_led(r=255, g=193, b=7, effect="flash")
                break

            self.scan_surroundings_with_gimbal()
            path_to_execute = self.decide_next_path()

            if not path_to_execute:
                print("\n--- MISSION COMPLETE! All areas explored. ---")
                self.ep_led.set_led(r=0, g=255, b=0, effect="on")
                break
            
            self.execute_path(path_to_execute)
        
        print("\n--- Final Marker Map ---")
        if self.marker_map:
            for name, findings in sorted(self.marker_map.items()):
                print(f"  Marker '{name}':")
                for details in findings:
                    print(f"    - Found at Grid={details[0]}, Wall={details[1]}")
        else:
            print("  No markers were logged.")


# ==============================================================================
# ส่วนหลักของโปรแกรม (Main Execution)
# ==============================================================================
if __name__ == '__main__':
    ep_robot = None
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        print("Robot connected.")

        # เปิดการใช้งาน Vision และ ToF
        ep_robot.vision.enable_detection(name="marker")
        print("Vision Marker detection enabled.")
        tof_handler = TofDataHandler()
        vision_handler = VisionDataHandler()
        ep_robot.sensor.sub_distance(freq=10, callback=tof_handler.update)
        ep_robot.vision.sub_detect_info(name="marker", callback=vision_handler.update)
        print("Subscribed to front TOF and Vision sensors.")
        time.sleep(1)

        # เริ่มภารกิจ
        explorer = MazeExplorer(ep_robot, tof_handler, vision_handler)
        explorer.run_mission()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ep_robot:
            # ปิดการเชื่อมต่อและคืนทรัพยากรเสมอ
            ep_robot.sensor.unsub_distance()
            ep_robot.vision.unsub_detect_info(name="marker")
            ep_robot.close()
            print("Resources released and connection closed.")