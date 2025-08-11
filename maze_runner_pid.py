import time
import threading
import math

from robomaster import robot, vision
import matplotlib.pyplot as plt

# ==============================================================================
# Constants
# ==============================================================================
GRID_SIZE_M = 0.6
WALL_THRESHOLD_MM = 500
VISION_SCAN_DURATION_S = 0.6
GIMBAL_TURN_SPEED = 200

ORIENTATIONS = {0: "North", 1: "East", 2: "South", 3: "West"}
WALL_NAMES = {0: "North Wall", 1: "East Wall", 2: "South Wall", 3: "West Wall"}

# ==============================================================================
# Data Handlers
# ==============================================================================
class TofDataHandler:
    def __init__(self):
        self.distance = 0
        self._lock = threading.Lock()

    def update(self, sub_info):
        with self._lock:
            try:
                self.distance = int(sub_info[0])
            except Exception:
                self.distance = 0

    def get_distance(self):
        with self._lock:
            return self.distance


class VisionDataHandler:
    """Parse marker info from vision callback.
    Expected like: [count, id1, x1, y1, w1, h1, id2, x2, ...]. We keep only ids.
    """
    def __init__(self):
        self.markers = []
        self._lock = threading.Lock()

    def update(self, vision_info):
        with self._lock:
            try:
                if not vision_info:
                    self.markers = []
                    return
                count = int(vision_info[0])
                found = []
                base = 1
                stride = 5  # id, x, y, w, h (SDKs vary slightly; keep robust)
                for i in range(count):
                    idx = base + i * (stride + 1)
                    if idx >= len(vision_info):
                        idx = base + i * stride
                    try:
                        found.append(str(vision_info[idx]))
                    except Exception:
                        pass
                self.markers = found
            except Exception:
                self.markers = []

    def get_markers(self):
        with self._lock:
            return list(self.markers)

    def clear(self):
        with self._lock:
            self.markers = []


class PoseDataHandler:
    def __init__(self):
        self.pose = [0.0] * 6  # x, y, z, yaw, pitch, roll
        self._lock = threading.Lock()

    def update_position(self, pos_info):
        with self._lock:
            self.pose[0], self.pose[1], self.pose[2] = float(pos_info[0]), float(pos_info[1]), float(pos_info[2])

    def update_attitude(self, att_info):
        with self._lock:
            self.pose[3], self.pose[4], self.pose[5] = float(att_info[0]), float(att_info[1]), float(att_info[2])

    def get_pose(self):
        with self._lock:
            return tuple(self.pose)

    def set_xy(self, x_m, y_m):
        with self._lock:
            self.pose[0], self.pose[1] = float(x_m), float(y_m)

    def set_yaw(self, yaw_deg):
        with self._lock:
            self.pose[3] = float(yaw_deg)


# ==============================================================================
# Map + Walls
# ==============================================================================
class RobotMap:
    def __init__(self):
        self.graph = {}          # adjacency between grid centers (x, y)
        self.explored = set()    # explored grid cells
        self.walls = set()       # set of segments: ((x1, y1), (x2, y2)) in grid coords

    def _norm_seg(self, p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        return (p1, p2) if (x1, y1) <= (x2, y2) else (p2, p1)

    def wall_segment_for(self, cell, direction):
        x, y = cell
        # walls are drawn on half-offsets around the cell center
        if direction == 0:   # North
            return self._norm_seg((x - 0.5, y + 0.5), (x + 0.5, y + 0.5))
        if direction == 1:   # East
            return self._norm_seg((x + 0.5, y - 0.5), (x + 0.5, y + 0.5))
        if direction == 2:   # South
            return self._norm_seg((x - 0.5, y - 0.5), (x + 0.5, y - 0.5))
        if direction == 3:   # West
            return self._norm_seg((x - 0.5, y - 0.5), (x - 0.5, y + 0.5))
        raise ValueError("direction must be 0..3")

    def add_wall(self, cell, direction):
        seg = self.wall_segment_for(cell, direction)
        if seg not in self.walls:
            self.walls.add(seg)
            print(f"      Map: Added WALL at {seg}")

    def add_connection(self, pos1, pos2):
        if pos1 not in self.graph:
            self.graph[pos1] = set()
        if pos2 not in self.graph:
            self.graph[pos2] = set()
        if pos2 not in self.graph[pos1]:
            self.graph[pos1].add(pos2)
            self.graph[pos2].add(pos1)
            print(f"      Map: Added connection between {pos1} and {pos2}")

    def mark_explored(self, position):
        self.explored.add(position)

    def get_unexplored_neighbors(self, position):
        if position not in self.graph:
            return []
        return [n for n in self.graph.get(position, []) if n not in self.explored]

    def get_path(self, start, goal):
        if start == goal:
            return [start]
        queue = [(start, [start])]
        visited = {start}
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


# ==============================================================================
# Explorer
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
        self.current_orientation = 0  # 0:N, 1:E, 2:S, 3:W
        self.internal_map = RobotMap()
        self.marker_map = {}
        self.visited_path = [self.current_position]

        print("Robot Explorer Initialized. Starting at (0,0), facing North.")
        self.ep_led.set_led(r=0, g=0, b=255)
        self.ep_gimbal.recenter().wait_for_completed()
        self.pose_handler.set_xy(0.0, 0.0)
        self.pose_handler.set_yaw(0.0)

    def run_mission(self):
        start_time = time.time()
        time_limit_seconds = 600
        print(f"Mission started! Time limit: {time_limit_seconds} seconds.")

        while True:
            if time.time() - start_time >= time_limit_seconds:
                print("--- TIME'S UP! ---")
                self.ep_led.set_led(r=255, g=193, b=7, effect="flash")
                break

            if self.current_position not in self.internal_map.explored:
                previous_pos = self.visited_path[-2] if len(self.visited_path) > 1 else None
                wall_distances = self.scan_surroundings_with_gimbal(previous_position=previous_pos)
                self.recenter_robot(wall_distances)
                self.correct_orientation_with_wall(wall_distances)
            else:
                print(f"Position {self.current_position} already explored. Skipping scan.")

            path_to_execute = self.decide_next_path()
            if not path_to_execute:
                print("--- MISSION COMPLETE! All areas explored. ---")
                self.ep_led.set_led(r=0, g=255, b=0, effect="on")
                break
            self.execute_path(path_to_execute)

        print("--- Final Marker Map ---")
        if self.marker_map:
            for name, findings in sorted(self.marker_map.items()):
                print(f"   Marker '{name}':")
                for details in findings:
                    print(f"         - Found at Grid={details[0]}, Wall={details[1]}")
        else:
            print("   No markers were logged.")

        plot_map_and_path(self.internal_map.graph, self.internal_map.walls, self.visited_path)

    def scan_surroundings_with_gimbal(self, previous_position=None):
        print(f"Scanning surroundings at {self.current_position} with Gimbal...")
        self.ep_led.set_led(r=255, g=255, b=0, effect="breathing")
        self.internal_map.mark_explored(self.current_position)
        x, y = self.current_position

        wall_distances = {}

        direction_to_skip = -1
        if previous_position:
            print(f"   -> Path from {previous_position} is known. Adding connection automatically.")
            self.internal_map.add_connection(self.current_position, previous_position)
            direction_to_skip = (self.current_orientation + 2) % 4
            print(f"   -> Skip scanning {ORIENTATIONS.get(direction_to_skip)}.")

        for scan_direction in range(4):
            if scan_direction == direction_to_skip:
                continue

            neighbor_pos = (
                (x, y + 1) if scan_direction == 0 else
                (x + 1, y) if scan_direction == 1 else
                (x, y - 1) if scan_direction == 2 else
                (x - 1, y)
            )

            if neighbor_pos in self.internal_map.explored:
                print(f"   -> Neighbor {neighbor_pos} ({ORIENTATIONS[scan_direction]}) already explored. Skip scan.")
                continue

            print(f"   Scanning {ORIENTATIONS[scan_direction]}...")
            angle_to_turn_gimbal = (scan_direction - self.current_orientation) * 90
            if angle_to_turn_gimbal > 180:
                angle_to_turn_gimbal -= 360
            if angle_to_turn_gimbal < -180:
                angle_to_turn_gimbal += 360
            self.ep_gimbal.moveto(yaw=angle_to_turn_gimbal, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()

            self.vision_handler.clear()
            time.sleep(VISION_SCAN_DURATION_S)

            distance_mm = self.tof_handler.get_distance()
            print(f"         - ToF distance: {distance_mm} mm")
            wall_distances[str(scan_direction)] = distance_mm

            if distance_mm >= WALL_THRESHOLD_MM:
                self.internal_map.add_connection(self.current_position, neighbor_pos)
            else:
                self.internal_map.add_wall(self.current_position, scan_direction)

            detected_markers = self.vision_handler.get_markers()
            print(f"         - Markers: {detected_markers}")
            if detected_markers and distance_mm < WALL_THRESHOLD_MM:
                for marker_name in detected_markers:
                    wall_name = WALL_NAMES.get(scan_direction, "Unknown")
                    finding = (self.current_position, wall_name)
                    self.marker_map.setdefault(marker_name, [])
                    if finding not in self.marker_map[marker_name]:
                        self.marker_map[marker_name].append(finding)
                        print(f"         !!! Marker Found & Logged: '{marker_name}' at Grid {finding[0]} on the {finding[1]} !!!")

        self.ep_gimbal.recenter().wait_for_completed()
        print("Scan complete. Gimbal recentered.")
        return wall_distances

    # ----------------------------- Centering ---------------------------------
    def recenter_robot(self, wall_distances):
        print("   -> Recenter using ToF data...")
        x_err_mm = 0.0  # world X (east +)
        y_err_mm = 0.0  # world Y (north +)

        # East-West (dirs 1,3) → world X
        if '1' in wall_distances and '3' in wall_distances:
            dist_e, dist_w = wall_distances['1'], wall_distances['3']
            if dist_e < WALL_THRESHOLD_MM and dist_w < WALL_THRESHOLD_MM:
                x_err_mm = (dist_e - dist_w) / 2.0
            elif dist_e < WALL_THRESHOLD_MM:
                x_err_mm = 200 - dist_e
            elif dist_w < WALL_THRESHOLD_MM:
                x_err_mm = -(200 - dist_w)

        # North-South (dirs 0,2) → world Y
        if '0' in wall_distances and '2' in wall_distances:
            dist_n, dist_s = wall_distances['0'], wall_distances['2']
            if dist_n < WALL_THRESHOLD_MM and dist_s < WALL_THRESHOLD_MM:
                y_err_mm = (dist_n - dist_s) / 2.0
            elif dist_n < WALL_THRESHOLD_MM:
                y_err_mm = 200 - dist_n
            elif dist_s < WALL_THRESHOLD_MM:
                y_err_mm = -(200 - dist_s)

        dx = x_err_mm / 1000.0
        dy = y_err_mm / 1000.0
        if abs(dx) < 0.005 and abs(dy) < 0.005:
            print("   Robot is well-centered. No adjustment needed.")
            return

        # Map desired world delta (dx, dy) → body frame (bx, by)
        ori = self.current_orientation % 4
        if ori == 0:         # Facing North: world X = -by, world Y = bx
            bx, by = dy, -dx
        elif ori == 1:       # Facing East:  world X = bx,  world Y = by
            bx, by = dx, dy
        elif ori == 2:       # Facing South: world X = by,  world Y = -bx
            bx, by = -dy, dx
        else:                # Facing West:  world X = -bx, world Y = -by
            bx, by = -dx, -dy

        print(f"   Applying offset movement (world): dx={dx:.3f}m, dy={dy:.3f}m → (body): x={bx:.3f}m, y={by:.3f}m")
        try:
            self.ep_chassis.move(x=bx, y=by, z=0, xy_speed=0.2).wait_for_completed()
            time.sleep(0.3)
        except Exception as e:
            print(f"   Offset move failed: {e}")

    # -------------------------- Orientation trim -----------------------------
    def correct_orientation_with_wall(self, wall_distances):
        front = self.current_orientation % 4
        if str(front) not in wall_distances or wall_distances[str(front)] >= WALL_THRESHOLD_MM:
            print("   -> Orientation Correction: No wall in front to calibrate with.")
            return

        print("   -> Correcting orientation using front wall...")
        self.ep_led.set_led(r=139, g=0, b=255, effect="breathing")

        SWEEP_ANGLE_DEG = 5
        SWEEP_SPEED_DPS = 20
        min_dist_mm = float('inf')
        angle_at_min_dist = 0

        self.ep_gimbal.moveto(yaw=-SWEEP_ANGLE_DEG, pitch=0, yaw_speed=SWEEP_SPEED_DPS).wait_for_completed()
        time.sleep(0.3)
        self.ep_gimbal.moveto(yaw=SWEEP_ANGLE_DEG, pitch=0, yaw_speed=SWEEP_SPEED_DPS).wait_for_completed()

        start = time.time()
        sweep_duration = (2 * SWEEP_ANGLE_DEG / SWEEP_SPEED_DPS) + 0.5
        while time.time() - start < sweep_duration:
            try:
                att = self.ep_gimbal.get_attitude(mode="relative_to_chassis")
                gimbal_yaw = float(att[0]) if isinstance(att, (list, tuple)) and len(att) else 0.0
                d = self.tof_handler.get_distance()
                if d < min_dist_mm:
                    min_dist_mm = d
                    angle_at_min_dist = gimbal_yaw
                time.sleep(0.01)
            except Exception:
                pass

        self.ep_gimbal.recenter().wait_for_completed()

        error_angle = angle_at_min_dist
        print(f"      - Min distance {min_dist_mm}mm at gimbal angle {error_angle:.2f}°")
        if abs(error_angle) < 0.5:
            print("      - Orientation is acceptable. No correction needed.")
            self.ep_led.set_led(r=0, g=0, b=255)
            return

        _, _, _, current_yaw, _, _ = self.pose_handler.get_pose()
        correction_angle = current_yaw + error_angle
        if correction_angle > 180:
            correction_angle -= 360
        if correction_angle < -180:
            correction_angle += 360

        print(f"      - Chassis yaw error {error_angle:.2f}°. Correcting...")
        self.turn_pid(correction_angle)

        target_yaw = [0, 90, 180, -90][self.current_orientation % 4]
        self.pose_handler.set_yaw(target_yaw)
        print(f"      - Orientation corrected. Yaw reset to {target_yaw}°.")
        self.ep_led.set_led(r=0, g=0, b=255)

    # -------------------------- Motion primitives ----------------------------
    def execute_path(self, path):
        if not path or len(path) < 2:
            return
        print(f"Executing path with PID: {path}")
        self.ep_led.set_led(r=0, g=0, b=255)
        for i in range(len(path) - 1):
            start_node, end_node = path[i], path[i + 1]
            dx, dy = end_node[0] - start_node[0], end_node[1] - start_node[1]

            target_orientation = 0 if (dx == 0 and dy == 1) else 1 if (dx == 1 and dy == 0) else 2 if (dx == 0 and dy == -1) else 3
            target_angle = [0, 90, 180, -90][target_orientation]

            self.turn_pid(target_angle)
            self.current_orientation = target_orientation
            time.sleep(0.2)
            self.move_forward_pid(GRID_SIZE_M)
            self.current_position = end_node
            self.visited_path.append(self.current_position)

            self.pose_handler.set_xy(end_node[0] * GRID_SIZE_M, end_node[1] * GRID_SIZE_M)
            self.pose_handler.set_yaw(target_angle)
            time.sleep(0.2)

    def decide_next_path(self):
        unexplored = self.internal_map.get_unexplored_neighbors(self.current_position)
        if unexplored:
            return [self.current_position, unexplored[0]]
        for pos in reversed(self.visited_path):
            if self.internal_map.get_unexplored_neighbors(pos):
                print(f"No new paths here. Backtracking to {pos}...")
                return self.internal_map.get_path(self.current_position, pos)
        return None

    def move_forward_pid(self, distance_m, speed_limit=2.5):
        print(f"   PID Move: {distance_m}m forward.")
        pid = PIDController(Kp=2.5, Ki=0.1, Kd=0.8, setpoint=distance_m, output_limits=(-speed_limit, speed_limit))
        start_x, start_y, _, _, _, _ = self.pose_handler.get_pose()
        while True:
            curr_x, curr_y, _, _, _, _ = self.pose_handler.get_pose()
            dist_traveled = math.hypot(curr_x - start_x, curr_y - start_y)
            if abs(distance_m - dist_traveled) < 0.01:
                break
            vx_speed = pid.update(dist_traveled)
            self.ep_chassis.drive_speed(x=vx_speed, y=0, z=0, timeout=0.1)
            time.sleep(0.01)
        self.ep_chassis.drive_speed(0, 0, 0)
        print("   PID Move: Completed.")

    def turn_pid(self, target_angle, speed_limit=180):
        print(f"   PID Turn: to {target_angle}°.")
        pid = PIDController(Kp=1.8, Ki=0.1, Kd=0.8, setpoint=0, output_limits=(-speed_limit, speed_limit))
        while True:
            _, _, _, current_yaw, _, _ = self.pose_handler.get_pose()
            error = target_angle - current_yaw
            if error > 180:
                error -= 360
            if error < -180:
                error += 360
            if abs(error) < 0.5:
                break
            vz_speed = pid.update(-error)
            self.ep_chassis.drive_speed(x=0, y=0, z=vz_speed, timeout=0.1)
            time.sleep(0.01)
        self.ep_gimbal.recenter().wait_for_completed()
        self.ep_chassis.drive_speed(0, 0, 0)
        print("   PID Turn: Completed.")


# ==============================================================================
# Plotting (now includes walls)
# ==============================================================================
def plot_map_and_path(graph, walls, visited_path, filename='maze_map_pid.png'):
    plt.figure(figsize=(8, 8))
    plt.title('Robot Map – Connections, Walls & Path (PID)')

    # Draw connections (open passages) between centers
    for node, neighbors in graph.items():
        x1, y1 = node
        for nb in neighbors:
            x2, y2 = nb
            plt.plot([x1, x2], [y1, y2])

    # Draw walls as thick black lines
    if walls:
        for (p1, p2) in walls:
            (x1, y1), (x2, y2) = p1, p2
            plt.plot([x1, x2], [y1, y2], linewidth=3)

    # Draw nodes and path
    if graph:
        xs, ys = zip(*graph.keys())
        plt.scatter(xs, ys, s=40)

    if visited_path:
        px, py = [p[0] for p in visited_path], [p[1] for p in visited_path]
        plt.plot(px, py, linewidth=2)
        plt.scatter(px[0], py[0], s=120, marker='o')
        plt.scatter(px[-1], py[-1], s=120, marker='X')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.savefig(filename, dpi=150)
    print(f"Map saved to '{filename}'")
    plt.close()


# ==============================================================================
# Main
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
        try:
            ep_robot.vision.enable_detection(vision.DETECT_MARKER)
        except Exception:
            pass

        ep_robot.sensor.sub_distance(freq=10, callback=tof_handler.update)
        ep_robot.vision.sub_detect_info(name="marker", callback=vision_handler.update)
        ep_robot.chassis.sub_position(freq=20, callback=pose_handler.update_position)
        ep_robot.chassis.sub_attitude(freq=20, callback=pose_handler.update_attitude)

        print("Subscribed to all required sensors.")
        time.sleep(1.0)

        explorer = MazeExplorer(ep_robot, tof_handler, vision_handler, pose_handler)
        explorer.run_mission()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if ep_robot:
            for fn in (
                lambda: ep_robot.sensor.unsub_distance(),
                lambda: ep_robot.vision.unsub_detect_info(name="marker"),
                lambda: ep_robot.chassis.sub_position(0, None),  # quick disable if API supports
                lambda: ep_robot.chassis.unsub_position(),
                lambda: ep_robot.chassis.unsub_attitude(),
                lambda: ep_robot.vision.disable_detection(vision.DETECT_MARKER),
            ):
                try:
                    fn()
                except Exception:
                    pass
            ep_robot.close()
            print("break")
