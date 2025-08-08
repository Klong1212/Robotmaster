import time
from maze_runner import MazeExplorer, TofDataHandler, VisionDataHandler, ORIENTATIONS

# ==============================================================================
#  คลาสจำลองการทำงานของหุ่นยนต์ (Mock Robot Classes)
# ==============================================================================
class MockAction:
    """จำลอง Action ที่ต้องรอให้ทำเสร็จ"""
    def wait_for_completed(self):
        time.sleep(0.1) # จำลองว่าใช้เวลาทำงานเล็กน้อย
        return True

class MockGimbal:
    """จำลองการทำงานของ Gimbal"""
    def __init__(self, simulator):
        self.simulator = simulator
        self.yaw = 0
        self.pitch = 0

    def recenter(self):
        print("    [SIM] Gimbal: Recenter")
        self.yaw = 0
        self.pitch = 0
        # ในการจำลองจริง เราจะอัปเดตเซ็นเซอร์เมื่อมีการ moveto
        return MockAction()

    def moveto(self, yaw, pitch, yaw_speed):
        print(f"    [SIM] Gimbal: Moving to yaw={yaw}, pitch={pitch}")
        self.yaw = yaw
        self.pitch = pitch
        # จุดสำคัญ: เมื่อ Gimbal หมุนแล้ว ให้ Simulator อัปเดตข้อมูลเซ็นเซอร์ทันที
        self.simulator.update_sensors_for_gimbal(yaw)
        return MockAction()

class MockChassis:
    """จำลองการทำงานของ Chassis"""
    def __init__(self):
        # ไม่ต้องทำอะไรเป็นพิเศษ เพราะสถานะของหุ่นถูกจัดการโดย MazeExplorer
        pass

    def move(self, x=0, y=0, z=0, xy_speed=0, z_speed=0):
        if z != 0:
            print(f"    [SIM] Chassis: Turning {z} degrees.")
        if x > 0:
            print(f"    [SIM] Chassis: Moving forward {x}m.")
        return MockAction()

class MockLed:
    """จำลองการทำงานของ LED"""
    def set_led(self, r, g, b, effect="on"):
        print(f"    [SIM] LED: Set to R={r} G={g} B={b} with effect '{effect}'")

class MockRobot:
    """คลาสจำลองตัวหุ่นยนต์ RoboMaster EP ทั้งตัว"""
    def __init__(self, simulator):
        self.chassis = MockChassis()
        self.gimbal = MockGimbal(simulator) # ส่ง simulator เข้าไปเพื่อให้ gimbal อัปเดตเซ็นเซอร์ได้
        self.led = MockLed()
        # vision และ sensor ไม่ต้องมี object จริง เพราะเราจะเรียก handler โดยตรง
        self.vision = None
        self.sensor = None

# ==============================================================================
#  คลาสจัดการการจำลอง (Simulation Manager)
# ==============================================================================
class Simulator:
    def __init__(self, maze_layout, marker_layout):
        self.maze_layout = maze_layout
        self.marker_layout = marker_layout
        
        # สร้าง Handler เหมือนของจริง
        self.tof_handler = TofDataHandler()
        self.vision_handler = VisionDataHandler()
        
        # สร้างหุ่นยนต์จำลอง
        self.mock_robot = MockRobot(self)

        # สร้าง Explorer โดยใช้หุ่นจำลองและ Handler
        self.explorer = MazeExplorer(self.mock_robot, self.tof_handler, self.vision_handler)
        
        print("="*50)
        print("Starting Maze Simulation")
        print("="*50)

    def run(self):
        """เริ่มการจำลองการทำงานของภารกิจ"""
        self.explorer.run_mission()
        print("\n" + "="*50)
        print("Simulation Finished")
        print("="*50)

    def update_sensors_for_gimbal(self, gimbal_yaw_angle):
        """
        หัวใจของ Simulator: คำนวณว่าเซ็นเซอร์ควรจะเห็นอะไร
        แล้วเรียก .update() ของ handler เพื่อส่งข้อมูลให้ Explorer
        """
        robot_pos = self.explorer.current_position
        robot_orient = self.explorer.current_orientation # 0:N, 1:E, 2:S, 3:W
        
        # 1. คำนวณทิศทางที่ Gimbal มองอยู่จริงๆ (World Direction)
        # แปลงมุม Gimbal [-180, 180] เป็นทิศทาง 0-3
        gimbal_direction_offset = round(gimbal_yaw_angle / 90)
        world_scan_direction = (robot_orient + gimbal_direction_offset) % 4

        # 2. จำลอง ToF Sensor (เช็คกำแพง)
        x, y = robot_pos
        next_pos = robot_pos
        if world_scan_direction == 0: next_pos = (x, y + 1)
        elif world_scan_direction == 1: next_pos = (x + 1, y)
        elif world_scan_direction == 2: next_pos = (x, y - 1)
        elif world_scan_direction == 3: next_pos = (x - 1, y)

        # ถ้ามีทางเชื่อม = ไม่มีกำแพง
        if next_pos in self.maze_layout.get(robot_pos, set()):
            # จำลองว่าไม่มีกำแพง (ระยะทางไกล)
            simulated_tof_dist = 1000 # > WALL_THRESHOLD_MM
        else:
            # จำลองว่ามีกำแพง (ระยะทางใกล้)
            simulated_tof_dist = 200 # < WALL_THRESHOLD_MM
        
        self.tof_handler.update([simulated_tof_dist])

        # 3. จำลอง Vision Sensor (เช็ค Marker)
        simulated_markers = []
        wall_id = (robot_pos, world_scan_direction) # (ตำแหน่งที่มอง, ทิศที่มอง)
        if wall_id in self.marker_layout:
             marker_name = self.marker_layout[wall_id]
             # ส่งข้อมูลใน format เดียวกับที่ Vision sub คืนมา
             simulated_markers = [1, [marker_name, 0, 0, 0, 0]] # [count, [name, x,y,w,h]]

        self.vision_handler.update(simulated_markers)

# ==============================================================================
#  ส่วนหลักสำหรับรัน Simulation
# ==============================================================================
if __name__ == '__main__':
    # กำหนดแผนผังเขาวงกต
    # key คือ Grid (x,y), value คือ Set ของ Grid ที่เชื่อมต่อกัน
    # เขาวงกตรูปตัว T:
    # (0,2)
    #  |
    # (0,1)--(1,1)
    #  |
    # (0,0)  <-- Start
    maze = {
        (0,0): {(0,1)},
        (0,1): {(0,0), (0,2), (1,1)},
        (0,2): {(0,1)},
        (1,1): {(0,1)},
    }
    
    # กำหนดตำแหน่งของ Marker
    # key คือ (ตำแหน่ง Grid ที่ยืนมอง, ทิศที่มอง)
    # value คือ ชื่อของ Marker
    # เช่น ยืนที่ (0,2) มองไปทิศเหนือ (0) จะเจอ Marker 'Goal' (บนกำแพง)
    markers = {
        ((0,2), 0): "Goal",       # ที่ Grid(0,2) มองไปทิศเหนือ (เจอกำแพง)
        ((1,1), 1): "Danger",     # ที่ Grid(1,1) มองไปทิศตะวันออก (เจอกำแพง)
    }

    # สร้างและรัน Simulator
    my_simulation = Simulator(maze, markers)
    my_simulation.run()