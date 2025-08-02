from robomaster import robot
import time
import threading
import copy
import pandas as pd
import numpy as np
from collections import deque

# Define the filter classes (these were missing in your code!)
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def filter(self, value):
        self.values.append(value)
        return sum(self.values) / len(self.values)

class MedianFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def filter(self, value):
        self.values.append(value)
        sorted_values = sorted(list(self.values))
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]

class LowPassFilter:
    def __init__(self, cutoff_freq=1.0, sample_rate=20):
        # Calculate alpha for low pass filter
        dt = 1.0 / sample_rate
        rc = 1.0 / (2 * np.pi * cutoff_freq)
        self.alpha = dt / (rc + dt)
        self.prev_output = None
    
    def filter(self, value):
        if self.prev_output is None:
            self.prev_output = value
        else:
            self.prev_output = self.alpha * value + (1 - self.alpha) * self.prev_output
        return self.prev_output

data_log = []
stop_event = threading.Event()
robot_graph = {
    'tof': 500,
    'filter_MA_tof': 500,  # Changed from 1000 to match initial tof value
    'filter_ME_tof': 500,  # Changed from 1000 to match initial tof value
    'filter_LP_tof': 500   # Changed from 1000 to match initial tof value
}

# Initialize filters
filter_MA = MovingAverageFilter(window_size=5)
filter_ME = MedianFilter(window_size=5)
filter_LP = LowPassFilter(cutoff_freq=1.0, sample_rate=20)

# Keep track of data points for proper MAE calculation
data_count = 0
sum_ma = 0
sum_me = 0
sum_lp = 0

def sub_data_handler(sub_info):
    global data_count, sum_ma, sum_me, sum_lp
    
    # Get TOF distance with 100mm offset
    raw_tof = sub_info[0] - 100
    robot_graph['tof'] = raw_tof
    
    # Apply filters
    robot_graph['filter_MA_tof'] = filter_MA.filter(raw_tof)
    robot_graph['filter_ME_tof'] = filter_ME.filter(raw_tof)
    robot_graph['filter_LP_tof'] = filter_LP.filter(raw_tof)
    
    # Update MAE calculations
    data_count += 1
    sum_ma += abs(robot_graph['filter_MA_tof'] - robot_graph['tof'])
    sum_me += abs(robot_graph['filter_ME_tof'] - robot_graph['tof'])
    sum_lp += abs(robot_graph['filter_LP_tof'] - robot_graph['tof'])
    
    print(f'Raw TOF: {raw_tof:.1f}, MA: {robot_graph["filter_MA_tof"]:.1f}, ME: {robot_graph["filter_ME_tof"]:.1f}, LP: {robot_graph["filter_LP_tof"]:.1f}')

def data_logger(log_interval):
    """
    ฟังก์ชันนี้จะทํางานใน background thread
    เพื่อบันทึกสถานะของหุ่นยนต์ลง list ตามช่วงเวลาที่กําหนด
    """
    while not stop_event.is_set():
        robot_graph['timestamp'] = pd.Timestamp.now()
        current_log = copy.deepcopy(robot_graph)
        data_log.append(current_log)
        time.sleep(log_interval)
    print("Data logger thread stopped.")

# สร้างออปเจ็คและกําหนดค่าเชื่อมต่อกับหุ่นยนต์
if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_gimbal = ep_robot.gimbal
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_sensor = ep_robot.sensor
    ep_chassis = ep_robot.chassis
    
    # ปรับความถี่การเก็บข้อมูลของเซ็นเซอร์ TOF และกิมบอล
    log_interval = 0.05
    logger_thread = threading.Thread(target=data_logger, args=(log_interval,))
    logger_thread.start()
    
    # Subscribe to TOF sensor
    ep_sensor.sub_distance(freq=20, callback=sub_data_handler)
    
    # Wait a moment for initial sensor readings
    time.sleep(1)
    
    # Start moving
    t_detect = time.time()
    ep_chassis.drive_speed(x=0.25, y=0, z=0)
    print("Robot started moving...")
    
    try:
        while True:
            tof_distance_MA = robot_graph['filter_MA_tof']
            tof_distance_ME = robot_graph['filter_ME_tof']
            tof_distance_LP = robot_graph['filter_LP_tof']
            raw_distance = robot_graph['tof']
            
            print(f"Raw: {raw_distance:.1f} mm | MA: {tof_distance_MA:.1f} mm | ME: {tof_distance_ME:.1f} mm | LP: {tof_distance_LP:.1f} mm")
            if tof_distance_MA <= 400:
                t_stop_command = time.time()
                response_time = t_stop_command - t_detect
                print(f"Obstacle detected at {tof_distance_MA:.1f} mm, stopping robot...")
            if tof_distance_ME <= 400:
                t_stop_command = time.time()
                response_time = t_stop_command - t_detect
                print(f"Obstacle detected at {tof_distance_ME:.1f} mm, stopping robot...")
            if tof_distance_LP <= 400:
                t_stop_command = time.time()
                response_time = t_stop_command - t_detect
                print(f"Obstacle detected at {tof_distance_LP:.1f} mm, stopping robot...")
            # Check if all filters detect obstacle within 400mm
            if tof_distance_MA <= 400 and tof_distance_ME <= 400 and tof_distance_LP <= 400:
                # Calculate MAE properly using data_count
                if data_count > 0:
                    MAE_ma = sum_ma / data_count
                    MAE_me = sum_me / data_count
                    MAE_lp = sum_lp / data_count
                else:
                    MAE_ma = MAE_me = MAE_lp = 0
                
                ep_chassis.drive_speed(x=0, y=0, z=0)  # สั่งหยุดด้วยความเร็ว 0
                ep_chassis.move(x=0, y=0, z=0).wait_for_completed()  # รอให้หยุดสนิท
                ep_chassis.sub_position(freq=50, callback=None)  #ยกเลิกการติดตามตำแหน่ง
                ep_chassis.stick_overlay(0, 0, 0, 0)  # ยกเลิกคำสั่งที่ค้างอยู่
                
                # เพิ่มแรงต้านเพื่อหยุด
                ep_chassis.drive_wheels(0, 0, 0, 0)  # หยุดล้อทั้งหมด
                
                
                print(f"\n{'='*50}")
                print("OBSTACLE DETECTED - ROBOT STOPPED")
                print(f"{'='*50}")
                print(f"Stopped! Response time: {response_time:.4f} seconds")
                print(f"Data points collected: {data_count}")
                print(f"Mean Absolute Error of Moving Average filter: {MAE_ma:.2f} mm")
                print(f"Mean Absolute Error of Median filter: {MAE_me:.2f} mm")
                print(f"Mean Absolute Error of Low Pass filter: {MAE_lp:.2f} mm")
                
                # Find best filter
                filters = [("Moving Average", MAE_ma), ("Median", MAE_me), ("Low Pass", MAE_lp)]
                best_filter = min(filters, key=lambda x: x[1])
                print(f"Best filter: {best_filter[0]} with MAE: {best_filter[1]:.2f} mm")
                
                break
                
            time.sleep(0.1)  # Check every 100ms (not 250ms for better responsiveness)
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        ep_chassis.drive_speed(x=0, y=0, z=0)
        
    finally:
        print("\nCleaning up...")
        stop_event.set()
        logger_thread.join()
        ep_sensor.unsub_distance()  # Unsubscribe from sensor
        ep_robot.close()
        
        # บันทึกข้อมูลที่เก็บรวบรวมลงในไฟล์ CSV
        if data_log:
            df = pd.DataFrame(data_log)
            # จัดลําดับคอลัมน์ให้สวยงาม
            cols = ['timestamp', 'tof', 'filter_MA_tof', 'filter_ME_tof', 'filter_LP_tof']
            df = df[cols]
            df.to_csv("distance_speed_filter.csv", index=False)
            print(f"Data saved to distance_speed.csv successfully. ({len(data_log)} records)")
        else:
            print("No data to save.")
        
        print("Program completed.")
# end Raw TOF: 351.0, MA: 376.8, ME: 379.0, LP: 391.7           