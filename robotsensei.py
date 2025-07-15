# -*-coding:utf-8-*-
import time

from robomaster import robot
import pandas as pd
import threading
import copy
import robomaster
# --- 1. โครงสร้างข้อมูลสำหรับเก็บค่าล่าสุดและ list สำหรับบันทึก ---

# Dictionary สำหรับเก็บสถานะล่าสุดของเซ็นเซอร์ทั้งหมด
robot_state = {
    'pos_x': 0, 'pos_y': 0, 'pos_z': 0,
    'att_pitch': 0, 'att_roll': 0, 'att_yaw': 0,
    'imu_acc_x': 0, 'imu_acc_y': 0, 'imu_acc_z': 0,
    'imu_gyro_x': 0, 'imu_gyro_y': 0, 'imu_gyro_z': 0,
    'esc_speed_0': 0, 'esc_angle_0': 0, # มอเตอร์ 1
    'esc_speed_1': 0, 'esc_angle_1': 0, # มอเตอร์ 2
    'robot_status': 0
}

# List สำหรับเก็บข้อมูลทั้งหมดเพื่อสร้าง CSV
data_log = []

# Event สำหรับสั่งให้ Thread หยุดทำงาน
stop_event = threading.Event()


# --- 2. ฟังก์ชัน Handler แยกสำหรับแต่ละประเภทข้อมูล ---

def position_handler(position_info):
    x, y, z = position_info
    robot_state['pos_x'] = x
    robot_state['pos_y'] = y
    robot_state['pos_z'] = z

def attitude_handler(attitude_info):
    yaw, pitch, roll = attitude_info
    robot_state['att_yaw'] = yaw
    robot_state['att_pitch'] = pitch
    robot_state['att_roll'] = roll

def imu_handler(imu_info):
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = imu_info
    robot_state['imu_acc_x'] = acc_x
    robot_state['imu_acc_y'] = acc_y
    robot_state['imu_acc_z'] = acc_z
    robot_state['imu_gyro_x'] = gyro_x
    robot_state['imu_gyro_y'] = gyro_y
    robot_state['imu_gyro_z'] = gyro_z
    
def esc_handler(esc_info):
    # esc_info จะเป็น tuple ที่มีข้อมูลของมอเตอร์ 4 ตัว
    # ในที่นี้เราจะสนใจแค่ 2 ตัวแรก (ล้อหน้า)
    speed1, angle1, _, _ = esc_info[0]
    speed2, angle2, _, _ = esc_info[1]
    robot_state['esc_speed_0'] = speed1
    robot_state['esc_angle_0'] = angle1
    robot_state['esc_speed_1'] = speed2
    robot_state['esc_angle_1'] = angle2

def status_handler(status_info):
    # status_info เป็น tuple ที่มีค่าเดียว
    robot_state['robot_status'] = status_info[0]


# --- 3. ฟังก์ชันสำหรับ Thread ที่จะบันทึกข้อมูล ---

def data_logger(log_interval):
    """
    ฟังก์ชันนี้จะทำงานใน background thread
    เพื่อบันทึกสถานะของหุ่นยนต์ลง list ตามช่วงเวลาที่กำหนด
    """
    while not stop_event.is_set():
        # คัดลอกสถานะล่าสุดเพื่อป้องกันปัญหาข้อมูลเปลี่ยนขณะบันทึก
        current_log = copy.deepcopy(robot_state)
        data_log.append(current_log)
        time.sleep(log_interval) # รอตามช่วงเวลาที่กำหนด
    print("Data logger thread stopped.")


# --- 4. ส่วนการทำงานหลัก ---

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    # เริ่มการทำงานของ Thread สำหรับบันทึกข้อมูล
    # ในที่นี้จะบันทึกทุกๆ 0.1 วินาที (10 Hz)
    log_interval = 0.1
    logger_thread = threading.Thread(target=data_logger, args=(log_interval,))
    logger_thread.start()
    print("Data logger thread started.")

    # Subscribe ข้อมูลทั้งหมด โดยใช้ handler ของตัวเอง
    ep_chassis.sub_position(freq=10, callback=position_handler)
    ep_chassis.sub_attitude(freq=10, callback=attitude_handler)
    ep_chassis.sub_imu(freq=10, callback=imu_handler)
    ep_chassis.sub_esc(freq=10, callback=esc_handler)
    ep_chassis.sub_status(freq=10, callback=status_handler)
    print("Subscribed to all sensors.")
    time.sleep(1)

    # เคลื่อนที่เป็นรูปสี่เหลี่ยม
    distance_per_side = 0.6
    speed_xy = 1
    speed_z = 45
    
    print("Starting square path movement...")
    for i in range(4):
        print(f"Side {i+1}/4: Moving forward...")
        ep_chassis.move(x=distance_per_side, y=0, z=0, xy_speed=speed_xy).wait_for_completed()
        
        print(f"Side {i+1}/4: Turning...")
        ep_chassis.move(x=0, y=0, z=-90, z_speed=speed_z).wait_for_completed()
    print("Movement completed.")

    # หยุดการทำงานทั้งหมด
    time.sleep(1) # รอเก็บข้อมูลสุดท้าย
    
    # หยุด Subscribe
    ep_chassis.unsub_position()
    ep_chassis.unsub_attitude()
    ep_chassis.unsub_imu()
    ep_chassis.unsub_esc()
    ep_chassis.unsub_status()
    print("Unsubscribed from all sensors.")

    # สั่งให้ Thread หยุดและรอจนกว่าจะหยุดสนิท
    stop_event.set()
    logger_thread.join()

    # ปิดการเชื่อมต่อ
    ep_robot.close()

    # บันทึกข้อมูลลง CSV
    if data_log:
        print(f"Saving {len(data_log)} records to CSV...")
        df = pd.DataFrame(data_log)
        # จัดลำดับคอลัมน์ให้สวยงาม
        cols = ['timestamp'] + list(robot_state.keys())
        df = df[cols]
        df.to_csv("robot_full_data.csv", index=False)
        print("Data saved to robot_full_data.csv successfully.")
    else:
        print("No data was collected.")