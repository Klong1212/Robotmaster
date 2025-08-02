from robomaster import robot
import pandas as pd
import time

# เก็บข้อมูลทั้งหมดใน list
data_log = []

# เริ่มนับเวลา
start_time = time.time()

# ตัวแปร global สำหรับข้อมูลล่าสุดของเซ็นเซอร์อื่น ๆ
latest_data = {
    "position_x": 0, "position_y": 0, "position_z": 0 }

# === Callback Functions ===
def position_handler(data):
    x, y, z = data
    latest_data.update({"position_x": x, "position_y": y, "position_z": z})
    log_data()

# เพิ่ม timestamp และบันทึกข้อมูลลง data_log
def log_data():
    timestamp = round(time.time() - start_time, 2)
    row = {"timestamp": timestamp}
    row.update(latest_data)
    data_log.append(row)
    print(f"[{timestamp}s] {latest_data}")

# === Main ===
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal


    # เริ่ม subscription สำหรับข้อมูลเซ็นเซอร์
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_chassis.sub_position(freq=10, callback=position_handler)

    # เคลื่อนที่เป็นสี่เหลี่ยม
    for _ in range(4):
        ep_chassis.move(x=0.6, y=0, z=0, xy_speed=0.6).wait_for_completed()
        ep_chassis.move(x=0, y=0, z=-90, z_speed=45).wait_for_completed()

    # ปิด subscription
    ep_chassis.unsub_position()


    # บันทึกลง CSV
    df = pd.DataFrame(data_log)
    df.to_csv("draw_squre1.csv", index=False)
    print("✅ Data saved to draw_squre1.csv")

    ep_robot.close()