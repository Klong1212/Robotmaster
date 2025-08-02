import robomaster
from robomaster import robot
import pandas as pd
import time

data_log = [] # รายการสำหรับเก็บข้อมูลที่เก็บได้จากเซ็นเซอร์ TOF และ Gimbal

latest_data = {"tof_distance": 0, "yaw_angle": 0} # ตัวแปรสำหรับเก็บข้อมูลล่าสุดจากเซ็นเซอร์ TOF และ Gimbal


def tof_handler(data): # Callback function เพื่อจัดการข้อมูลจากเซ็นเซอร์ TOF
    distance = data
    latest_data.update({"tof_distance": distance[0]+100}) # เพิ่มค่า 100 mm เพื่อปรับค่าระยะทางเพิ่มเติมจากเซ็นเซอร์ โดยเลือกเก็บแค่ TOF1
    data_log.append(latest_data.copy())


def yaw_handler(data): # Callback function เพื่อจัดการข้อมูลจากเซ็นเซอร์ Gimbal
    pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = data
    latest_data.update({"yaw_angle": yaw_angle}) # เก็บมุม Yaw ของ Gimbal
    data_log.append(latest_data.copy())


if __name__ == "__main__": 
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_gimbal = ep_robot.gimbal # Gimbal ของหุ่นยนต์

    ep_sensor = ep_robot.sensor # เซ็นเซอร์ TOF ของหุ่นยนต์


    # ตั้งค่า Gimbal ให้เริ่มต้นที่มุม 0 องศา
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()

    # ตั้งค่า Gimbal ให้หมุนไปที่มุม 90 องศา เพื่อเริ่มต้นการเก็บข้อมูล
    ep_gimbal.moveto(pitch=0, yaw=180, pitch_speed=0, yaw_speed=100).wait_for_completed()

    # เริ่มการเก็บข้อมูลจากเซ็นเซอร์ TOF และ Gimbal
    ep_gimbal.sub_angle(freq=20, callback=yaw_handler)
    ep_sensor.sub_distance(freq=20, callback=tof_handler)

    # ให้ Gimbal หมุนไปที่มุม -90 องศา เพื่อเก็บข้อมูล
    ep_gimbal.moveto(pitch=0, yaw=-180, pitch_speed=0, yaw_speed=10).wait_for_completed()

    # เลิกการเก็บข้อมูลจากเซ็นเซอร์ TOF และ Gimbal
    ep_gimbal.unsub_angle()
    ep_sensor.unsub_distance()

    # ให้ Gimbal กลับไปที่มุม 0 องศาตามเดิม
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()

    # บันทึกข้อมูลที่เก็บได้ลงในไฟล์ robot_log.csv
    df = pd.DataFrame(data_log)
    df.to_csv("draw_squre2.csv", index=False)
    print("✅ Data saved to draw_squre2.csv")

    ep_robot.close()