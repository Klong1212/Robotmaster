# servo_demo.py
from robomaster import robot
import time

if __name__ == "__main__":
    ep = robot.Robot()
    ep.initialize(conn_type="ap")

    ep_servo = ep.servo
    try:
        # อ่านมุมเริ่มต้นของเซอร์โวหมายเลข 0
        angle_now = ep_servo.get_angle(index=1)
        print(f"Servo0 angle = {angle_now}°")

        # หมุนไปที่ +90°
        print("Go to +90° …")
        ep_servo.moveto(index=1, angle=20).wait_for_completed()
        time.sleep(1)

        # หมุนไปที่ -90°
        print("Go to -90° …")
        ep_servo.moveto(index=1, angle=-20).wait_for_completed()
        time.sleep(1)

        # ทดสอบโหมดความเร็ว
        print("Spin at ~00 °/s for 0s …")
        ep_servo.drive_speed(index=1, speed=20)
        time.sleep(2.0)
        ep_servo.pause(index=0)  # หยุดหมุน

        angle_after = ep_servo.get_angle(index=1)
        print(f"Servo0 angle now = {angle_after}°")

    finally:
        ep.close()
