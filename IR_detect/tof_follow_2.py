


import robomaster
from robomaster import robot
import time


def sub_data_handler(sub_info):
    global front_tof_cm
    try:
        front_tof_cm = float(sub_info[0]) / 10.0
    except:
        front_tof_cm = 9999.0
def adc_to_cm1(adc):   # เซนเซอร์ 1 (ซ้ายหน้า)
    return (892.5 - float(adc)) / 57.5

def adc_to_cm2(adc):   # เซนเซอร์ 2 (ซ้ายหลัง)
    return (917.5 - float(adc)) / 62.5
if __name__ == "__main__":
    
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    counter_wall = 0
    ep_chassis = ep.chassis
    ep_sensor  = ep.sensor
    ep_adaptor = ep.sensor_adaptor
    ep_gimbal  = ep.gimbal
    ep_gimbal.recenter().wait_for_complete
    ep_sensor.sub_distance(freq=5, callback=sub_data_handler)
    while True:
        ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        adc_front_left = ep_adaptor.get_adc(id=1, port=1)
        adc_back_left  = ep_adaptor.get_adc(id=1, port=2)
        adc_front_left_cm = adc_to_cm1(adc_front_left)
        adc_back_left_cm  = adc_to_cm2(adc_back_left)
        if front_tof_cm < 10 and counter_wall == 0:
            ep_chassis.move(x=0, y=0, z=90, xy_speed=5.7).wait_for_completed()
            ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
            counter_wall += 1
            print("Turn Right 90 degree")
        if front_tof_cm >= 10 and counter_wall == 1:
            ep_chassis.move(x=0, y=0, z=30, xy_speed=5.7).wait_for_completed()
            ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
            counter_wall += 1
            print("Turn Left 30 degree")
        if front_tof_cm >= 10 and counter_wall == 2:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=2)
            break
        error_z = adc_front_left_cm - adc_back_left_cm
        error_y = 5 - (adc_front_left_cm + adc_back_left_cm)/2
        if error_z > -1 and error_z < 1:
            error_z = 0
        if error_y > -1 and error_y < 1:
            error_y = 0
        
        kp = 5
        ep_chassis.drive_speed(x=1, y=error_y, z=kp * error_z, timeout=1)
