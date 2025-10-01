from robomaster import robot
import msvcrt
import time

stop_flag = False  

def sub_data_handler(sub_info):
    distance = sub_info
    # print("tof1:{0}".format(distance[0]))
    # time.sleep(0.5)
    if distance[0] < 250:
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=5)
        stop_flag = True

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor
    ep_sensor_adaptor = ep_robot.sensor_adaptor

    ep_gimbal.moveto(pitch=-15, yaw=0).wait_for_completed()
    time.sleep(0.1)
    
    tof = ep_sensor.sub_distance(freq=20, callback=sub_data_handler)
    # time.sleep(0.2)
    
    while True:
        if stop_flag:
            break
        io = ep_sensor_adaptor.get_adc(id=1, port=2)
        # print('io',io)
        ep_chassis.drive_speed(x=0.2, y=0, z=0, timeout=5)
        if io > 400:
            ep_chassis.drive_speed(x=0, y=0.05, z=0, timeout=5)
        if io < 300:
            ep_chassis.drive_speed(x=0, y=-0.05, z=0, timeout=5)
        time.sleep(0.5)

        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                break

    ep_sensor.unsub_distance()
    ep_robot.close()