from robomaster import robot
import keyboard
thread_hold = 80
front_speed=0.1

def adc_to_cm(adc):
    return -0.2326 * adc + 121.5

def sub_data_handler(sub_info):
    global distance_tof
    distance=sub_info
    distance_tof = sub_info[0]
    print("tof1:{0}  tof2:{1}  tof3:{2}  tof4:{3}".format(distance[0], distance[1], distance[2], distance[3]))

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_sensor_adaptor = ep_robot.sensor_adaptor
    ep_chassis = ep_robot.chassis
    ep_chassis.drive_speed(x=0.1, y=0, z=0)
    while True:
        adc = ep_sensor_adaptor.get_adc(id=1, port=1)
        adc_cm = adc_to_cm(adc)
        if keyboard.is_pressed('q'):
            break
        elif adc_cm < 5 and distance_tof < thread_hold:
            ep_chassis.drive_speed(x=0, y=0, z=5)
        elif adc_cm < 5 and distance_tof > thread_hold:
            ep_chassis.drive_speed(x=front_speed, y=0, z=-2)
        elif adc_cm > 5 and distance_tof > thread_hold:
            ep_chassis.drive_speed(x=front_speed, y=0, z=2)
        else:
            ep_chassis.drive_speed(x=front_speed, y=0, z=0)
        


    ep_robot.close()