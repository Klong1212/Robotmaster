from robomaster import robot
import keyboard
thread_hold = 200
def adc_to_cm(adc):
    return -0.2326 * adc + 121.5
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
        elif adc_cm < 10:
            ep_chassis.drive_speed(x=0.1, y=0, z=-5)
        elif adc_cm > 15:
            ep_chassis.drive_speed(x=0.1, y=0, z=5)
        else:
            ep_chassis.drive_speed(x=0.1, y=0, z=0)
        


    ep_robot.close()