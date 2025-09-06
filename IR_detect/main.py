from robomaster import robot

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_sensor_adaptor = ep_robot.sensor_adaptor
    ep_chassis = ep_robot.chassis
    ep_chassis.drive_speed(x=0, y=0.1, z=0)
    while True:
        adc = ep_sensor_adaptor.get_io(id=1, port=1)
        if adc == 0:
            ep_chassis.drive_speed(x=0, y=0, z=0)
            break


    ep_robot.close()