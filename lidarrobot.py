from robomaster import robot

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_gimbal = ep_robot.gimbal

    ep_gimbal.moveto(pitch=0, yaw=-90, pitch_speed=0, yaw_speed=50).wait_for_completed()
    ep_gimbal.moveto(pitch=0, yaw=180, pitch_speed=0, yaw_speed=50).wait_for_completed()
    

    ep_robot.close()
