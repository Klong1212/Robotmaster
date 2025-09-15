from robomaster import robot
import cv2
import time

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution='720p')

    print("กด Q เพื่อออก")
    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=1)
            if frame is not None:
                cv2.imshow("Camera Stream", frame)
            else:
                print("No frame received")
                time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
    finally:
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()