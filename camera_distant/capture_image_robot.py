import robomaster
from robomaster import robot, camera
import time
import cv2

ep_robot = robot.Robot()
ep_robot.initialize(conn_type="ap")
ep_gimbal = ep_robot.gimbal
ep_gimbal.recenter()
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=False, resolution="720p")

picture = ep_camera.read_cv2_image(strategy="newest", timeout=1)
if picture is not None:
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    img_binary = ((h > 30) & (h < 40) & (s > 100))
    img_binary = img_binary.astype("uint8") * 255
    cv2.imwrite("sponser_z4.jpg", img_binary)
else:
    print("ไม่พบภาพจากกล้อง")

ep_camera.stop_video_stream()
ep_robot.close()