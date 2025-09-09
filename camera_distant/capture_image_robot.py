import robomaster
from robomaster import robot, camera
import time
import cv2
import keyboard
ep_robot=robot.Robot()
ep_robot.initialize(conn_type="ap")
ep_gimbal=ep_robot.gimbal
ep_gimbal.recenter()
ep_camera=ep_robot.camera
ep_camera.start_video_stream(display=False,resolution="720p")
picture=ep_camera.read_cv2_image(strategy="newest", timeout=2)

cv2.imwrite("sponser_z3.jpg", picture)
