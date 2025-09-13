import robomaster
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, tempfile, time

def detect_red_object(img_read):
    img_hsv = cv2.cvtColor(img_read, cv2.COLOR_BGR2HSV) # แปลงเป็น HSV
    h,s,v=cv2.split(img_hsv) # แยก HSV
    img_binary = ((h > 30) & (h < 40) & (s > 170))# ทำเป็น binary

    templat1=cv2.imread(r"find_coke\box_1.jpg") # โหลดภาพ
    template_hsv=cv2.cvtColor(templat1,cv2.COLOR_BGR2HSV) # แปลงเป็น HSV
    template_h,template_s,template_v_1=cv2.split(template_hsv) # แยก HSV
    template_binary1 = (template_v_1 > 100) # สร้าง mask

    templat2=cv2.imread(r"find_coke\box_2.jpg") # โหลดภาพ
    template_hsv=cv2.cvtColor(templat2,cv2.COLOR_BGR2HSV) # แปลงเป็น HSV
    template_h,template_s,template_v_2=cv2.split(template_hsv) # แยก HSV
    template_binary2 = (template_v_2 > 100) # สร้าง mask

    templat3=cv2.imread(r"find_coke\box_3.jpg") # โหลดภาพ
    template_hsv=cv2.cvtColor(templat3,cv2.COLOR_BGR2HSV) # แปลงเป็น HSV
    template_h,template_s,template_v_3=cv2.split(template_hsv) # แยก HSV
    template_binary3 = (template_v_3 > 100) # สร้าง mask

    # เปลี่ยนเป็น uint8
    img_binary_uint8 = img_binary.astype(np.uint8) * 255
    template_binary_uint8_1 = template_binary1.astype(np.uint8) * 255
    template_binary_uint8_2 = template_binary2.astype(np.uint8) * 255
    template_binary_uint8_3 = template_binary3.astype(np.uint8) * 255

    # จับคู่ภาพกับ template
    value = cv2.matchTemplate(img_binary_uint8, template_binary_uint8_1, cv2.TM_CCOEFF_NORMED)
    value2 = cv2.matchTemplate(img_binary_uint8, template_binary_uint8_2, cv2.TM_CCOEFF_NORMED)
    value3 = cv2.matchTemplate(img_binary_uint8, template_binary_uint8_3, cv2.TM_CCOEFF_NORMED)
    
    # หาค่าสูงสุด
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(value)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(value2)
    min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(value3)

    # เลือก template ที่ตรงกันมากที่สุด
    if max(max_val, max_val2, max_val3) == max_val:
        top_left = max_loc
        template_h, template_w = template_binary_uint8_1.shape
        max_val = max_val
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        template_binary_uint8=template_binary_uint8_1
    elif max(max_val, max_val2, max_val3) == max_val2:
        top_left = max_loc2
        template_h, template_w = template_binary_uint8_2.shape
        max_val = max_val2
        template_binary_uint8=template_binary_uint8_2
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
    else:
        top_left = max_loc3
        template_h, template_w = template_binary_uint8_3.shape
        max_val = max_val3
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        template_binary_uint8=template_binary_uint8_3
    
    # for y in range(template_h):
    #     if np.any(template_binary_uint8[y, :] == 255):
    #         top = y
    #         continue
    # # หาขอบล่าง
    # for y in range(template_h-1, -1, -1):
    #     if np.any(template_binary_uint8[y, :] == 255):
    #         bottom = y
    #         continue
    # # หาขอบซ้าย
    # for x in range(template_w):
    #     if np.any(template_binary_uint8[:, x] == 255):
    #         left = x
    #         continue
    # # หาขอบขวา
    # for x in range(template_w-1, -1, -1):
    #     if np.any(template_binary_uint8[:, x] == 255):
    #         right = x
    #         continue

    # top_left = (max_loc[0] + left,max_loc[1] + top)
    # bottom_right = (max_loc[0] + right, max_loc[1] + bottom)
    # print(top,bottom,left,right)
    
    if max_val > 0.4 and max_val is not None:
        cv2.rectangle(img_read, top_left, bottom_right, (0, 255, 0), 2) # วาดกรอบ (สีเขียว)
        size_y = bottom_right[1] - top_left[1]
        size_x = bottom_right[0] - top_left[0]
        # ป้องกันหารศูนย์
        if size_x > 0 and size_y > 0:
            distance_x = 671.43 * 5.6 / size_x
            distance_y = 671.43 * 4.2 / size_y
            distance_avg = (distance_x + distance_y) / 2
        else:
            distance_avg = 0
    else:
        # ไม่เจอวัตถุ: ไม่วาดกรอบ, ระยะเป็น 0
        distance_avg = 0
        max_val = 0
# ...existing code...
    return img_read,max_val, distance_avg

# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import robomaster
from robomaster import robot
from robomaster import vision


robots = []

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution='720p')  # 1080p ช้าได้

    print("กด Q เพื่อออก")
    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest")
            # เรียกฟังก์ชัน detect_red_object และรับภาพที่วาดกรอบแล้ว
            frame_res,max_val, distance_avg = detect_red_object(frame)

            # แสดงผลภาพที่วาดกรอบแล้ว
            cv2.imshow("Live (with detection)", frame_res)
            print(f"Max val: {max_val}, Distance avg: {distance_avg}")

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
    except Exception as e:
        print(e)
    finally:
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()