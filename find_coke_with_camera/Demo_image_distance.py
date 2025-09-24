import robomaster
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os, tempfile, time

# ========= utils เล็กๆ =========
def _clip(v, lo, hi):
    return max(lo, min(int(v), hi))

def _tight_bbox_from_mask(mask_bool):
    """
    mask_bool: HxW (dtype=bool) ของบริเวณย่อย (window)
    return: (x_min, y_min, x_max, y_max) ในพิกัดของ window (inclusive x_min/y_min, exclusive x_max/y_max)
            ถ้าไม่พบพิกเซล True เลย ให้คืน None
    """
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    y_min, y_max = int(ys.min()), int(ys.max())+1
    x_min, x_max = int(xs.min()), int(xs.max())+1
    return x_min, y_min, x_max, y_max

def detect_red_object(img_read):
    # ----- เตรียมภาพ -----
    img_hsv = cv2.cvtColor(img_read, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # หมายเหตุ: ช่วงนี้คือ "เหลือง" (20-40) ตามที่คุณตั้งไว้
    img_binary = ((h > 20) & (h < 50) & (s > 100)&(v>100))

    # ทำเป็น uint8 เพื่อใช้กับ matchTemplate
    img_binary_uint8 = (img_binary.astype(np.uint8) * 255)

    # ----- โหลด template 3 อัน -----
    templat1 = cv2.imread(r"find_coke\box_1_test.jpg")
    templat2 = cv2.imread(r"find_coke\box_2_test.jpg")
    templat3 = cv2.imread(r"find_coke\box_3_test.jpg")

    if templat1 is None or templat2 is None or templat3 is None:
        raise FileNotFoundError("ไม่พบไฟล์ template บางไฟล์ในโฟลเดอร์ find_coke")

    # แปลงเป็น binary เช่นเดียวกับภาพหลัก (ใช้ช่อง V > 100)
    def _tpl_to_bin_u8(img_bgr):
        tpl_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        _, _, tpl_v = cv2.split(tpl_hsv)
        tpl_bin = (tpl_v > 100).astype(np.uint8) * 255
        return tpl_bin

    template_binary_uint8_1 = _tpl_to_bin_u8(templat1)
    template_binary_uint8_2 = _tpl_to_bin_u8(templat2)
    template_binary_uint8_3 = _tpl_to_bin_u8(templat3)

    # ตรวจสอบขนาด: template ต้องเล็กกว่าหรือเท่าภาพหลัก
    H, W = img_binary_uint8.shape[:2]
    for tname, T in [("box_1", template_binary_uint8_1),
                     ("box_2", template_binary_uint8_2),
                     ("box_3", template_binary_uint8_3)]:
        th, tw = T.shape[:2]
        if th > H or tw > W:
            raise ValueError(f"Template {tname} ใหญ่กว่าภาพอินพุต (th={th}, tw={tw}, H={H}, W={W})")

    # ----- matchTemplate กับทั้ง 3 อัน -----
    value1 = cv2.matchTemplate(img_binary_uint8, template_binary_uint8_1, cv2.TM_CCOEFF_NORMED)
    value2 = cv2.matchTemplate(img_binary_uint8, template_binary_uint8_2, cv2.TM_CCOEFF_NORMED)
    value3 = cv2.matchTemplate(img_binary_uint8, template_binary_uint8_3, cv2.TM_CCOEFF_NORMED)

    min_val1, max_val1, _, max_loc1 = cv2.minMaxLoc(value1)
    min_val2, max_val2, _, max_loc2 = cv2.minMaxLoc(value2)
    min_val3, max_val3, _, max_loc3 = cv2.minMaxLoc(value3)

    # ----- เลือก template ที่คะแนนสูงสุด -----
    best_idx = int(np.argmax([max_val1, max_val2, max_val3]))
    if best_idx == 0:
        top_left = max_loc1
        tpl = template_binary_uint8_1
        max_val = max_val1
    elif best_idx == 1:
        top_left = max_loc2
        tpl = template_binary_uint8_2
        max_val = max_val2
    else:
        top_left = max_loc3
        tpl = template_binary_uint8_3
        max_val = max_val3

    th, tw = tpl.shape[:2]

    # ----- ถ้าคะแนนดีพอ ค่อยทำ tight bbox ภายใน "หน้าต่าง" รอบตำแหน่งแมตช์ -----
    # เกณฑ์เดิม: > 0.4
    distance_avg = 0.0

    if max_val is not None and max_val > 0.4:
        x0, y0 = int(top_left[0]), int(top_left[1])
        x1, y1 = x0 + tw, y0 + th

        # กันขอบภาพ
        x0c = _clip(x0, 0, W - 1)
        y0c = _clip(y0, 0, H - 1)
        x1c = _clip(x1, 1, W)     # exclusive ขอบขวา ใช้ W ได้เลย
        y1c = _clip(y1, 1, H)     # exclusive ขอบล่าง ใช้ H ได้เลย

        # หน้าต่างค้นหาใน img_binary (bool)
        window_mask = img_binary[y0c:y1c, x0c:x1c]
        # หา bbox ที่ “แน่น” จาก True pixels ภายในหน้าต่าง
        tight = _tight_bbox_from_mask(window_mask)
        if tight is not None:
            tx0, ty0, tx1, ty1 = tight  # พิกัดภายใน window
            # แปลงกลับเป็นพิกัดเต็มภาพ
            top_left_tight = (x0c + tx0, y0c + ty0)
            bottom_right_tight = (x0c + tx1, y0c + ty1)

            # วาดกรอบ
            cv2.rectangle(img_read, top_left_tight, bottom_right_tight, (0, 255, 0), 2)

            size_x = bottom_right_tight[0] - top_left_tight[0]
            size_y = bottom_right_tight[1] - top_left_tight[1]
            print(size_x, size_y)

            if size_x > 0 and size_y > 0:
                # ใช้ค่าคงที่เดิม
                distance_x = 675.57  * 5.6 / size_x
                distance_y = 616.33 * 14.7 / size_y
                print(distance_y,distance_x)
                distance_avg = (distance_x + distance_y) / 2.0
            else:
                distance_avg = 0.0
        else:
            # ในหน้าต่างไม่มีพิกเซล True เลย (อาจกรองสีไม่ผ่าน)
            distance_avg = 0.0
            max_val = 0.0
    else:
        distance_avg = 0.0
        max_val = 0.0

    return img_read, max_val, distance_avg

# ================== main ==================
import cv2
import robomaster
from robomaster import robot
from robomaster import vision

robots = []

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_gimbal=ep_robot.gimbal
    ep_gimbal.moveto(yaw = 0, pitch = 0).wait_for_completed()
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution='720p')  # 1080p อาจช้า

    print("กด Q เพื่อออก")
    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest")
            if frame is None:
                continue

            frame_res, max_val, distance_avg = detect_red_object(frame)

            cv2.imshow("Live (with detection)", frame_res)
            print(f"Max val: {max_val:.2f}, Distance avg: {distance_avg:.2f} cm")

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
    except Exception as e:
        print("Error:", e)
    finally:
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()