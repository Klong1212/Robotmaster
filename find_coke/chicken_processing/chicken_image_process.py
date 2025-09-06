import robomaster
import cv2
import numpy as np
import time
from robomaster import robot

# ======= กล้อง/กิมบอล & PID พารามิเตอร์ =======
CAMERA_W = 1280
CAMERA_H = 720
CAMERA_HFOV_DEG = 96.0           # มุมมองแนวนอนของกล้อง (ปรับตามรุ่นได้)
GIMBAL_TURN_SPEED = 360          # deg/s สำหรับคำสั่ง move
TEMPLATE_MATCH_THRESHOLD = 0.45   # ค่าความมั่นใจขั้นต่ำของการจับคู่เทมเพลต

# PID สำหรับ yaw (หน่วย "องศา" ของ error)
class PID:
    def __init__(self, kp=0.8, ki=0.0, kd=0.22, integ_clip=30.0, out_clip=12.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integ = 0.0
        self.prev_err = 0.0
        self.integ_clip = integ_clip     # จำกัดผลรวมสะสม
        self.out_clip = out_clip         # จำกัดคำสั่งออก (deg/step)

    def reset(self):
        self.integ = 0.0
        self.prev_err = 0.0

    def step(self, err, dt):
        # ป้องกัน dt = 0
        dt = max(dt, 1e-3)
        self.integ += err * dt
        # กันอินทิกรัลสะสมเกิน
        self.integ = max(-self.integ_clip, min(self.integ, self.integ_clip))
        deriv = (err - self.prev_err) / dt
        self.prev_err = err
        out = self.kp * err + self.ki * self.integ + self.kd * deriv
        # จำกัด magnitude ต่อเฟรม (หน่วย deg)
        out = max(-self.out_clip, min(self.out_clip, out))
        return out

# ======= ฟังก์ชันหา/กรอบวัตถุสีแดง + center =======
def detect_red_object(img_read):
    img_hsv = cv2.cvtColor(img_read, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # โทนสีแดงสองช่วงใน HSV (รวมเป็น mask เดียว)
    img_binary = ((h > 0) & (h < 10)) | ((h > 140) & (h < 180))
    img_binary_uint8 = img_binary.astype(np.uint8) * 255

    # โหลดเทมเพลต (สว่างพอเป็นวัตถุ)
    t1 = cv2.imread(r"find_coke\chicken_processing\chicken_template_1.jpg", cv2.IMREAD_COLOR)
    t2 = cv2.imread(r"find_coke\chicken_processing\chicken_template_2.jpg", cv2.IMREAD_COLOR)
    t3 = cv2.imread(r"find_coke\chicken_processing\chicken_template_3.jpg", cv2.IMREAD_COLOR)

    # ถ้าไม่มีไฟล์เทมเพลต ให้ข้ามด้วยการคืน None
    if t1 is None or t2 is None or t3 is None:
        return img_read, None, 0.0

    def to_bin(img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        return (v > 100).astype(np.uint8) * 255

    tb1 = to_bin(t1)
    tb2 = to_bin(t2)
    tb3 = to_bin(t3)

    # matchTemplate แบบ normalized
    v1 = cv2.matchTemplate(img_binary_uint8, tb1, cv2.TM_CCOEFF_NORMED)
    v2 = cv2.matchTemplate(img_binary_uint8, tb2, cv2.TM_CCOEFF_NORMED)
    v3 = cv2.matchTemplate(img_binary_uint8, tb3, cv2.TM_CCOEFF_NORMED)

    # หาค่าสูงสุดแต่ละตัว
    _, m1, _, loc1 = cv2.minMaxLoc(v1)
    _, m2, _, loc2 = cv2.minMaxLoc(v2)
    _, m3, _, loc3 = cv2.minMaxLoc(v3)

    # เลือกตัวที่มั่นใจสุด
    max_val = m1
    top_left = loc1
    th, tw = tb1.shape[:2]
    chosen = 1

    if m2 > max_val:
        max_val = m2
        top_left = loc2
        th, tw = tb2.shape[:2]
        chosen = 2
    if m3 > max_val:
        max_val = m3
        top_left = loc3
        th, tw = tb3.shape[:2]
        chosen = 3

    if max_val < TEMPLATE_MATCH_THRESHOLD:
        # มั่นใจไม่พอ: ไม่วาดกรอบ/ไม่ให้ศูนย์กลาง
        return img_read, None, float(max_val)

    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(img_read, top_left, bottom_right, (0, 255, 0), 2)

    # center ของกรอบ (พิกเซลภาพ)
    cx = top_left[0] + tw // 2
    cy = top_left[1] + th // 2
    cv2.circle(img_read, (cx, cy), 4, (0, 255, 255), -1)
    # จุดกึ่งกลางภาพ
    cv2.circle(img_read, (CAMERA_W // 2, CAMERA_H // 2), 4, (255, 0, 0), -1)

    return img_read, (cx, cy), float(max_val)

# ======= main: สตรีม + PID คุมกิมบอลให้ศูนย์กลางตรงวัตถุ =======
if __name__ == "__main__":
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal

    # เซ็นเตอร์กิมบอลก่อนเริ่ม
    try:
        ep_gimbal.recenter().wait_for_completed()
    except Exception:
        pass

    ep_camera.start_video_stream(display=False, resolution='720p')
    print("เริ่มสตรีมแล้ว — กด Q เพื่อออก")

    pid_yaw = PID(kp=0.9, ki=0.0, kd=0.25, integ_clip=40.0, out_clip=10.0)
    last_t = time.monotonic()

    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            now = time.monotonic()
            dt = now - last_t
            last_t = now

            if frame is None:
                time.sleep(0.01)
                continue

            # ตรวจวัตถุแดง + center
            vis, center, conf = detect_red_object(frame)

            # คำนวณ error → มุม (deg) เฉพาะแกน yaw จากตำแหน่งแนวนอน
            # ถ้าไม่มีเป้า: ค่อย ๆ ปล่อยอินทิกรัล และไม่ส่งคำสั่ง
            if center is None:
                pid_yaw.integ *= 0.9
            else:
                cx, cy = center
                # error ในหน่วย "องศา": ซ้าย = บวก, ขวา = ลบ (สมมติ +yaw หมุนซ้าย)
                # สัดส่วนพิกเซล → องศา: (cx - Cx) / half_width * (HFOV/2)
                half_w = CAMERA_W / 2.0
                px_offset = (cx - half_w)
                err_norm = px_offset / half_w                          # [-1..1]
                err_deg = err_norm * (CAMERA_HFOV_DEG / 2.0)           # องศาที่ต้องชดเชย

                # อัปเดต PID ได้ "คำสั่ง yaw (deg) ต่อเฟรม"
                delta_yaw_deg = pid_yaw.step(err_deg, dt)

                # ส่งคำสั่งกิมบอล (relative move ทีละนิด ปลอดภัย และลื่น)
                if abs(delta_yaw_deg) > 0.05:
                    try:
                        ep_gimbal.move(yaw=delta_yaw_deg, pitch=0.0,
                                       yaw_speed=GIMBAL_TURN_SPEED, pitch_speed=0).wait_for_completed()
                    except Exception:
                        # ถ้า wait_for_completed มีปัญหา (เฟรมถี่มาก) ก็ส่งแบบไม่รอผลได้
                        try:
                            ep_gimbal.move(yaw=delta_yaw_deg, pitch=0.0,
                                           yaw_speed=GIMBAL_TURN_SPEED, pitch_speed=0)
                        except Exception:
                            pass

                # วาดเส้นชี้กึ่งกลางภาพช่วยดีบัก
                cv2.line(vis, (CAMERA_W//2, 0), (CAMERA_W//2, CAMERA_H), (255, 0, 0), 1)
                cv2.putText(vis, f"yaw_err(deg): {err_deg:+.2f} / pid_out(deg): {delta_yaw_deg:+.2f} / conf: {conf:.2f}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("Live (PID tracking red object)", vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

    finally:
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        try:
            ep_robot.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
