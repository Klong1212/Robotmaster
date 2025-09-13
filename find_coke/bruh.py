# หากระป๋อง ขนาดจริง กว้าง 5.2 cm สูง 13.4
# ได้ K_xavg = 780.77 (คาดเคลื่อนปานกลาง)
# ได้ K_yavg = 756.72 (คาดเคลื่อนเล็กน้อย)

import robomaster
from robomaster import robot
import pandas as pd
import time
import cv2
import numpy as np

# ---------- ค่าคงที่ของวัตถุและการคาลิเบรต ----------
REAL_W_CM = 5.2    # ความกว้างจริงของกระป๋อง (cm)
REAL_H_CM = 13.4   # ความสูงจริงของกระป๋อง (cm)
KX = 623.08        # ค่าคาลิเบรตแกน X (pixel * cm / pixel_size) ที่คุณคำนวณไว้
KY = 610.45        # ค่าคาลิเบรตแกน Y

# ฟังก์ชันทั้งหมด ดึงมาจาก Quadbotz_test4_copy.ipynb
# ฟังก์ชันสำหรับ event ของเมาส์
def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        pixel = out_img[y, x]
        print(f"ตำแหน่ง: ({x}, {y}) ค่า BGR = {pixel}")

def _clip(a, lo, hi):
    return max(lo, min(int(a), hi))

def ring_blue_ratio_at(blue_f, x, y, w_tpl, h_tpl, pad_frac=0.15):
    H, W = blue_f.shape[:2]
    """คำนวณสัดส่วนสีน้ำเงินใน 'แถบรอบกรอบ' (ring) ที่ได้จากการขยายกรอบออก
    ถ้าแถบรอบมีสีน้ำเงินเยอะ => มีสีน้ำเงินล้นนอกกรอบ => เทมเพลตเล็กไป"""
    pad = int(round(pad_frac * max(w_tpl, h_tpl)))
    # กรอบใน (inner) = กรอบเทมเพลต, กรอบนอก (outer) = ขยายออกด้วย pad
    x1_in, y1_in = x, y
    x2_in, y2_in = x + w_tpl, y + h_tpl

    x1_out = _clip(x1_in - pad, 0, W - 1)
    y1_out = _clip(y1_in - pad, 0, H - 1)
    x2_out = _clip(x2_in + pad, 1, W)
    y2_out = _clip(y2_in + pad, 1, H)

    # พื้นที่วงแหวน = พื้นที่กรอบนอก - พื้นที่กรอบใน
    area_out = max(0, x2_out - x1_out) * max(0, y2_out - y1_out)
    area_in  = max(0, x2_in - x1_in)   * max(0, y2_in - y1_in)
    area_ring = max(1, area_out - area_in)  # กันหารศูนย์

    # รวมสีน้ำเงินในกรอบนอกและใน
    sum_out = float(np.sum(blue_f[y1_out:y2_out, x1_out:x2_out]))
    sum_in  = float(np.sum(blue_f[y1_in:y2_in,   x1_in:x2_in]))
    sum_ring = max(0.0, sum_out - sum_in)

    return sum_ring / float(area_ring)

def detetct_can(image):
    img_bgr = image
    tpl_bgr = cv2.imread(r"find_coke\box_1.jpg")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = tpl_bgr.shape[:2]
    d0 = 0.6
    distances1 = 1.2
    distances2 = 1.8

    scale1 = d0 / distances1
    new_w1 = int(round(w0 * scale1))
    new_h1 = int(round(h0 * scale1))
    tpl_resized_12 = cv2.resize(tpl_bgr, (new_w1, new_h1), interpolation=cv2.INTER_AREA)

    scale2 = d0 / distances2
    new_w2 = int(round(w0 * scale2))
    new_h2 = int(round(h0 * scale2))
    tpl_resized_18 = cv2.resize(tpl_bgr, (new_w2, new_h2), interpolation=cv2.INTER_AREA)

    tpl_resized_12_gray = cv2.cvtColor(tpl_resized_12, cv2.COLOR_BGR2GRAY)
    tpl_resized_18_gray = cv2.cvtColor(tpl_resized_18, cv2.COLOR_BGR2GRAY)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([140, 255, 255])
    median_ksize = 5

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.medianBlur(blue_mask, median_ksize)

    binary_img = blue_mask.copy()

    THRESH_WHITE = 0.5
    THRESH_BLACK = 0.5
    THRESH_COMBINED = 0.5  # ใช้ช่วยกรอง/ผูกคะแนน
    BETA = 6.0             # ความแรงของโทษต่อ ring_blue_ratio (ลอง 2–6 ได้)

    templates_gray = [
        ("base_0.6m(or original)", tpl_gray),
        ("scaled_1.2m",            tpl_resized_12_gray),
        ("scaled_1.8m",            tpl_resized_18_gray),
    ]

    blue_f     = (binary_img // 255).astype(np.float32)  # 0/1
    not_blue_f = 1.0 - blue_f

    best = {
        "name": None,
        "metric": -1.0,   # metric = min(W,B) * scale_penalty
        "score_w": None,
        "score_b": None,
        "score_c": None,
        "pos": None,      # (x, y)
        "tpl_hw": None,   # (h, w)
        "passed": False,
    }

    for name, tpl_g in templates_gray:
        # ทำ binary mask ของเทมเพลต: ขาว=1, ดำ=0 (float32)
        _, tpl_bin = cv2.threshold(tpl_g, 127, 255, cv2.THRESH_BINARY)
        tpl_white = (tpl_bin // 255).astype(np.float32)
        tpl_black = 1.0 - tpl_white

        # คำนวณคะแนน
        score_white = cv2.matchTemplate(blue_f,     tpl_white, method=cv2.TM_CCORR_NORMED)
        score_black = cv2.matchTemplate(not_blue_f, tpl_black, method=cv2.TM_CCORR_NORMED)
        combined_score = score_white * score_black

        # ใช้ตำแหน่งที่ combined สูงสุดของเทมเพลตนี้เป็นตัวแทน (อ่านค่า W/B ที่จุดเดียวกัน)
        _, max_c, _, max_loc_c = cv2.minMaxLoc(combined_score)
        x, y = max_loc_c
        val_w = float(score_white[y, x])
        val_b = float(score_black[y, x])
        val_c = float(combined_score[y, x])

        h_tpl, w_tpl = tpl_white.shape[:2]

        # --- โทษจากน้ำเงินล้นรอบกรอบ (กรอบเล็กไป) ---
        r_ratio = ring_blue_ratio_at(blue_f, x, y, w_tpl, h_tpl, pad_frac=0.15)  # 0..1
        scale_penalty = float(np.exp(-BETA * r_ratio))  # 1 เมื่อล้นน้อย, -> 0 เมื่อล้นมาก

        base_metric = min(val_w, val_b)
        metric = base_metric * scale_penalty

        passed_here = (val_w >= THRESH_WHITE and
                       val_b >= THRESH_BLACK and
                       val_c >= THRESH_COMBINED)

        # เลือก best ด้วย metric เป็นหลัก; ถ้าเท่ากัน ใช้ combined เป็น tie-break
        better = (metric > best["metric"]) or (np.isclose(metric, best["metric"]) and val_c > (best["score_c"] if best["score_c"] is not None else -1))
        if better:
            best.update({
                "name": name,
                "metric": metric,
                "score_w": val_w,
                "score_b": val_b,
                "score_c": val_c,
                "pos": (x, y),
                "tpl_hw": (h_tpl, w_tpl),
                "passed": bool(passed_here),
            })

    out_rgb = img_rgb.copy()
    Zx = None
    Zy = None
    Zavg = None

    if best["pos"] is not None:
        bx, by = best["pos"]
        h_tpl, w_tpl = best["tpl_hw"]
        top_left = (bx, by)
        bottom_right = (bx + w_tpl, by + h_tpl)

        # วาดกรอบถ้าผ่านเกณฑ์
        if best["passed"]:
            cv2.rectangle(out_rgb, top_left, bottom_right, (255, 0, 0), 3)

            # ---- คำนวณขนาดพิกเซลของวัตถุจากพิกเซลสีน้ำเงินภายใน bbox ----
            # ใช้ blue_mask (0/255) ภายใน bbox
            H, W = blue_mask.shape[:2]
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(W, bx + w_tpl)
            y2 = min(H, by + h_tpl)

            roi = blue_mask[y1:y2, x1:x2]  # พื้นที่สีน้ำเงินเฉพาะใน bbox
            nz = cv2.findNonZero(roi)      # คืนจุดที่เป็น non-zero (สีน้ำเงิน)
            if nz is not None and len(nz) >= 2:
                # หา min/max ของคอลัมน์และแถวภายใน roi
                xs = nz[:, 0, 0]
                ys = nz[:, 0, 1]
                min_x, max_x = int(xs.min()), int(xs.max())
                min_y, max_y = int(ys.min()), int(ys.max())

                # ความกว้าง/สูงของ "วัตถุจริง" ในหน่วยพิกเซล (จากสีน้ำเงินจริง ๆ)
                obj_w_px = max(1, (max_x - min_x + 1))
                obj_h_px = max(1, (max_y - min_y + 1))

                # ---- คำนวณระยะจากแกน X และ Y ----
                Zx = KX * (REAL_W_CM / float(obj_w_px))
                Zy = KY * (REAL_H_CM / float(obj_h_px))
                Zavg = 0.5 * (Zx + Zy)

                # วาดกรอบย่อยเฉพาะบริเวณน้ำเงินจริงใน bbox เพื่อให้เห็นว่าที่เราใช้วัดคือส่วนนี้
                sub_top_left  = (x1 + min_x, y1 + min_y)
                sub_bottom_right = (x1 + max_x, y1 + max_y)
                cv2.rectangle(out_rgb, sub_top_left, sub_bottom_right, (0, 255, 0), 2)

                # แสดงตัวเลขบนภาพ
                base_y = max(20, y1 - 10)
                txt1 = f"Zx = {Zx:.2f} cm  (Wpx={obj_w_px})"
                txt2 = f"Zy = {Zy:.2f} cm  (Hpx={obj_h_px})"
                txt3 = f"Z  = {Zavg:.2f} cm (avg)"
                cv2.putText(out_rgb, txt1, (x1, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)
                cv2.putText(out_rgb, txt2, (x1, base_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)
                cv2.putText(out_rgb, txt3, (x1, base_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # ไม่มีสีน้ำเงินพอที่จะวัด
                cv2.putText(out_rgb, "No blue pixels in bbox to measure.", (top_left[0], max(20, top_left[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        # ---- วาดเส้นกากบาทกลางภาพ (บน out_rgb ซึ่งเป็น RGB) ----
    h, w = out_rgb.shape[:2]
    center_x, center_y = w // 2, h // 2
    cv2.line(out_rgb, (0, center_y), (w, center_y), (0, 255, 0), 1)   # เขียวแนวนอน (RGB)
    cv2.line(out_rgb, (center_x, 0), (center_x, h), (255, 0, 0), 1)   # แดงแนวตั้ง (RGB)
    return out_rgb, Zavg, Zx, Zy

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal

    ep_camera.start_video_stream(display=False, resolution="720p") # เปิดกล้อง
    ep_gimbal.moveto(yaw = 0, pitch = 0).wait_for_completed()


    try:
        while True:
            # วาดเส้นแกน x (แนวนอน) และ y (แนวตั้ง)
            img = ep_camera.read_cv2_image(strategy="newest") # อ่านภาพจากกล้อง
            out_img, Zavg, Zx, Zy = detetct_can(img)
            
            # แสดงภาพที่มี overlay แล้ว
            cv2.imshow("Can Detection + Distance", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            cv2.setMouseCallback("Can Detection + Distance", show_pixel_value)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()