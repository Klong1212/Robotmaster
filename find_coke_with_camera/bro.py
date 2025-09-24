# -*- coding: utf-8 -*-
import os, time, math, traceback
import cv2
import numpy as np
import robomaster
from robomaster import robot

# ==========================
# Config ปรับได้ตามงานจริง
# ==========================
TEMPLATE_PATHS = [
    r"find_coke\box_1.jpg",
    r"find_coke\box_2.jpg",
    r"find_coke\box_3.jpg",
]

# HSV สำหรับ "สีเหลือง" (เดิมโค้ดใช้ช่วงนี้อยู่แล้ว)
HSV_H_LO, HSV_H_HI = 30, 40
HSV_S_MIN = 100
HSV_V_MIN_FOR_TEMPLATE = 100

# เกณฑ์ความเชื่อมั่นของ template matching (TM_CCOEFF_NORMED)
ACQUIRE_THR = 0.50  # ต้องมากพอเวลา "เริ่มเจอ" เป้าใหม่
KEEP_THR    = 0.40  # ผ่อนปรนกว่าเวลา "เกาะติด" เป้าเดิม

# State Machine params
MAX_LOST_FRAMES = 12   # หลุด detection ต่อเนื่องกี่เฟรมจึงถือว่า LOST -> SEARCHING
EMA_ALPHA = 0.25       # smoothing ระยะทาง

# ค่าคงที่คำนวณระยะทาง (จากสูตรเดิม)
K_PIXEL = 671.43
OBJ_W_CM = 5.6
OBJ_H_CM = 4.2

# วาด UI
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ==========================
# Utilities
# ==========================
def preprocess_template(path):
    img = cv2.imread(path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # สร้าง binary จาก V > threshold (ตามโค้ดเดิม)
    binary = (v > HSV_V_MIN_FOR_TEMPLATE).astype(np.uint8) * 255
    # ทำ morphology ลด noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    return binary


def load_templates():
    tpls = []
    for p in TEMPLATE_PATHS:
        tpl = preprocess_template(p)
        if tpl is not None and tpl.ndim == 2 and tpl.size > 0:
            tpls.append(tpl)
        else:
            print(f"[WARN] โหลด template ไม่ได้หรือว่าง: {p}")
    if not tpls:
        raise RuntimeError("ไม่พบ template ใช้งานได้เลยสักอัน ตรวจเส้นทางไฟล์ใน TEMPLATE_PATHS")
    return tpls


def make_yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h > HSV_H_LO) & (h < HSV_H_HI) & (s > HSV_S_MIN)).astype(np.uint8) * 255
    # ลบจุดรบกวนเล็ก ๆ
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def pick_best_match(mask, templates):
    """
    คืนค่า: (best_top_left, best_br, best_score) หรือ (None, None, 0.0) ถ้าหาไม่ได้
    """
    H, W = mask.shape[:2]
    best = (None, None, 0.0)
    for tpl in templates:
        th, tw = tpl.shape[:2]
        if H < th or W < tw:
            # template ใหญ่กว่าเฟรมปัจจุบัน ข้าม
            continue
        res = cv2.matchTemplate(mask, tpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best[2]:
            top_left = max_loc
            br = (top_left[0] + tw, top_left[1] + th)
            best = (top_left, br, float(max_val))
    return best


def compute_distance(px_w, px_h, ema_dist):
    """
    ระยะโดยใช้ทั้งแกน X/Y แล้วทำ EMA smoothing
    """
    if px_w <= 0 or px_h <= 0:
        return None, ema_dist

    dist_x = K_PIXEL * OBJ_W_CM / float(px_w)
    dist_y = K_PIXEL * OBJ_H_CM / float(px_h)
    dist = 0.5 * (dist_x + dist_y)

    if ema_dist is None:
        ema = dist
    else:
        ema = EMA_ALPHA * dist + (1.0 - EMA_ALPHA) * ema_dist
    return dist, ema


# ==========================
# Detector (State Machine)
# ==========================
class DetectorState:
    def __init__(self):
        self.mode = "SEARCHING"   # SEARCHING | LOCKED | LOST
        self.miss_count = 0
        self.ema_distance = None
        self.last_bbox = None     # (tl, br)

    def on_hit(self, tl, br, score):
        self.mode = "LOCKED"
        self.miss_count = 0
        self.last_bbox = (tl, br)

    def on_miss(self):
        self.miss_count += 1
        if self.mode == "LOCKED" and self.miss_count < MAX_LOST_FRAMES:
            self.mode = "LOST"   # ยังไม่ reset ทันที เผื่อกลับมาเจอ
        else:
            self.mode = "SEARCHING"
            if self.miss_count >= MAX_LOST_FRAMES:
                self.last_bbox = None  # ลืมกรอบเก่า


def detect_and_track(frame_bgr, templates, state: DetectorState):
    """
    คืนค่า: (frame_annotated, info_dict)

    info_dict:
      {
        'state': 'SEARCHING|LOCKED|LOST',
        'score': float,
        'distance': float|None,
        'distance_ema': float|None,
        'bbox': ((x1,y1),(x2,y2))|None
      }
    """
    overlay = frame_bgr.copy()
    info = {'state': state.mode, 'score': 0.0, 'distance': None, 'distance_ema': state.ema_distance, 'bbox': None}

    try:
        mask = make_yellow_mask(frame_bgr)
        tl, br, score = pick_best_match(mask, templates)
        info['score'] = score

        # เกณฑ์ตาม state (hysteresis)
        need = ACQUIRE_THR if state.mode in ("SEARCHING", "LOST") else KEEP_THR

        if tl is not None and score >= need:
            # hit
            px_w = max(1, br[0] - tl[0])
            px_h = max(1, br[1] - tl[1])
            dist, ema = compute_distance(px_w, px_h, state.ema_distance)
            state.ema_distance = ema
            state.on_hit(tl, br, score)

            # วาดกรอบ
            cv2.rectangle(overlay, tl, br, (0, 255, 0), 2)
            info.update({'state': state.mode, 'distance': dist, 'distance_ema': ema, 'bbox': (tl, br)})

            # ใส่ text
            cv2.putText(overlay, f"LOCKED  conf={score:.2f}", (10, 24), FONT, 0.7, (0,255,0), 2, cv2.LINE_AA)
            if ema is not None:
                cv2.putText(overlay, f"Dist(EMA) ~ {ema:.2f}", (10, 50), FONT, 0.7, (0,255,0), 2, cv2.LINE_AA)

        else:
            # miss
            prev_mode = state.mode
            state.on_miss()
            info['state'] = state.mode

            color = (0,255,255) if state.mode == "LOST" else (0,165,255)  # LOST=เหลือง, SEARCHING=ส้ม
            label = "LOST" if state.mode == "LOST" else "SEARCHING"
            cv2.putText(overlay, f"{label}  conf={score:.2f}", (10, 24), FONT, 0.7, color, 2, cv2.LINE_AA)

            # ถ้าเพิ่งหลุด (LOCKED->LOST) โชว์กรอบล่าสุดจาง ๆ (optional)
            if prev_mode == "LOCKED" and state.last_bbox is not None:
                (pt1, pt2) = state.last_bbox
                cv2.rectangle(overlay, pt1, pt2, (0, 255, 255), 1)

    except Exception as e:
        # อย่าให้ทั้งโปรแกรมดับ: ใส่ข้อความ error แล้วปล่อยวนต่อ
        cv2.putText(overlay, "ERROR in detect", (10, 24), FONT, 0.7, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(e).split("\n")[0][:60], (10, 50), FONT, 0.6, (0,0,255), 2, cv2.LINE_AA)
        # print traceback ไว้ดูใน console
        traceback.print_exc()

    # crosshair ตรงกลาง (ช่วยเล็ง)
    H, W = overlay.shape[:2]
    cx, cy = W//2, H//2
    cv2.drawMarker(overlay, (cx, cy), (255,255,255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)

    return overlay, info


# ==========================
# Main Loop (ทำงานตลอด)
# ==========================
def main():
    # เตรียมหุ่นและกล้อง
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution="720p")

    # โหลด template ครั้งเดียว
    templates = load_templates()

    # เตรียม state
    state = DetectorState()

    print("เริ่มทำงาน… กด Q เพื่อออก")
    try:
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=2.0)
            if frame is None:
                # ไม่มีเฟรมก็วนต่อ (ไม่หยุด)
                time.sleep(0.01)
                continue

            # ตรวจและวาดผล
            vis, info = detect_and_track(frame, templates, state)

            # แสดงผล
            cv2.imshow("Live (yellow detection + state machine)", vis)

            # กด Q เพื่อออก
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

    finally:
        cv2.destroyAllWindows()
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        try:
            ep_robot.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()