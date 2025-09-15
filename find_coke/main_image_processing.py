# -*- coding: utf-8 -*-
"""
RoboMaster EP — Distance-first + PID Tracking (Yaw/Pitch) + Overlay
- โฟกัสวัดระยะจากขนาดกรอบ (tight bbox) แล้วใช้ PID หันตามเป้า
- สีหลัก: โทนเหลือง (HSV h≈25-40, s>160, v>100)
"""

import os, sys, time, math, csv, traceback
import cv2
import numpy as np
import robomaster
from robomaster import robot

# ========= กล้อง/กิมบอล =========
CAMERA_W = 1280
CAMERA_H = 720
CAMERA_HFOV_DEG = 120.0
ASPECT = CAMERA_W / CAMERA_H
CAMERA_VFOV_DEG = math.degrees(2.0 * math.atan(math.tan(math.radians(CAMERA_HFOV_DEG/2.0)) / ASPECT))

GIMBAL_TURN_SPEED = 360
FLUSH_EVERY_N_ROWS = 20

# ========= ค่าคงที่วัดระยะ (จากงานเดิมของคุณ) =========
# ใช้สองแกน แล้วเฉลี่ย
KPIX_X = 578.57; OBJ_W_CM = 5.6   # จาก size_x
KPIX_Y = 587.75; OBJ_H_CM = 14.7  # จาก size_y

# ========= เกณฑ์การตรวจจับ =========
TEMPLATE_MATCH_THRESHOLD = 0.40  # เกณฑ์ผ่าน
HSV_H_LO, HSV_H_HI = 25, 40
HSV_S_MIN = 160
HSV_V_MIN = 100

TEMPLATE_PATHS = [
    r"find_coke\box_1_test.jpg",
    r"find_coke\box_2_test.jpg",
    r"find_coke\box_3_test.jpg",
]

# ========= PID =========
class PID:
    def __init__(self, kp=0.2, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integ = 0.0
        self.prev_err = 0.0
        self.integ_clip = float(integ_clip)
        self.out_clip = float(out_clip)
    def reset(self):
        self.integ = 0.0; self.prev_err = 0.0
    def step(self, err, dt):
        dt = max(dt, 1e-3)
        self.integ += err * dt
        self.integ = max(-self.integ_clip, min(self.integ, self.integ_clip))
        deriv = (err - self.prev_err) / dt
        self.prev_err = err
        out = self.kp*err + self.ki*self.integ + self.kd*deriv
        return max(-self.out_clip, min(self.out_clip, out))

# ========= utils =========
def _clip(v, lo, hi): return max(lo, min(int(v), hi))

def _tight_bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if ys.size == 0: return None
    y0, y1 = int(ys.min()), int(ys.max())+1
    x0, x1 = int(xs.min()), int(xs.max())+1
    return x0, y0, x1, y1  # x_min, y_min, x_max(excl), y_max(excl)

def _tpl_to_bin_u8(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return (v > 100).astype(np.uint8) * 255

def detect_and_measure(img_bgr):
    """
    return:
      vis_bgr, center_xy or None, conf(float), size_x(int), size_y(int), dist_x_cm(float), dist_y_cm(float), dist_avg_cm(float)
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h > HSV_H_LO) & (h < HSV_H_HI) & (s > HSV_S_MIN) & (v > HSV_V_MIN)
    mask_u8 = (mask.astype(np.uint8) * 255)

    # โหลด template
    tpl_bins = []
    for p in TEMPLATE_PATHS:
        t = cv2.imread(p)
        if t is None:
            raise FileNotFoundError(f"ไม่พบไฟล์เทมเพลต: {p}")
        tpl_bins.append(_tpl_to_bin_u8(t))

    # ตรวจสอบขนาด
    for i, T in enumerate(tpl_bins):
        th, tw = T.shape[:2]
        if th > H or tw > W:
            raise ValueError(f"Template index {i} ใหญ่กว่าภาพ (th={th}, tw={tw}, H={H}, W={W})")

    # template matching
    scores = []
    locs = []
    sizes = []
    for T in tpl_bins:
        val = cv2.matchTemplate(mask_u8, T, cv2.TM_CCOEFF_NORMED)
        _, m, _, loc = cv2.minMaxLoc(val)
        scores.append(m); locs.append(loc); sizes.append(T.shape[:2])

    best_idx = int(np.argmax(scores))
    max_val = float(scores[best_idx])
    top_left = locs[best_idx]
    th, tw = sizes[best_idx]

    vis = img_bgr.copy()
    cx = cy = np.nan
    size_x = size_y = 0
    dist_x = dist_y = dist_avg = 0.0
    center = None

    if max_val > TEMPLATE_MATCH_THRESHOLD:
        # window จากตำแหน่งแมตช์
        x0, y0 = int(top_left[0]), int(top_left[1])
        x1, y1 = x0 + tw, y0 + th
        x0c, y0c = _clip(x0, 0, W-1), _clip(y0, 0, H-1)
        x1c, y1c = _clip(x1, 1, W), _clip(y1, 1, H)

        win_mask = mask[y0c:y1c, x0c:x1c]
        tight = _tight_bbox_from_mask(win_mask)
        if tight is not None:
            tx0, ty0, tx1, ty1 = tight
            X0, Y0 = x0c + tx0, y0c + ty0
            X1, Y1 = x0c + tx1, y0c + ty1
            cv2.rectangle(vis, (X0, Y0), (X1, Y1), (0, 255, 0), 2)

            size_x = int(X1 - X0)
            size_y = int(Y1 - Y0)

            if size_x > 0: dist_x = KPIX_X * OBJ_W_CM / float(size_x)
            if size_y > 0: dist_y = KPIX_Y * OBJ_H_CM / float(size_y)
            # ถ้ามีทั้งสองแกน ใช้เฉลี่ย, ถ้ามีแกนเดียว ใช้อันนั้น
            if dist_x > 0 and dist_y > 0:
                dist_avg = 0.5 * (dist_x + dist_y)
            else:
                dist_avg = max(dist_x, dist_y)

            cx = int((X0 + X1) / 2)
            cy = int((Y0 + Y1) / 2)
            center = (cx, cy)

            # วาด crosshair/ข้อความ
            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)
            cv2.circle(vis, (W//2, H//2), 4, (255, 0, 0), -1)
            cv2.line(vis, (W//2, 0), (W//2, H), (255, 0, 0), 1)
            cv2.line(vis, (0, H//2), (W, H//2), (255, 0, 0), 1)
            cv2.putText(vis, f"conf={max_val:.2f}  w={size_x}px h={size_y}px",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.putText(vis, f"dist_x={dist_x:.2f}cm  dist_y={dist_y:.2f}cm  avg={dist_avg:.2f}cm",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        else:
            max_val = 0.0  # หน้าต่างไม่มีพิกเซลผ่าน mask
    else:
        # ไม่ผ่านเกณฑ์
        pass

    return vis, center, max_val, size_x, size_y, dist_x, dist_y, dist_avg

def safe_flush(file_obj, path):
    try:
        file_obj.flush(); os.fsync(file_obj.fileno())
    except Exception as e:
        print("Flush/Fsync error:", repr(e), file=sys.stderr)

def abs_path(p):
    try: return os.path.abspath(p)
    except Exception: return p

if __name__ == "__main__":
    print("Working directory:", abs_path(os.getcwd()))
    print(f"HFOV={CAMERA_HFOV_DEG:.2f}°, VFOV={CAMERA_VFOV_DEG:.2f}°")

    ep_robot = None; ep_camera = None; ep_gimbal = None

    try:
        ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_gimbal = ep_robot.gimbal
        ep_camera = ep_robot.camera

        try: ep_gimbal.recenter().wait_for_completed()
        except Exception: pass

        # ---- seed/subscribe มุมกิมบอล (กันค่าว่าง) ----
        gimbal_state = {"yaw": 0.0, "pitch": 0.0}
        _got = {"ok": False}
        def _gimbal_cb(ang):
            pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = ang
            gimbal_state["yaw"]   = float(yaw_angle)
            gimbal_state["pitch"] = float(pitch_angle)
            _got["ok"] = True
        ep_gimbal.sub_angle(freq=20, callback=_gimbal_cb)

        t_wait0 = time.monotonic()
        while not _got["ok"] and (time.monotonic()-t_wait0) < 1.0:
            time.sleep(0.01)

        est_yaw = gimbal_state["yaw"]
        est_pitch = gimbal_state["pitch"]

        ep_camera.start_video_stream(display=False, resolution="720p")
        print("เริ่มสตรีม — กด Q เพื่อออก")

        pid_yaw   = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=10.0)
        pid_pitch = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip= 8.0)

        t0 = time.monotonic(); last_t = t0

        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=2)
            now = time.monotonic(); dt = now - last_t; last_t = now
            if frame is None:
                time.sleep(0.01); continue

            vis, center, conf, sx, sy, dx, dy, davg = detect_and_measure(frame)

            yaw_err_deg = pitch_err_deg = 0.0
            yaw_cmd = pitch_cmd = 0.0
            cx = cy = ""

            if center is not None:
                cx_i, cy_i = center
                cx, cy = int(cx_i), int(cy_i)

                # error เป็นองศาตามสัดส่วนพิกเซล
                half_w = CAMERA_W / 2.0
                half_h = CAMERA_H / 2.0
                err_x = (cx - half_w) / half_w
                err_y = (cy - half_h) / half_h

                yaw_err_deg   =  err_x * (CAMERA_HFOV_DEG/2.0)
                pitch_err_deg = -err_y * (CAMERA_VFOV_DEG/2.0)

                yaw_cmd   = pid_yaw.step(yaw_err_deg, dt)
                pitch_cmd = pid_pitch.step(pitch_err_deg, dt)

                if abs(yaw_cmd) > 0.05 or abs(pitch_cmd) > 0.05:
                    try:
                        ep_gimbal.move(yaw=yaw_cmd, pitch=pitch_cmd,
                                       yaw_speed=GIMBAL_TURN_SPEED, pitch_speed=GIMBAL_TURN_SPEED
                                       ).wait_for_completed()
                    except Exception:
                        try:
                            ep_gimbal.move(yaw=yaw_cmd, pitch=pitch_cmd,
                                           yaw_speed=GIMBAL_TURN_SPEED, pitch_speed=GIMBAL_TURN_SPEED)
                        except Exception:
                            pass
            else:
                # soften integral ถ้าไม่เห็นเป้า
                pid_yaw.integ *= 0.9
                pid_pitch.integ *= 0.9

            # อัปเดต estimator ให้ไม่ว่าง
            est_yaw  += yaw_cmd
            est_pitch += pitch_cmd
            yaw_out   = gimbal_state["yaw"]   if isinstance(gimbal_state["yaw"], (int,float)) else est_yaw
            pitch_out = gimbal_state["pitch"] if isinstance(gimbal_state["pitch"], (int,float)) else est_pitch

            # overlay เพิ่ม
            cv2.putText(vis, f"dist_avg={davg:.2f} cm", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
            cv2.putText(vis, f"yaw_err={yaw_err_deg:+.2f} pitch_err={pitch_err_deg:+.2f}",
                        (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
            cv2.putText(vis, f"pid_yaw={yaw_cmd:+.2f} pid_pitch={pitch_cmd:+.2f}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)

            cv2.imshow("Measure + PID track (yellow object)", vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — stopping.")
    except Exception:
        print("\n[ERROR] Unhandled exception:", file=sys.stderr)
        traceback.print_exc()
    finally:
        try:
            if ep_camera is not None: ep_camera.stop_video_stream()
        except Exception: pass
        try:
            if ep_gimbal is not None: ep_gimbal.unsub_angle()
        except Exception: pass
        try:
            if ep_robot is not None: ep_robot.close()
        except Exception: pass
        cv2.destroyAllWindows()
