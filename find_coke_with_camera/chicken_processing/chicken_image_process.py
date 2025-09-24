# -*- coding: utf-8 -*-
"""
RoboMaster EP — Red Object Tracking + PID (Yaw/Pitch) + Full CSV Logging (angles-always-logged)
- ปรับให้ gimbal_yaw_angle_deg / gimbal_pitch_angle_deg ไม่เป็นค่าว่าง:
  1) seed 0 หลัง recenter
  2) รอ sub_angle callback ครั้งแรกสั้นๆ
  3) ถ้า callback ขาดช่วง ใช้ estimator จากคำสั่งที่ส่งไปชั่วคราว
"""

import robomaster
import cv2
import numpy as np
import time, csv, math, os, sys, traceback
from robomaster import robot

# ========= กล้อง/กิมบอล =========
CAMERA_W = 1280
CAMERA_H = 720
CAMERA_HFOV_DEG = 120.0
ASPECT = CAMERA_W / CAMERA_H
CAMERA_VFOV_DEG = math.degrees(2.0 * math.atan(math.tan(math.radians(CAMERA_HFOV_DEG/2.0)) / ASPECT))

GIMBAL_TURN_SPEED = 360
TEMPLATE_MATCH_THRESHOLD = 0.5
FLUSH_EVERY_N_ROWS = 20

class PID:
    def __init__(self, kp=0.8, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=10.0):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integ = 0.0; self.prev_err = 0.0
        self.integ_clip = integ_clip; self.out_clip = out_clip
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

def detect_red_object(img_read):
    img_hsv = cv2.cvtColor(img_read, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    img_binary = ((h > 0) & (h < 5)) | ((h > 170) & (h < 180))
    img_binary_uint8 = img_binary.astype(np.uint8) * 255

    t1 = cv2.imread(r"find_coke\bounding_box_1.jpg", cv2.IMREAD_COLOR)
    t2 = cv2.imread(r"find_coke\bounding_box_2.jpg", cv2.IMREAD_COLOR)
    t3 = cv2.imread(r"find_coke\bounding_box_3.jpg", cv2.IMREAD_COLOR)
    if t1 is None or t2 is None or t3 is None:
        return img_read, None, 0.0

    def to_bin(img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        _,_,vv = cv2.split(hsv)
        return (vv > 100).astype(np.uint8) * 255

    tb1, tb2, tb3 = to_bin(t1), to_bin(t2), to_bin(t3)
    v1 = cv2.matchTemplate(img_binary_uint8, tb1, cv2.TM_CCOEFF_NORMED)
    v2 = cv2.matchTemplate(img_binary_uint8, tb2, cv2.TM_CCOEFF_NORMED)
    v3 = cv2.matchTemplate(img_binary_uint8, tb3, cv2.TM_CCOEFF_NORMED)
    _, m1, _, loc1 = cv2.minMaxLoc(v1)
    _, m2, _, loc2 = cv2.minMaxLoc(v2)
    _, m3, _, loc3 = cv2.minMaxLoc(v3)

    max_val = m1; top_left = loc1; th, tw = tb1.shape[:2]
    if m2 > max_val: max_val = m2; top_left = loc2; th, tw = tb2.shape[:2]
    if m3 > max_val: max_val = m3; top_left = loc3; th, tw = tb3.shape[:2]
    if max_val < TEMPLATE_MATCH_THRESHOLD:
        return img_read, None, float(max_val)

    br = (top_left[0]+tw, top_left[1]+th)
    cv2.rectangle(img_read, top_left, br, (0,255,0), 2)
    cx = top_left[0] + tw//2
    cy = top_left[1] + th//2
    cv2.circle(img_read, (cx,cy), 4, (0,255,255), -1)
    cv2.circle(img_read, (CAMERA_W//2, CAMERA_H//2), 4, (255,0,0), -1)
    return img_read, (cx,cy), float(max_val)

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

    ep_robot = None; ep_camera = None; ep_gimbal = None

    LOG_DIR = os.path.join(os.getcwd(), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    ts_name = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"track_log_{ts_name}.csv")
    print("Log directory:", abs_path(LOG_DIR))
    print("CSV path:", abs_path(csv_path))

    with open(csv_path, "w", newline="", encoding="utf-8", buffering=1) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "t_s", "dt_s", "conf",
            "cx_px", "cy_px",
            "err_x_deg", "err_y_deg", "err_total_deg",
            "accumulate_err_x_deg_s", "accumulate_err_y_deg_s", "accumulate_err_total_deg_s",
            "controller_output_x_deg", "controller_output_y_deg",
            "gimbal_yaw_angle_deg", "gimbal_pitch_angle_deg",
        ])
        safe_flush(csv_file, csv_path)

        try:
            ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
            ep_camera = ep_robot.camera; ep_gimbal = ep_robot.gimbal

            try: ep_gimbal.recenter().wait_for_completed()
            except Exception: pass

            # ----- seed angles (0,0) และรอ callback ครั้งแรก -----
            gimbal_state = {"yaw": 0.0, "pitch": 0.0}  # seed หลัง recenter
            _got_cb = {"ok": False}
            def _gimbal_cb(angel_info):
                pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = angel_info
                gimbal_state["yaw"] = float(yaw_angle)
                gimbal_state["pitch"] = float(pitch_angle)
                _got_cb["ok"] = True

            ep_gimbal.sub_angle(freq=20, callback=_gimbal_cb)

            # รอ callback แรกเล็กน้อย (สูงสุด 1s) เพื่อให้ค่าไม่ว่างในบรรทัดแรกๆ
            t_wait0 = time.monotonic()
            while not _got_cb["ok"] and (time.monotonic() - t_wait0) < 1.0:
                time.sleep(0.01)

            # estimator สำรองถ้า callback ขาดช่วง
            est_yaw = gimbal_state["yaw"]
            est_pitch = gimbal_state["pitch"]

            ep_camera.start_video_stream(display=False, resolution='720p')
            print("เริ่มสตรีมแล้ว — กด Q เพื่อออก")
            print(f"HFOV={CAMERA_HFOV_DEG:.2f}°, VFOV={CAMERA_VFOV_DEG:.2f}°")

            pid_yaw   = PID(kp=0.2, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=10.0)
            pid_pitch = PID(kp=0.2, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=8.0)

            last_t = time.monotonic(); t0 = last_t
            acc_err_x = acc_err_y = acc_err_total = 0.0
            rows_written = 0

            while True:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=2)
                now = time.monotonic(); dt = now - last_t; last_t = now
                if frame is None:
                    time.sleep(0.01); continue

                vis, center, conf = detect_red_object(frame)

                yaw_err_deg = 0.0; pitch_err_deg = 0.0
                yaw_cmd = 0.0; pitch_cmd = 0.0
                cx = np.nan; cy = np.nan

                if center is None:
                    pid_yaw.integ *= 0.9; pid_pitch.integ *= 0.9
                else:
                    cx, cy = center
                    half_w = CAMERA_W / 2.0
                    px_off_x = (cx - half_w)
                    err_norm_x = px_off_x / half_w
                    yaw_err_deg = err_norm_x * (CAMERA_HFOV_DEG / 2.0)

                    half_h = CAMERA_H / 2.0
                    px_off_y = (cy - half_h)
                    err_norm_y = px_off_y / half_h
                    pitch_err_deg = - err_norm_y * (CAMERA_VFOV_DEG / 2.0)

                    yaw_cmd   = pid_yaw.step(yaw_err_deg, dt)
                    pitch_cmd = pid_pitch.step(pitch_err_deg, dt)

                    if abs(yaw_cmd) > 0.05 or abs(pitch_cmd) > 0.05:
                        try:
                            ep_gimbal.move(
                                yaw=yaw_cmd, pitch=pitch_cmd,
                                yaw_speed=GIMBAL_TURN_SPEED, pitch_speed=GIMBAL_TURN_SPEED
                            ).wait_for_completed()
                        except Exception:
                            try:
                                ep_gimbal.move(
                                    yaw=yaw_cmd, pitch=pitch_cmd,
                                    yaw_speed=GIMBAL_TURN_SPEED, pitch_speed=GIMBAL_TURN_SPEED
                                )
                            except Exception:
                                pass

                    cv2.line(vis, (CAMERA_W//2,0), (CAMERA_W//2,CAMERA_H), (255,0,0), 1)
                    cv2.line(vis, (0,CAMERA_H//2), (CAMERA_W,CAMERA_H//2), (255,0,0), 1)
                    cv2.putText(vis, f"yaw_err: {yaw_err_deg:+.2f}  pitch_err: {pitch_err_deg:+.2f}",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
                    cv2.putText(vis, f"pid_yaw: {yaw_cmd:+.2f}  pid_pitch: {pitch_cmd:+.2f}  conf:{conf:.2f}",
                                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

                # รวม/สะสม
                err_total_deg = math.hypot(yaw_err_deg, pitch_err_deg)
                acc_err_x     += yaw_err_deg   * dt
                acc_err_y     += pitch_err_deg * dt
                acc_err_total += err_total_deg * dt

                # ===== มุมกิมบอล: ใช้ค่าจริงถ้ามี มิฉะนั้นใช้ estimator =====
                # อัปเดต estimator ตามคำสั่งที่เพิ่งส่ง (approx. incremental)
                est_yaw  += yaw_cmd
                est_pitch += pitch_cmd

                # เลือกค่าใช้งาน: ถ้า callback เพิ่งอัปเดต ให้ใช้ค่าจริง
                yaw_out   = gimbal_state["yaw"]   if isinstance(gimbal_state["yaw"], (int,float)) else est_yaw
                pitch_out = gimbal_state["pitch"] if isinstance(gimbal_state["pitch"], (int,float)) else est_pitch

                # เขียน CSV
                writer.writerow([
                    f"{now - t0:.3f}", f"{dt:.4f}", f"{conf:.3f}",
                    "" if np.isnan(cx) else int(cx), "" if np.isnan(cy) else int(cy),
                    f"{yaw_err_deg:.3f}", f"{pitch_err_deg:.3f}", f"{err_total_deg:.3f}",
                    f"{acc_err_x:.3f}", f"{acc_err_y:.3f}", f"{acc_err_total:.3f}",
                    f"{yaw_cmd:.3f}", f"{pitch_cmd:.3f}",
                    f"{yaw_out:.3f}", f"{pitch_out:.3f}",
                ])
                rows_written += 1
                if rows_written % FLUSH_EVERY_N_ROWS == 0:
                    safe_flush(csv_file, csv_path)

                # UI
                cv2.imshow("Live (PID tracking red object: yaw+pitch)", vis)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')): break

        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt — stopping gracefully.")
        except Exception:
            print("\n[ERROR] Unhandled exception occurred:", file=sys.stderr)
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

            safe_flush(csv_file, csv_path)
            cv2.destroyAllWindows()
            try:
                size = os.path.getsize(csv_path)
                print(f"CSV saved → {abs_path(csv_path)}  ({size} bytes)")
            except Exception:
                print(f"CSV saved → {abs_path(csv_path)}")
