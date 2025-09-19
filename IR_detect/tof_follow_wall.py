# -*- coding: utf-8 -*-
from robomaster import robot
import time, math

# ====== CONFIG ======
TARGET_SIDE_CM = 10.0
SIDE_TOL_CM    = 10.0
FRONT_SPEED    = 0.20        # m/s
K_SIDE         = 0.02
K_PARALLEL     = 0.010
K_HEADING      = 0.015
MAX_VY         = 5        # m/s
MAX_YAW        = 360.0        # deg/s

STOP_FRONT_CM  = 35.0        # เกณฑ์ TOF หน้า
HIT_REARM_CM   = 33.0        # ต้อง “พ้น” ระยะนี้ก่อน ถึงจะนับ hit ครั้งถัดไปได้ (hysteresis)

# เกณฑ์ตัดสินมุม/ผนังหาย
WALL_NEAR_CM   = 30.0
WALL_GONE_CM   = 40.0
FRONT_SAFE_CM  = 35.0
CORNER_HOLD_S  = 0.30

# ====== Calibration: ADC -> cm ======
def adc_to_cm1(adc):   # เซนเซอร์ 1 (ซ้ายหน้า)
    return (892.5 - float(adc)) / 57.5

def adc_to_cm2(adc):   # เซนเซอร์ 2 (ซ้ายหลัง)
    return (917.5 - float(adc)) / 62.5

def clamp(v, lo, hi): 
    return max(lo, min(v, hi))

class EMA:
    def __init__(self, a=0.3, init=None):
        self.a=a; self.v=init
    def update(self, x):
        self.v = x if self.v is None else (self.a*x + (1-self.a)*self.v)
        return self.v

# ---- helpers ----
def ang_norm180(a):
    """ นอร์มองศาให้อยู่ในช่วง [-180, 180) """
    a = (a + 180.0) % 360.0 - 180.0
    return a

if __name__ == "__main__":
    ep = robot.Robot(); ep.initialize(conn_type="ap")
    ep_chassis = ep.chassis
    ep_sensor  = ep.sensor
    ep_adaptor = ep.sensor_adaptor
    ep_gimbal  = ep.gimbal
    ep_gimbal.recenter().wait_for_completed()
    # ---------- Subscriptions ----------
    front_tof_cm = 9999.0
    yaw_deg = 0.0

    def tof_cb(sub_info):
        nonlocal_front = sub_info  # just to avoid linter warning
        # sub_info[0] หน่วยมักเป็น mm หรือ dm แล้วแต่ SDK เวอร์ชัน (ที่นี่หาร 10 เป็น cm ตามโค้ดเดิม)
        # ป้องกัน error ด้วย try/except ไว้เหมือนเดิม
        global front_tof_cm
        try:
            front_tof_cm = float(sub_info[0]) / 10.0
        except:
            front_tof_cm = 9999.0

    def imu_cb(sub_info):
        global yaw_deg
        try:
            yaw_deg = float(sub_info[0])
        except:
            pass

    ep_sensor.sub_distance(freq=20, callback=tof_cb)
    ep_chassis.sub_attitude(freq=20, callback=imu_cb)

    # ตั้งกิมบอลครั้งเดียว (เลี่ยงชน action)
    try:
        ep_gimbal.moveto(pitch=0, yaw=0)  # ครั้งเดียวตอนเริ่ม
    except Exception:
        pass

    time.sleep(0.3)
    heading_ref = yaw_deg

    ema_side = EMA(0.3)
    ema_diff = EMA(0.3)
    ema_f    = EMA(0.4)

    # ---------- Hit counter (TOF front) ----------
    hit_count = 0
    can_count_hit = True   # จะถูก reset เป็น True เมื่อระยะหน้ากลับปลอดภัยพอ ( > HIT_REARM_CM )

    # ---------- State machine ----------
    FOLLOW, CORNER_RIGHT, SEARCH = range(3)
    state = FOLLOW
    corner_until = 0.0

    t0 = time.time()

    # ---- หมุนขวา ~90° พร้อมให้กิมบอลตามตลอด ----
    def rotate_right_deg(angle_deg=90.0, max_rate=60.0, tol_deg=2.0, timeout_s=3.0):
        """หมุนขวา angle_deg องศา โดยคุมลูปเอง + กิมบอลตามหัวหุ่น"""
        global heading_ref
        start = time.time()
        # target = yaw_deg (ปัจจุบัน) + angle
        target = yaw_deg + angle_deg
        # ปรับให้อยู่ช่วง [-180, 180)
        # เราจะคุมด้วย error แบบ normalized
        while True:
            e = ang_norm180(target - yaw_deg)  # error [-180, 180)
            if abs(e) <= tol_deg:
                break
            # ใช้ P-control กับ clamp
            z_cmd = clamp(1.5 * e, -max_rate, max_rate)  # deg/s
            ep_chassis.drive_speed(x=0.0, y=0.0, z=z_cmd)
            # หมุนกิมบอลตามทิศเดียวกัน
            try:
                ep_gimbal.drive_speed(pitch_speed=0.0, yaw_speed=z_cmd)
            except Exception:
                pass
            if time.time() - start > timeout_s:
                break
            time.sleep(0.02)
        # หยุดหมุน
        ep_chassis.drive_speed(0,0,0)
        try:
            ep_gimbal.drive_speed(0.0, 0.0)
        except Exception:
            pass
        # อัปเดต heading อ้างอิงใหม่
        heading_ref = yaw_deg

    try:
        while True:
            # อ่าน ADC (ปรับ id/port ให้ตรงฮาร์ดแวร์)
            adc_front_left = ep_adaptor.get_adc(id=1, port=1)
            adc_back_left  = ep_adaptor.get_adc(id=1, port=2)

            d_front = adc_to_cm1(adc_front_left)
            d_back  = adc_to_cm2(adc_back_left)

            # กรองนุ่ม ๆ
            d_front = EMA(0.35, d_front).update(d_front)
            d_back  = EMA(0.35, d_back ).update(d_back)

            d_side = ema_side.update((d_front + d_back)/2.0)
            diff   = ema_diff.update(d_back - d_front)
            fcm    = ema_f.update(front_tof_cm)

            # ----- Heading control รักษาทิศ -----
            e_heading = ang_norm180(heading_ref - yaw_deg)
            yaw_from_heading = clamp(K_HEADING * e_heading, -MAX_YAW, MAX_YAW)

            # ----- นับ Hit จาก TOF หน้า (ด้วย hysteresis) -----
            # ถ้าระยะหน้ากลับปลอดภัยมากพอ (> HIT_REARM_CM) → เปิดให้นับครั้งถัดไป
            if fcm > HIT_REARM_CM:
                can_count_hit = True

            # ถ้าต่ำกว่าเกณฑ์ และพร้อมนับ
            if fcm <= STOP_FRONT_CM and can_count_hit:
                if hit_count == 0:
                    # HIT ครั้งแรก → หมุนขวา 90° แล้ววิ่งต่อ
                    print(f"[HIT#1] Front TOF={fcm:.1f} cm  → rotate right 90°")
                    rotate_right_deg(angle_deg=90.0, max_rate=MAX_YAW, tol_deg=2.0, timeout_s=4.0)
                    hit_count += 1
                    can_count_hit = False  # ต้องรอให้หน้าปลอดภัยก่อนค่อยนับครั้งต่อไป
                elif hit_count == 1:
                    # HIT ครั้งที่สอง → หมุนขวา 30° แล้ววิ่งต่อ
                    print(f"[HIT#2] Front TOF={fcm:.1f} cm  → rotate right 30°")
                    rotate_right_deg(angle_deg=30.0, max_rate=MAX_YAW, tol_deg=2.0, timeout_s=2.0)
                    hit_count += 1
                    can_count_hit = False
                elif hit_count == 2:
                    # HIT ครั้งที่สาม → หยุด
                    print(f"[HIT#3] Front TOF={fcm:.1f} cm  → STOP")
                    ep_chassis.drive_speed(0,0,0)
                    try:
                        ep_gimbal.drive_speed(0.0, 0.0)
                    except Exception:
                        pass
                    break  # จบภารกิจ

            # ====== ตรวจมุม/ผนังซ้ายหาย (ใช้กับ FOLLOW) ======
            wall_left_now  = (d_front < WALL_NEAR_CM) or (d_back < WALL_NEAR_CM)
            wall_gone_now  = (d_front > WALL_GONE_CM) and (d_back > WALL_GONE_CM)

            # ----- ตรรกะหลักแต่ละสถานะ -----
            if state == FOLLOW:
                # 1) เกาะกำแพงซ้าย
                e_side = TARGET_SIDE_CM - d_side
                vy_cmd = clamp(K_SIDE * e_side, -MAX_VY, MAX_VY)

                # 2) ปรับให้ขนานกำแพงจากส่วนต่างหน้า-หลัง
                yaw_from_parallel = clamp(K_PARALLEL * diff * 180.0, -MAX_YAW, MAX_YAW)
                wz_cmd = clamp(yaw_from_parallel + yaw_from_heading, -MAX_YAW, MAX_YAW)

                # สั่งวิ่ง + สั่งกิมบอลให้หมุนตาม z (รักษาทิศเดียวกับหุ่น)
                ep_chassis.drive_speed(x=FRONT_SPEED, y=vy_cmd, z=wz_cmd)
                try:
                    ep_gimbal.drive_speed(pitch_speed=0.0, yaw_speed=wz_cmd)
                except Exception:
                    pass

                # ถ้ากำแพงซ้าย "หาย" และหน้าปลอดภัย → เข้าโหมดเลี้ยวขวานิ่ม ๆ
                if wall_gone_now and fcm > FRONT_SAFE_CM:
                    state = CORNER_RIGHT
                    corner_until = time.time() + CORNER_HOLD_S
                    print("[CORNER_RIGHT] prepare")

            elif state == CORNER_RIGHT:
                turn_cmd = +MAX_YAW * 0.6
                ep_chassis.drive_speed(x=FRONT_SPEED*0.5, y=0.0, z=turn_cmd)
                try:
                    ep_gimbal.drive_speed(0.0, turn_cmd)
                except Exception:
                    pass

                if (d_front < WALL_NEAR_CM) or (d_back < WALL_NEAR_CM):
                    state = FOLLOW
                    heading_ref = yaw_deg
                elif time.time() > corner_until + 1.2:
                    state = SEARCH

            else:  # SEARCH
                sweep = +MAX_YAW*0.5 if math.sin(time.time()*1.2) > 0 else -MAX_YAW*0.5
                ep_chassis.drive_speed(x=0.06, y=0.0, z=sweep)
                try:
                    ep_gimbal.drive_speed(0.0, sweep)
                except Exception:
                    pass
                if (d_front < WALL_NEAR_CM) or (d_back < WALL_NEAR_CM):
                    heading_ref = yaw_deg
                    state = FOLLOW

            print(f"{state=}  LF={d_front:5.1f} LB={d_back:5.1f}  Side={d_side:5.1f}  "
                  f"TOF={fcm:5.1f}  yaw={yaw_deg:+6.2f}  hits={hit_count}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        ep_chassis.drive_speed(0,0,0)
        try:
            ep_gimbal.drive_speed(0.0, 0.0)
        except Exception:
            pass
        ep_sensor.unsub_distance()
        ep_chassis.unsub_attitude()
        ep.close()