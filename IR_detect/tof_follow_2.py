# -*- coding: utf-8 -*-
from robomaster import robot
import time, math

# ================= CONFIG =================
TARGET_SIDE_CM = 10.0        # ระยะที่อยากรักษาจากกำแพงซ้าย (cm)
SIDE_TOL_CM    = 10.0

FRONT_SPEED    = 0.10        # m/s ความเร็วเดินหน้า
K_SIDE         = 0.02        # gain ปรับ y ตาม error ระยะด้านข้าง
K_PARALLEL     = 0.010       # gain ปรับ yaw ให้ขนานกำแพง (ต่างหน้า-หลังซ้าย)
K_HEADING      = 0.015       # gain ปรับ yaw ให้ตรง heading อ้างอิง (IMU)

MAX_VY         = 0.50        # m/s จำกัดความเร็วแกน y
MAX_YAW        = 60.0        # deg/s จำกัดอัตราหมุน

# เงื่อนไขหยุด (ปรับให้ไม่สะดุดคาโค้ง)
STOP_FRONT_CM       = 12.0   # หยุดเมื่อ TOF หน้าต่ำกว่าค่านี้ (เฉพาะตอนวิ่งตรง)
STOP_HOLD_S         = 0.50   # ต้องต่ำกว่าค่านี้ "ต่อเนื่อง" อย่างน้อยเท่านี้วินาที
STOP_STRAIGHT_DEG   = 5.0    # ต้องวิ่งตรง (|error heading| <= 5°)
STOP_WHEN_WZ_DEG    = 8.0    # ต้องไม่กำลังหักเลี้ยวแรง (|wz_cmd| <= 8 deg/s)

# เกณฑ์ตรวจจับมุม/ผนังหาย
WALL_NEAR_CM   = 30.0        # ถือว่ามีผนังซ้าย ถ้าระยะ < ค่านี้
WALL_GONE_CM   = 40.0        # ถือว่าผนังหาย ถ้าระยะ > ค่านี้ (hysteresis)
FRONT_SAFE_CM  = 45.0        # หน้าปลอดภัยพอจะเลี้ยว (กันตัน)
CORNER_HOLD_S  = 0.35        # เลี้ยวขวาค้างช่วงสั้น ๆ เพื่อ “กัดมุม”

# กิมบอลให้ชี้ไปข้างหน้าเสมอ (และไม่ยิงซ้ำ)
GIMBAL_PITCH_DEG = -10       # ก้มลงเล็กน้อย
GIMBAL_RESYNC_S  = 0.30      # คอยสั่งย้ำทุก ๆ กี่วินาทีให้ชี้ไปข้างหน้า

# ============ Calibration: ADC -> cm ============
def adc_to_cm1(adc):   # เซนเซอร์ 1 (ซ้ายหน้า)
    # x(cm) = (892.5 - y(adc)) / 57.5
    return (892.5 - float(adc)) / 57.5

def adc_to_cm2(adc):   # เซนเซอร์ 2 (ซ้ายหลัง)
    # x(cm) = (917.5 - y(adc)) / 62.5
    return (917.5 - float(adc)) / 62.5

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

class EMA:
    """ตัวกรองค่าเฉลี่ยเอ็กซ์โปเนนเชียล กันสั่น"""
    def __init__(self, a=0.3, init=None):
        self.a = a
        self.v = init
    def update(self, x):
        self.v = x if self.v is None else (self.a*x + (1-self.a)*self.v)
        return self.v

if __name__ == "__main__":
    ep = robot.Robot()
    ep.initialize(conn_type="ap")

    ep_chassis = ep.chassis
    ep_sensor  = ep.sensor
    ep_adaptor = ep.sensor_adaptor
    ep_gimbal  = ep.gimbal

    # ---------- Subscriptions ----------
    front_tof_cm = 9999.0
    yaw_deg = 0.0

    def tof_cb(sub_info):
        global front_tof_cm
        try:
            front_tof_cm = float(sub_info[0]) / 10.0  # mm -> cm
        except:
            front_tof_cm = 9999.0

    def imu_cb(sub_info):
        global yaw_deg
        try:
            yaw_deg = float(sub_info[0])  # yaw (deg)
        except:
            pass

    ep_sensor.sub_distance(freq=20, callback=tof_cb)
    ep_chassis.sub_attitude(freq=20, callback=imu_cb)

    # ตั้งกิมบอลให้ชี้ไปข้างหน้า + เตรียม handle สำหรับกันยิงซ้ำ
    ep_gimbal.recenter()
    gimbal_action = ep_gimbal.moveto(pitch=GIMBAL_PITCH_DEG, yaw=0)
    last_gimbal_sync = time.time()

    # รอค่าแรก ๆ แล้วล็อก heading อ้างอิง
    time.sleep(0.3)
    heading_ref = yaw_deg

    # --------- Filters ---------
    ema_side = EMA(0.3)
    ema_diff = EMA(0.3)
    ema_f    = EMA(0.4)

    # ---------- State machine ----------
    FOLLOW, CORNER_RIGHT, SEARCH = range(3)
    state = FOLLOW
    corner_until = 0.0
    stop_since = None

    t0 = time.time()
    try:
        while True:
            # อ่าน ADC (ปรับ id/port ให้ตรงฮาร์ดแวร์จริง)
            adc_front_left = ep_adaptor.get_adc(id=1, port=1)
            adc_back_left  = ep_adaptor.get_adc(id=1, port=2)

            d_front = adc_to_cm1(adc_front_left)
            d_back  = adc_to_cm2(adc_back_left)

            # กรองค่าให้เนียน
            d_side = ema_side.update((d_front + d_back) / 2.0)
            diff   = ema_diff.update(d_back - d_front)     # >0 = ท้ายห่างกว่าหน้า (หน้าชิดกำแพง)
            fcm    = ema_f.update(front_tof_cm)

            # --- Heading error ---
            e_heading = (heading_ref - yaw_deg)
            e_heading = (e_heading + 180) % 360 - 180  # นอมอลไลซ์ [-180, 180]
            yaw_from_heading = clamp(K_HEADING * e_heading, -MAX_YAW, MAX_YAW)

            # เตรียมค่าเริ่มต้นสำหรับการเช็ค stop
            wz_cmd = 0.0
            vy_cmd = 0.0

            if state == FOLLOW:
                # 1) เกาะระยะด้านข้าง
                e_side = TARGET_SIDE_CM - d_side           # >0 = ไกลกำแพงไป → เลื่อนเข้า (y+)
                vy_cmd = clamp(K_SIDE * e_side, -MAX_VY, MAX_VY)

                # 2) ทำขนานกำแพงจากต่างหน้า-หลัง
                yaw_from_parallel = clamp(K_PARALLEL * diff * 180.0, -MAX_YAW, MAX_YAW)

                # 3) รวม yaw ทั้งสองแหล่ง
                wz_cmd = clamp(yaw_from_parallel + yaw_from_heading, -MAX_YAW, MAX_YAW)

                # 4) Corner detection → เข้าโหมดเลี้ยวขวา
                wall_gone_now = (d_front > WALL_GONE_CM) and (d_back > WALL_GONE_CM)
                if wall_gone_now and (fcm > FRONT_SAFE_CM):
                    state = CORNER_RIGHT
                    corner_until = time.time() + CORNER_HOLD_S
                    ep_chassis.drive_speed(x=FRONT_SPEED*0.6, y=0.0, z=+MAX_YAW*0.6)
                else:
                    ep_chassis.drive_speed(x=FRONT_SPEED, y=vy_cmd, z=wz_cmd)

            elif state == CORNER_RIGHT:
                # เลี้ยวขวาค้างช่วงสั้น ๆ เพื่อเจอแนวผนังซ้ายชุดใหม่
                turn_cmd = +MAX_YAW * 0.6
                wz_cmd = turn_cmd
                ep_chassis.drive_speed(x=FRONT_SPEED*0.5, y=0.0, z=turn_cmd)

                # กลับเข้า FOLLOW เมื่อเริ่มเห็นผนังซ้าย
                if (d_front < WALL_NEAR_CM) or (d_back < WALL_NEAR_CM):
                    state = FOLLOW
                    heading_ref = yaw_deg  # อัปเดตแนวอ้างอิงตามแนวทางใหม่
                # ถ้าเลี้ยวนานเกิน → เข้า SEARCH
                elif time.time() > corner_until + 1.2:
                    state = SEARCH

            else:  # SEARCH
                # กวาดหา: เดินหน้าช้า + ส่ายหัวซ้ายขวา
                sweep = +MAX_YAW*0.5 if math.sin(time.time()*1.2) > 0 else -MAX_YAW*0.5
                wz_cmd = sweep
                ep_chassis.drive_speed(x=0.06, y=0.0, z=sweep)

                # เมื่อเห็นผนังซ้ายอีกครั้ง → FOLLOW
                if (d_front < WALL_NEAR_CM) or (d_back < WALL_NEAR_CM):
                    heading_ref = yaw_deg
                    state = FOLLOW

            # ---------- กิมบอลให้ชี้ไปข้างหน้าเสมอ (ป้องกัน “กำลังทำ action อยู่”) ----------
            now = time.time()
            if now - last_gimbal_sync >= GIMBAL_RESYNC_S:
                try:
                    finished = True
                    if gimbal_action is not None:
                        # SDK บางเวอร์ชันมี has_finished; ถ้าไม่มีจะ except
                        finished = bool(getattr(gimbal_action, "has_finished", True))
                    if finished:
                        gimbal_action = ep_gimbal.moveto(pitch=GIMBAL_PITCH_DEG, yaw=0)
                        last_gimbal_sync = now
                except Exception:
                    # ถ้ายังยุ่งอยู่ ไม่ยิงซ้ำ รอรอบหน้า
                    pass

            # ---------- STOP debounce (เฉพาะตอน FOLLOW และวิ่งตรง) ----------
            straight_ok = abs(e_heading) <= STOP_STRAIGHT_DEG
            turning_ok  = abs(wz_cmd)    <= STOP_WHEN_WZ_DEG
            stop_ok     = (state == FOLLOW) and straight_ok and turning_ok and (fcm <= STOP_FRONT_CM)

            if stop_ok:
                if stop_since is None:
                    stop_since = now
                elif now - stop_since >= STOP_HOLD_S:
                    ep_chassis.drive_speed(0, 0, 0)
                    print(f"[STOP] Front TOF={fcm:.1f} cm  t={now - t0:.2f}s")
                    break
            else:
                stop_since = None

            # ---------- Debug ----------
            print(f"state={['FOLLOW','CORNER_RIGHT','SEARCH'][state]}  "
                  f"LF={d_front:5.1f}  LB={d_back:5.1f}  Side={d_side:5.1f}  "
                  f"vy={vy_cmd:+.2f}  wz={wz_cmd:+5.1f}  "
                  f"TOF={fcm:5.1f}  yaw={yaw_deg:+7.2f}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ep_chassis.drive_speed(0,0,0)
            ep_sensor.unsub_distance()
            ep_chassis.unsub_attitude()
        finally:
            ep.close()
