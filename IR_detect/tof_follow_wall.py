# -*- coding: utf-8 -*-
from robomaster import robot
import time, math

# ====== CONFIG ======
TARGET_SIDE_CM = 10.0        # ระยะจากกำแพงซ้ายที่อยากรักษา (cm)
SIDE_TOL_CM    = 10.0
FRONT_SPEED    = 0.10       # m/s เดินหน้า
K_SIDE         = 0.02       # ปรับ y ตาม error ระยะด้านข้าง
K_PARALLEL     = 0.010      # ปรับ yaw ให้ขนานกำแพง (ต่างหน้า-หลังซ้าย)
K_HEADING      = 0.015      # ปรับ yaw ให้ตรงแนว (IMU)
MAX_VY         = 0.75       # m/s
MAX_YAW        = 60.0       # deg/s
STOP_FRONT_CM  = 25.0       # หยุดเมื่อ TOF หน้า <= 10 cm (ตามโจทย์)

# เกณฑ์ตัดสินมุม/ผนังหาย
WALL_NEAR_CM   = 30.0       # ถือว่ามี “ผนังซ้าย” ถ้าระยะ < ค่านี้
WALL_GONE_CM   = 40.0       # ถือว่า “ผนังหาย” ถ้าระยะ > ค่านี้ (ฮิสเทอรีซิส)
FRONT_SAFE_CM  = 35.0       # ด้านหน้าปลอดภัยพอจะเลี้ยว (กันตัน)
CORNER_HOLD_S  = 0.30       # ค้างเวลาเลี้ยวขวาขณะเข้ามุม (s) กันสะดุ้ง

# ====== Calibration: ADC -> cm ======
def adc_to_cm1(adc):   # เซนเซอร์ 1 (ซ้ายหน้า)
    return (892.5 - float(adc)) / 57.5

def adc_to_cm2(adc):   # เซนเซอร์ 2 (ซ้ายหลัง)
    return (917.5 - float(adc)) / 62.5

def clamp(v, lo, hi): return max(lo, min(v, hi))

class EMA:
    def __init__(self, a=0.3, init=None):
        self.a=a; self.v=init
    def update(self, x):
        self.v = x if self.v is None else (self.a*x + (1-self.a)*self.v)
        return self.v

if __name__ == "__main__":
    ep = robot.Robot(); ep.initialize(conn_type="ap")
    ep_chassis = ep.chassis
    ep_sensor  = ep.sensor
    ep_adaptor = ep.sensor_adaptor
    ep_gimbal = ep.gimbal

    # ---------- Subscriptions ----------
    front_tof_cm = 9999.0
    yaw_deg = 0.0

    def tof_cb(sub_info):
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

    time.sleep(0.3)
    heading_ref = yaw_deg

    ema_side = EMA(0.3)
    ema_diff = EMA(0.3)
    ema_f    = EMA(0.4)

    # ---------- State machine ----------
    FOLLOW, CORNER_RIGHT, SEARCH = range(3)
    state = FOLLOW
    corner_until = 0.0

    t0 = time.time()
    try:

        while True:
            # อ่าน ADC (ปรับ id/port ให้ตรงฮาร์ดแวร์)
            adc_front_left = ep_adaptor.get_adc(id=1, port=1)
            adc_back_left  = ep_adaptor.get_adc(id=1, port=2)

            d_front = adc_to_cm1(adc_front_left)
            d_back  = adc_to_cm2(adc_back_left)

            # กรอง
            d_front = EMA(0.35, d_front).update(d_front)  # one-shot smooth
            d_back  = EMA(0.35, d_back ).update(d_back )

            d_side = ema_side.update((d_front + d_back)/2.0)
            diff   = ema_diff.update(d_back - d_front)
            fcm    = ema_f.update(front_tof_cm)

            # ----- Heading control (ร่วมกับทุกสถานะ) -----
            e_heading = (heading_ref - yaw_deg)
            e_heading = (e_heading + 180) % 360 - 180
            yaw_from_heading = clamp(K_HEADING * e_heading, -MAX_YAW, MAX_YAW)

            # 3) Corner detection: ถ้าผนังซ้าย "หาย" แต่หน้าตัน (กำแพงแรก)
            wall_left_now  = (d_front < WALL_NEAR_CM) or (d_back < WALL_NEAR_CM)
            wall_gone_now  = (d_front > WALL_GONE_CM) and (d_back > WALL_GONE_CM)

            # เงื่อนไขเลี้ยวขวาเมื่อเจอกำแพงแรก (หน้าตันและผนังซ้ายหาย)
            if state == FOLLOW and wall_gone_now and fcm <= STOP_FRONT_CM:
                state = CORNER_RIGHT
                corner_until = time.time() + CORNER_HOLD_S
                ep_chassis.drive_speed(x=FRONT_SPEED*0.6, y=0.0, z=+MAX_YAW*0.6)
                print(f"[CORNER_RIGHT] Front TOF={fcm:.1f} cm  t={time.time()-t0:.2f}s")
                continue

            # ----- STOP -----
            if fcm <= STOP_FRONT_CM:
                ep_chassis.drive_speed(0,0,0)
                print(f"[STOP] Front TOF={fcm:.1f} cm  t={time.time()-t0:.2f}s")
                break

            if state == FOLLOW:
                # 1) เกาะกำแพงซ้าย
                e_side = TARGET_SIDE_CM - d_side
                vy_cmd = clamp(K_SIDE * e_side, -MAX_VY, MAX_VY)

                # 2) ทำขนานกำแพงจากต่างหน้า-หลัง
                yaw_from_parallel = clamp(K_PARALLEL * diff * 180.0, -MAX_YAW, MAX_YAW)
                wz_cmd = clamp(yaw_from_parallel + yaw_from_heading, -MAX_YAW, MAX_YAW)

                # Corner detection: ถ้าผนังซ้าย "หาย" แต่หน้าปลอดภัย -> เตรียมเลี้ยวขวา
                if wall_gone_now and fcm > FRONT_SAFE_CM:
                    state = CORNER_RIGHT
                    corner_until = time.time() + CORNER_HOLD_S
                    ep_chassis.drive_speed(x=FRONT_SPEED*0.6, y=0.0, z=+MAX_YAW*0.6)
                else:
                    ep_chassis.drive_speed(x=FRONT_SPEED, y=vy_cmd, z=wz_cmd)

            elif state == CORNER_RIGHT:
                turn_cmd = +MAX_YAW*0.6
                ep_chassis.drive_speed(x=FRONT_SPEED*0.5, y=0.0, z=turn_cmd)

                if (d_front < WALL_NEAR_CM) or (d_back < WALL_NEAR_CM):
                    state = FOLLOW
                    heading_ref = yaw_deg
                elif time.time() > corner_until + 1.2:
                    state = SEARCH

            else:  # SEARCH
                sweep = +MAX_YAW*0.5 if math.sin(time.time()*1.2) > 0 else -MAX_YAW*0.5
                ep_chassis.drive_speed(x=0.06, y=0.0, z=sweep)
                if (d_front < WALL_NEAR_CM) or (d_back < WALL_NEAR_CM):
                    heading_ref = yaw_deg
                    state = FOLLOW

            print(f"{state=}  LF={d_front:5.1f} LB={d_back:5.1f}  Side={d_side:5.1f}  "
                  f"TOF={fcm:5.1f}  yaw={yaw_deg:+6.2f}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        ep_chassis.drive_speed(0,0,0)
        ep_sensor.unsub_distance()
        ep_chassis.unsub_attitude()
        ep.close()
