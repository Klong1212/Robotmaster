import time, threading, math, csv, os
from datetime import datetime
from robomaster import robot

GRID_SIZE_M = 0.6
WALL_THRESHOLD_MM = 500
VISION_SCAN_DURATION_S = 0.5
GIMBAL_TURN_SPEED = 200
BUMP_STOP_MM = 230

ORIENT = {0: "N", 1: "E", 2: "S", 3: "W"}
WALLS = {0: "North", 1: "East", 2: "South", 3: "West"}

# ----------------------------- Data Handlers -----------------------------
class TofDataHandler:
    def __init__(self): self._d=0; self._l=threading.Lock()
    def update(self, info): 
        with self._l: self._d = info[0]
    def reset(self): 
        with self._l: self._d = 0
    def get(self):
        with self._l: return self._d

class VisionDataHandler:
    def __init__(self): self._m=[]; self._l=threading.Lock()
    def update(self, info):
        with self._l: self._m = [i[0] for i in info[1:]] if info and info[0]>0 else []
    def reset(self):
        with self._l: self._m=[]
    def get(self):
        with self._l: return list(self._m)

class PoseDataHandler:
    def __init__(self):
        self.pose = [0.0]*6
        self._lock = threading.Lock()
        self._yaw0 = None

    def update_position(self, pos):
        with self._lock:
            self.pose[0], self.pose[1], self.pose[2] = pos[0], pos[1], pos[2]

    def update_attitude(self, att):
        with self._lock:
            yaw = att[0]
            if self._yaw0 is None:
                self._yaw0 = yaw
            yaw_rel = yaw - self._yaw0
            if yaw_rel > 180: yaw_rel -= 360
            if yaw_rel < -180: yaw_rel += 360
            self.pose[3], self.pose[4], self.pose[5] = yaw_rel, att[1], att[2]

    def reset(self):
        with self._lock:
            self.pose = [0.0] * 6
            self._yaw0 = None  # รีเซ็ตออฟเซ็ต yaw ด้วย

    def get(self):
        with self._lock:
            return tuple(self.pose)

    def set_xy(self, x, y):
        with self._lock:
            self.pose[0], self.pose[1] = float(x), float(y)

    def set_yaw(self, yaw):
        with self._lock:
            self.pose[3] = float(yaw)


# ------------------------------ Logger CSV ------------------------------
class CSVLogger:
    def __init__(self, path="run_log.csv"):
        self.path = path
        self._ensure_header()
        self.log.log(self.pose.get(), dist, [f"{m}:{WALLS.get(d,'?')}"])

    def _ensure_header(self):
        new = not os.path.exists(self.path)
        if new:
            with open(self.path,"w",newline="") as f:
                csv.writer(f).writerow(["ts","x_m","y_m","yaw_deg","tof_mm","markers"])
    def log(self, pose, tof_mm, markers):
        ts = datetime.now().isoformat(timespec="seconds")
        x,y,_,yaw,_,_ = pose
        with open(self.path,"a",newline="") as f:
            csv.writer(f).writerow([ts,f"{x:.3f}",f"{y:.3f}",f"{yaw:.1f}",int(tof_mm), "|".join(markers)])

# ------------------------------ PID ------------------------------------
class PID:
    def __init__(self,Kp,Ki,Kd,setpoint=0.0,limits=(-1.0,1.0)):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd
        self.setpoint=setpoint; self.limits=limits
        self.i=0.0; self.prev_err=0.0; self.t=time.time()
    def update(self, val):
        now=time.time(); dt=max(1e-3, now-self.t)
        e=self.setpoint - val; self.i += e*dt
        d=(e-self.prev_err)/dt
        out=self.Kp*e + self.Ki*self.i + self.Kd*d
        out=max(self.limits[0], min(self.limits[1], out))
        self.prev_err=e; self.t=now
        return out

# --------------------------- Maze Explorer ------------------------------
class MazeExplorer:
    def __init__(self, ep, tof, vis, pose, logger):
        self.r = ep; self.ch=ep.chassis; self.led=ep.led; self.vs=ep.vision; self.gb=ep.gimbal
        self.tof, self.vis, self.pose, self.log = tof, vis, pose, logger
        self.pos=(0,0); self.ori=0
        self.graph={}; self.expl=set(); self.path=[self.pos]; self.marks={}
        # Reset internal states & hardware posture
        self._reset_all_states()

    def _reset_all_states(self):
        # รีเซ็ตค่าทุกอย่างเป็น 0 ในฝั่งซอฟต์แวร์
        self.tof.reset(); self.vis.reset(); self.pose.reset()
        self.pose.set_xy(0,0); self.pose.set_yaw(0)
        # ปรับสภาพหุ่นยนต์
        self.led.set_led(r=0,g=0,b=255)
        self.gb.recenter().wait_for_completed()
        self.r.reset_robot_mode()
        # บันทึกสถานะเริ่มต้น
        self.log.log(self.pose.get(), self.tof.get(), [])

    def add_conn(self,a,b):
        self.graph.setdefault(a,set()).add(b)
        self.graph.setdefault(b,set()).add(a)

    def run(self, limit_s=600):
        t0=time.time(); self.led.set_led(r=0,g=0,b=255)
        while True:
            if time.time()-t0 >= limit_s:
                self.led.set_led(r=255,g=193,b=7,effect="flash"); print("TIME UP"); break
            if self.pos not in self.expl: self.scan()
            nxt=self.next_path()
            if not nxt:
                self.led.set_led(r=0,g=255,b=0); print("DONE"); break
            self.exec_path(nxt)

    def scan(self):
        self.expl.add(self.pos)
        x,y=self.pos
        prev = self.path[-2] if len(self.path)>1 else None
        skip = -1
        if prev:
            self.add_conn(self.pos, prev)
            skip = (self.ori+2)%4
        for d in range(4):
            if d==skip: continue
            nb = (x, y+1) if d==0 else (x+1,y) if d==1 else (x,y-1) if d==2 else (x-1,y)
            if nb in self.expl: continue
            ang=((d-self.ori)*90+540)%360-180
            self.gb.moveto(yaw=ang, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(0.3)
            dist=self.tof.get()
            if dist>=WALL_THRESHOLD_MM: self.add_conn(self.pos, nb)
            time.sleep(VISION_SCAN_DURATION_S)
            ms=self.vis.get()
            if ms and dist<WALL_THRESHOLD_MM:
                for m in ms:
                    self.marks.setdefault(m,[]).append((self.pos, WALLS.get(d,"?")))
            # log ทุกทีก่อนเปลี่ยนมุม
            self.log.log(self.pose.get(), dist, ms)
        self.gb.recenter().wait_for_completed()

    def next_path(self):
        # ไปหาช่องที่ยังไม่สำรวจจากโหนดปัจจุบันก่อน
        for nb in self.graph.get(self.pos, []):
            if nb not in self.expl: return [self.pos, nb]
        # ถ้าไม่มี ให้ย้อนทางที่เคยเดิน หาโหนดที่ยังมีเพื่อนบ้านค้างอยู่
        for p in reversed(self.path):
            for nb in self.graph.get(p, []):
                if nb not in self.expl: return self._bfs(self.pos, p)
        return None

    def _bfs(self, s, g):
        if s==g: return [s]
        q=[(s,[s])]; seen={s}
        while q:
            cur, path = q.pop(0)
            for nb in self.graph.get(cur, []):
                if nb in seen: continue
                if nb==g: return path+[nb]
                seen.add(nb); q.append((nb, path+[nb]))
        return None

    def exec_path(self, path):
        for i in range(len(path)-1):
            a,b=path[i], path[i+1]
            dx,dy=b[0]-a[0], b[1]-a[1]
            tar = 0 if (dx,dy)==(0,1) else 90 if (dx,dy)==(1,0) else 180 if (dx,dy)==(0,-1) else -90
            self.turn_pid(tar); self.ori = (0 if tar==0 else 1 if tar==90 else 2 if tar==180 else 3)
            self.move_forward_pid(GRID_SIZE_M)
            self.pos=b; self.path.append(self.pos)
            self.pose.set_xy(b[0]*GRID_SIZE_M, b[1]*GRID_SIZE_M); self.pose.set_yaw(tar)
            # log หลังเคลื่อนที่ 1 ช่อง
            self.log.log(self.pose.get(), self.tof.get(), self.vis.get())

    def move_forward_pid(self, dist_m, speed_limit=2.5):
        print(f"Move {dist_m} m (PID). Bumper at ≤{BUMP_STOP_MM} mm.")
        pid=PID(2.5,0.1,0.8,setpoint=dist_m,limits=(-speed_limit,speed_limit))
        sx,sy,_,_,_,_=self.pose.get()
        while True:
            # กันชนด้วย ToF
            if self.tof.get() <= BUMP_STOP_MM:
                print("** BUMPER TRIGGERED: stop **")
                self._hard_stop(r=255,g=0,b=0)
                return
            cx,cy,_,_,_,_=self.pose.get()
            d=math.hypot(cx-sx, cy-sy)
            if abs(dist_m-d) < 0.01: break
            vx = pid.update(d)
            self.ch.drive_speed(x=vx, y=0, z=0, timeout=0.1)
            time.sleep(0.01)
        self._hard_stop()

    def turn_pid(self, target_angle, speed_limit=120):
        pid = PID(Kp=1.2, Ki=0.05, Kd=0.5, setpoint=0, limits=(-speed_limit, speed_limit))
        deadband = 1.0
        last_sign = 0
        while True:
            _,_,_,yaw,_,_ = self.pose.get()
            err = target_angle - yaw
            if err > 180: err -= 360
            if err < -180: err += 360
            if abs(err) < deadband:
                break
            sign = 1 if err > 0 else -1
            if sign != last_sign:
                pid.i = 0.0       # anti-windup
                last_sign = sign
            wz = pid.update(err)  # ใช้ err ตรง ๆ
            self.ch.drive_speed(x=0, y=0, z=wz, timeout=0.1)
            time.sleep(0.01)
        self.ch.drive_speed(0,0,0)



    def _hard_stop(self, r=0,g=0,b=255):
        self.ch.drive_speed(0,0,0)
        self.led.set_led(r=r,g=g,b=b)

# ------------------------------ Main -----------------------------------
if __name__ == "__main__":
    ep=None
    try:
        ep=robot.Robot(); ep.initialize(conn_type="ap")
        tof=TofDataHandler(); vis=VisionDataHandler(); pose=PoseDataHandler()
        ep.sensor.sub_distance(freq=10, callback=tof.update)
        ep.vision.sub_detect_info(name="marker", callback=vis.update)
        ep.chassis.sub_position(freq=20, callback=pose.update_position)
        ep.chassis.sub_attitude(freq=20, callback=pose.update_attitude)

        logger = CSVLogger("run_log.csv")
        time.sleep(0.5)  # รอ subscription เสถียรเล็กน้อย
        explorer = MazeExplorer(ep, tof, vis, pose, logger)
        explorer.run(limit_s=600)

    except Exception as e:
        print("Error:", e)
    finally:
        if ep:
            try:
                ep.sensor.unsub_distance()
                ep.vision.unsub_detect_info(name="marker")
                ep.chassis.unsub_position()
                ep.chassis.unsub_attitude()
            except: pass
            ep.close()
        print("break")
