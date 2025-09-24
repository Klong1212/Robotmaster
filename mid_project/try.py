# controllers/bayes_controller/bayes_controller.py
# Bayesian Filter (Histogram/Grid Filter) for 2D localization on Webots
# Belief over a 2D grid, motion from odometry, measurement from GPS (x,z)
from controller import Robot
import numpy as np
import math
import csv
############################
# ปรับตามหุ่นหุ่ ยนต์และโลก
############################
TIME_STEP = 32 # ms
DT = TIME_STEP / 1000.0
WHEEL_RADIUS = 0.0205 # m (e-puck)
WHEEL_BASE = 0.052 # m (ระยะศูนย์กลางล้อ)
# ขอบเขตพื้นที่ที่ต้องการ localize (เมตร)
X_MIN, X_MAX = -2.0, 2.0
Y_MIN, Y_MAX = -2.0, 2.0
RES = 0.05 # ขนาด cell (m/cell)
SIGMA_MEAS = 0.12 # ส่วส่ นเบี่ยงเบน GPS (m) ~ 12 cm
DIFF_ALPHA = 0.05 # ค่ากระจาย belief ต่อ step (0..0.25)
def ang_wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a
class BayesGrid2D:
"""
Grid-based Bayesian filter:
- belief: ความเชื่อความน่าน่ จะเป็น 2 มิติ (normalize เสมอ)
- motion update: roll (shift) ตาม dx, dy เป็นจำ นวน cell แล้วกระจายด้วย diffusion
- measurement update: Gaussian likelihood จาก GPS (x,y)
"""
def __init__(self, x_min, x_max, y_min, y_max, res):
self.res = res
self.xs = np.arange(x_min, x_max + res, res)
self.ys = np.arange(y_min, y_max + res, res)
self.ny = len(self.ys)
self.nx = len(self.xs)
self.X, self.Y = np.meshgrid(self.xs, self.ys) # shape (ny, nx)
self.bel = np.ones((self.ny, self.nx), dtype=np.float64)
self.normalize()
def normalize(self):
s = self.bel.sum()
if s > 0: self.bel /= s
def motion_update(self, dx, dy):
"""
- shift belief ตามการเคลื่อนที่ dx, dy (เมตร)
- แล้ว diffuse แบบ 4-neighborhood เพื่อแทน process noise
"""
# 1) คำ นวณจำ นวน cell ที่ต้อง roll
sx = int(round(dx / self.res))
sy = int(round(dy / self.res))
# roll (กำ หนดแกน 0 คือ y, แกน 1 คือ x)
self.bel = np.roll(self.bel, shift=(sy, sx), axis=(0, 1))
# 2) diffusion เล็กน้อย (discrete Laplacian)
b = self.bel
self.bel = (1 - 4*DIFF_ALPHA) * b \
+ DIFF_ALPHA * (np.roll(b, 1, axis=0) +
np.roll(b, -1, axis=0) +
np.roll(b, 1, axis=1) +
np.roll(b, -1, axis=1))
self.normalize()
def measurement_update(self, x_meas, y_meas, sigma=0.2):
"""
Likelihood model: Gaussian centered at (x_meas, y_meas)
"""
dx2 = (self.X - x_meas) ** 2
dy2 = (self.Y - y_meas) ** 2
L = np.exp(-(dx2 + dy2) / (2 * sigma**2)) + 1e-12
self.bel *= L
self.normalize()
def estimate_mean(self):
"""คำ นวณค่าเฉลี่ย (expected value) ของตำ แหน่งน่ จาก belief"""
px = (self.bel * self.X).sum()
py = (self.bel * self.Y).sum()
return float(px), float(py)
def estimate_map(self):
"""ตำ แหน่งน่ MAP (argmax)"""
iy, ix = np.unravel_index(np.argmax(self.bel), self.bel.shape)
return float(self.xs[ix]), float(self.ys[iy])
def main():
robot = Robot()
# === Devices ===
gps = robot.getDevice("gps"); gps.enable(TIME_STEP)
ml = robot.getDevice("left wheel motor")
mr = robot.getDevice("right wheel motor")
ml.setPosition(float('inf')); mr.setPosition(float('inf'))
ml.setVelocity(0.0); mr.setVelocity(0.0)
sl = ml.getPositionSensor(); sl.enable(TIME_STEP)
sr = mr.getPositionSensor(); sr.enable(TIME_STEP)
# รออุปกรณ์ warm up เล็กน้อย
for _ in range(10):
robot.step(TIME_STEP)
# เริ่มต้นทิศและ odometry
last_l = sl.getValue()
last_r = sr.getValue()
th_odo = 0.0 # ไม่จำม่ จำ เป็นต้องแม่นม่ (เราใช้แค่คำ นวณ dx,dy)
xg, yg, zg = gps.getValues()
# === Bayes filter ===
bf = BayesGrid2D(X_MIN, X_MAX, Y_MIN, Y_MAX, RES)
# ตั้งค่าเริ่มเชื่อ GPS มากหน่อน่ ย: update ครั้งแรก
bf.measurement_update(xg, zg, sigma=SIGMA_MEAS)
# CSV log เพื่อนำ ไป plot ภายหลัง
f = open("bayes_log.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["t","x_gps","y_gps","x_mean","y_mean","x_map","y_map"])
t = 0.0
sim_time = 120.0 # วินาทีสูงสุด
while robot.step(TIME_STEP) != -1 and t < sim_time:
t += DT
# 1) ตัวอย่าย่ งคำ สั่งวิ่งง่ายๆ: ช่วช่ งหนึ่งวิ่งตรง ช่วช่ งหนึ่งเลี้ยว
phase = (t % 10.0)
if phase < 7.0:
v_l = 6.0 # rad/s
v_r = 6.0
else:
v_l = 4.0
v_r = 7.0
ml.setVelocity(v_l); mr.setVelocity(v_r)
# 2) อ่านการวัด
xg, yg, zg = gps.getValues()
# 3) คำ นวณ odometry delta จาก position sensors
cur_l = sl.getValue(); cur_r = sr.getValue()
d_left = WHEEL_RADIUS * (cur_l - last_l)
d_right = WHEEL_RADIUS * (cur_r - last_r)
last_l, last_r = cur_l, cur_r
d_center = 0.5 * (d_left + d_right)
d_theta = (d_right - d_left) / WHEEL_BASE
th_odo = ang_wrap(th_odo + d_theta)
# world-frame delta (ประมาณจาก odometry)
dx = d_center * math.cos(th_odo)
dy = d_center * math.sin(th_odo)
# 4) Bayesian prediction → shift + diffuse
bf.motion_update(dx, dy)
# 5) Bayesian correction → คูณคู likelihood จาก GPS
bf.measurement_update(xg, zg, sigma=SIGMA_MEAS)
# 6) ประมาณค่าจาก belief
mx, my = bf.estimate_mean()
kx, ky = bf.estimate_map()
writer.writerow([t, xg, zg, mx, my, kx, ky])
if int(t*5) % 5 == 0:
print(f"t={t:5.1f} | GPS=({xg: .2f},{zg: .2f}) | MEAN=({mx: .2f},{my: .2f}) | MAP=({kx: .2f},{ky:
.2f})")
f.close()
print("บันทึกผลไว้ที่ bayes_log.csv เรียบร้อย")
if __name__ == "__main__":
main()
# controllers/bayes_controller/bayes_controller.py
# Bayesian Filter (Histogram/Grid Filter) for 2D localization on Webots
# Belief over a 2D grid, motion from odometry, measurement from GPS (x,z)
from controller import Robot
import numpy as np
import math
import csv
############################
# ปรับตามหุ่นหุ่ ยนต์และโลก
############################
TIME_STEP = 32 # ms
DT = TIME_STEP / 1000.0
WHEEL_RADIUS = 0.0205 # m (e-puck)
WHEEL_BASE = 0.052 # m (ระยะศูนย์กลางล้อ)
# ขอบเขตพื้นที่ที่ต้องการ localize (เมตร)
X_MIN, X_MAX = -2.0, 2.0
Y_MIN, Y_MAX = -2.0, 2.0
RES = 0.05 # ขนาด cell (m/cell)
SIGMA_MEAS = 0.12 # ส่วส่ นเบี่ยงเบน GPS (m) ~ 12 cm
DIFF_ALPHA = 0.05 # ค่ากระจาย belief ต่อ step (0..0.25)
def ang_wrap(a):
while a > math.pi: a -= 2*math.pi
while a < -math.pi: a += 2*math.pi
return a
class BayesGrid2D:
"""
Grid-based Bayesian filter:
- belief: ความเชื่อความน่าน่ จะเป็น 2 มิติ (normalize เสมอ)
- motion update: roll (shift) ตาม dx, dy เป็นจำ นวน cell แล้วกระจายด้วย diffusion
- measurement update: Gaussian likelihood จาก GPS (x,y)
"""
def __init__(self, x_min, x_max, y_min, y_max, res):
self.res = res
self.xs = np.arange(x_min, x_max + res, res)
self.ys = np.arange(y_min, y_max + res, res)
self.ny = len(self.ys)
self.nx = len(self.xs)
self.X, self.Y = np.meshgrid(self.xs, self.ys) # shape (ny, nx)
self.bel = np.ones((self.ny, self.nx), dtype=np.float64)
self.normalize()
def normalize(self):
s = self.bel.sum()
if s > 0: self.bel /= s
def motion_update(self, dx, dy):
"""
- shift belief ตามการเคลื่อนที่ dx, dy (เมตร)
- แล้ว diffuse แบบ 4-neighborhood เพื่อแทน process noise
"""
# 1) คำ นวณจำ นวน cell ที่ต้อง roll
sx = int(round(dx / self.res))
sy = int(round(dy / self.res))
# roll (กำ หนดแกน 0 คือ y, แกน 1 คือ x)
self.bel = np.roll(self.bel, shift=(sy, sx), axis=(0, 1))
# 2) diffusion เล็กน้อย (discrete Laplacian)
b = self.bel
self.bel = (1 - 4*DIFF_ALPHA) * b \
+ DIFF_ALPHA * (np.roll(b, 1, axis=0) +
np.roll(b, -1, axis=0) +
np.roll(b, 1, axis=1) +
np.roll(b, -1, axis=1))
self.normalize()
def measurement_update(self, x_meas, y_meas, sigma=0.2):
"""
Likelihood model: Gaussian centered at (x_meas, y_meas)
"""
dx2 = (self.X - x_meas) ** 2
dy2 = (self.Y - y_meas) ** 2
L = np.exp(-(dx2 + dy2) / (2 * sigma**2)) + 1e-12
self.bel *= L
self.normalize()
def estimate_mean(self):
"""คำ นวณค่าเฉลี่ย (expected value) ของตำ แหน่งน่ จาก belief"""
px = (self.bel * self.X).sum()
py = (self.bel * self.Y).sum()
return float(px), float(py)
def estimate_map(self):
"""ตำ แหน่งน่ MAP (argmax)"""
iy, ix = np.unravel_index(np.argmax(self.bel), self.bel.shape)
return float(self.xs[ix]), float(self.ys[iy])
def main():
robot = Robot()
# === Devices ===
gps = robot.getDevice("gps"); gps.enable(TIME_STEP)
ml = robot.getDevice("left wheel motor")
mr = robot.getDevice("right wheel motor")
ml.setPosition(float('inf')); mr.setPosition(float('inf'))
ml.setVelocity(0.0); mr.setVelocity(0.0)
sl = ml.getPositionSensor(); sl.enable(TIME_STEP)
sr = mr.getPositionSensor(); sr.enable(TIME_STEP)
# รออุปกรณ์ warm up เล็กน้อย
for _ in range(10):
robot.step(TIME_STEP)
# เริ่มต้นทิศและ odometry
last_l = sl.getValue()
last_r = sr.getValue()
th_odo = 0.0 # ไม่จำม่ จำ เป็นต้องแม่นม่ (เราใช้แค่คำ นวณ dx,dy)
xg, yg, zg = gps.getValues()
# === Bayes filter ===
bf = BayesGrid2D(X_MIN, X_MAX, Y_MIN, Y_MAX, RES)
# ตั้งค่าเริ่มเชื่อ GPS มากหน่อน่ ย: update ครั้งแรก
bf.measurement_update(xg, zg, sigma=SIGMA_MEAS)
# CSV log เพื่อนำ ไป plot ภายหลัง
f = open("bayes_log.csv", "w", newline="")
writer = csv.writer(f)
writer.writerow(["t","x_gps","y_gps","x_mean","y_mean","x_map","y_map"])
t = 0.0
sim_time = 120.0 # วินาทีสูงสุด
while robot.step(TIME_STEP) != -1 and t < sim_time:
t += DT
# 1) ตัวอย่าย่ งคำ สั่งวิ่งง่ายๆ: ช่วช่ งหนึ่งวิ่งตรง ช่วช่ งหนึ่งเลี้ยว
phase = (t % 10.0)
if phase < 7.0:
v_l = 6.0 # rad/s
v_r = 6.0
else:
v_l = 4.0
v_r = 7.0
ml.setVelocity(v_l); mr.setVelocity(v_r)
# 2) อ่านการวัด
xg, yg, zg = gps.getValues()
# 3) คำ นวณ odometry delta จาก position sensors
cur_l = sl.getValue(); cur_r = sr.getValue()
d_left = WHEEL_RADIUS * (cur_l - last_l)
d_right = WHEEL_RADIUS * (cur_r - last_r)
last_l, last_r = cur_l, cur_r
d_center = 0.5 * (d_left + d_right)
d_theta = (d_right - d_left) / WHEEL_BASE
th_odo = ang_wrap(th_odo + d_theta)
# world-frame delta (ประมาณจาก odometry)
dx = d_center * math.cos(th_odo)
dy = d_center * math.sin(th_odo)
# 4) Bayesian prediction → shift + diffuse
bf.motion_update(dx, dy)
# 5) Bayesian correction → คูณคู likelihood จาก GPS
bf.measurement_update(xg, zg, sigma=SIGMA_MEAS)
# 6) ประมาณค่าจาก belief
mx, my = bf.estimate_mean()
kx, ky = bf.estimate_map()
writer.writerow([t, xg, zg, mx, my, kx, ky])
if int(t*5) % 5 == 0:
print(f"t={t:5.1f} | GPS=({xg: .2f},{zg: .2f}) | MEAN=({mx: .2f},{my: .2f}) | MAP=({kx: .2f},{ky:
.2f})")
f.close()
print("บันทึกผลไว้ที่ bayes_log.csv เรียบร้อย")
if __name__ == "__main__":
main()
