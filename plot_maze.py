# Plot maze from run_log.csv
import os, math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

GRID_SIZE_M = 0.6

def _round_to_grid(v, grid=GRID_SIZE_M):
    return int(round(v / grid))

def load_csv(path="run_log.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # รองรับทั้งรูปแบบเก่า/ใหม่
    needed_min = {"x_m","y_m","yaw_deg","tof_mm","markers"}
    if not needed_min.issubset(df.columns):
        raise ValueError(f"CSV missing columns: need at least {needed_min}, got {set(df.columns)}")
    # cast
    for c in ["x_m","y_m","yaw_deg","tof_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "event" not in df.columns:
        df["event"] = ""
    if "scan_dir" not in df.columns:
        df["scan_dir"] = ""
    if "wall_state" not in df.columns:
        df["wall_state"] = ""
    df["markers"] = df["markers"].fillna("").astype(str)
    df = df.dropna(subset=["x_m","y_m"]).reset_index(drop=True)
    return df

def reconstruct_graph(df):
    # สร้างเส้นทางจากการเคลื่อนที่จริงตามเวลาบันทึก (event=='move') หากไม่มี ใช้ทั้งหมด
    if (df["event"] == "move").any():
        move_df = df[df["event"]=="move"].copy()
    else:
        move_df = df.copy()

    nodes = []
    for _,row in move_df.iterrows():
        gx = _round_to_grid(row["x_m"])
        gy = _round_to_grid(row["y_m"])
        nodes.append((gx,gy))

    # ลบซ้ำติดกัน
    path_nodes = [nodes[0]] if nodes else []
    for n in nodes[1:]:
        if n != path_nodes[-1]:
            path_nodes.append(n)

    # สร้างเส้นเชื่อมถ้าห่างแมนฮัตตัน 1 ช่อง
    edges = []
    for a,b in zip(path_nodes, path_nodes[1:]):
        if abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1:
            edges.append((a,b))

    return path_nodes, edges

def collect_markers(df):
    marker_map = defaultdict(Counter)
    for _,row in df.iterrows():
        items = [m.strip() for m in str(row["markers"]).split("|") if m.strip()]
        if items:
            cell = (_round_to_grid(row["x_m"]), _round_to_grid(row["y_m"]))
            for m in items:
                marker_map[cell][m] += 1
    return marker_map

def collect_walls(df):
    """
    ถ้าใช้ logger รูปแบบใหม่ จะมี event=='scan' + scan_dir + wall_state
    เราจะสร้างกำแพง (เส้นหนา) บนขอบ cell ที่ถูกบอกว่า blocked
    """
    wall_segments = []  # list of [(x1,y1),(x2,y2)]
    side_vec = {
        "North": (0, +1),
        "East":  (+1, 0),
        "South": (0, -1),
        "West":  (-1, 0)
    }
    scan_df = df[(df["event"]=="scan") & (df["wall_state"]!="") & (df["scan_dir"]!="")]
    for _,row in scan_df.iterrows():
        cell = (_round_to_grid(row["x_m"]), _round_to_grid(row["y_m"]))
        if str(row["wall_state"]).lower() != "blocked":
            continue
        face = str(row["scan_dir"])
        if face not in side_vec:
            continue
        # สร้าง segment ความยาว 1 ช่องบนขอบ cell ด้านนั้น (ในพิกัดเมตร)
        i,j = cell
        x0, y0 = i*GRID_SIZE_M, j*GRID_SIZE_M
        if face=="North":
            p1=(x0-0.5*GRID_SIZE_M, y0+0.5*GRID_SIZE_M)
            p2=(x0+0.5*GRID_SIZE_M, y0+0.5*GRID_SIZE_M)
        elif face=="South":
            p1=(x0-0.5*GRID_SIZE_M, y0-0.5*GRID_SIZE_M)
            p2=(x0+0.5*GRID_SIZE_M, y0-0.5*GRID_SIZE_M)
        elif face=="East":
            p1=(x0+0.5*GRID_SIZE_M, y0-0.5*GRID_SIZE_M)
            p2=(x0+0.5*GRID_SIZE_M, y0+0.5*GRID_SIZE_M)
        elif face=="West":
            p1=(x0-0.5*GRID_SIZE_M, y0-0.5*GRID_SIZE_M)
            p2=(x0-0.5*GRID_SIZE_M, y0+0.5*GRID_SIZE_M)
        wall_segments.append((p1,p2))
    return wall_segments

def plot_maze(df, title="Maze reconstruction"):
    path_nodes, edges = reconstruct_graph(df)
    marker_map = collect_markers(df)
    wall_segs = collect_walls(df)

    plt.figure(figsize=(9,9))

    # วาดทางเดินจาก edges
    for (a,b) in edges:
        ax, ay = a[0]*GRID_SIZE_M, a[1]*GRID_SIZE_M
        bx, by = b[0]*GRID_SIZE_M, b[1]*GRID_SIZE_M
        plt.plot([ax,bx], [ay,by], linewidth=2)

    # วาดเส้นทาง dashed สำหรับบริบท
    if path_nodes:
        xs = [n[0]*GRID_SIZE_M for n in path_nodes]
        ys = [n[1]*GRID_SIZE_M for n in path_nodes]
        plt.plot(xs, ys, linestyle="--", linewidth=1)

    # วาดโหนด
    if path_nodes:
        xs = [n[0]*GRID_SIZE_M for n in path_nodes]
        ys = [n[1]*GRID_SIZE_M for n in path_nodes]
        plt.scatter(xs, ys, s=25)

    # จุดเริ่ม/จบ
    if path_nodes:
        sx, sy = path_nodes[0][0]*GRID_SIZE_M, path_nodes[0][1]*GRID_SIZE_M
        ex, ey = path_nodes[-1][0]*GRID_SIZE_M, path_nodes[-1][1]*GRID_SIZE_M
        plt.sca
