"""
[ชุดที่ 1] x-IMU Real-time Motion Tracking - Standard Kalman Filter (KF)
- ใช้สมการเส้นตรง (Linear State Space) 
- State: [px, py, pz, vx, vy, vz] (6 มิติ)
"""

import socket
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread, Lock
import time
import csv
from scipy import signal

# ==========================================
# CONFIGURATION
# ==========================================
UDP_IP = "10.220.116.247"  
UDP_PORT = 5000
BUFFER_SIZE = 1024
CSV_FILENAME = "motion_tracking_linear_kf.csv"

SAMPLE_RATE = 200.0      
DT = 1.0 / SAMPLE_RATE

SCALE_X, SCALE_Y, SCALE_Z = 1.0, 1.0, 1.0
HP_FILTER_CUTOFF = 0.1
HP_FILTER_ORDER = 1
CALIBRATION_SAMPLES = int(2.0 * SAMPLE_RATE) 

STATIONARY_ACCEL_THRESHOLD = 0.22  
STATIONARY_GYRO_THRESHOLD = 3.5    
ZUPT_WINDOW = max(4, int(0.08 * SAMPLE_RATE)) 

# ==========================================
# 🧠 STANDARD LINEAR KALMAN FILTER CLASS
# ==========================================
class StandardKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        # State vector: [px, py, pz, vx, vy, vz]
        self.x = np.zeros(6)
        
        # State Transition Matrix (F) - เส้นตรงคงที่
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt
        
        # Control Input Matrix (B) - เส้นตรงคงที่
        self.B = np.zeros((6, 3))
        self.B[0:3, 0:3] = 0.5 * (dt**2) * np.eye(3)
        self.B[3:6, 0:3] = dt * np.eye(3)
        
        # Covariance Matrix (P)
        self.P = np.eye(6) * 0.1
        
        # Process Noise Covariance (Q)
        self.Q = np.eye(6) * 0.01
        self.Q[0:3, 0:3] *= 0.1  
        
        # Measurement Matrix (H) - ล็อกเฉพาะความเร็วตอนทำ ZUPT
        self.H = np.zeros((3, 6))
        self.H[:, 3:6] = np.eye(3)
        
        # Measurement Noise Covariance (R)
        self.R = np.eye(3) * 0.001

    def predict(self, acc):
        """ ทำนายล่วงหน้าด้วย Linear Model """
        self.x = self.F @ self.x + self.B @ acc
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_zupt(self):
        """ อัปเดตเมื่ออยู่นิ่งด้วย Linear Kalman Gain """
        z = np.zeros(3)  
        y = z - self.H @ self.x  
        S = self.H @ self.P @ self.H.T + self.R  
        K = self.P @ self.H.T @ np.linalg.inv(S)  
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1

kf = StandardKalmanFilter(DT)

# GLOBAL BUFFERS & VARIABLES
is_calibrating = True
calibration_counter = 0
calibration_buffer_accel, calibration_buffer_gyro = [], []
gyro_bias, accel_bias = np.zeros(3), np.zeros(3)
position, velocity, current_quat = np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
path_x, path_y, path_z = [0.0], [0.0], [0.0]
stat_window = []
is_stationary = True

sos = signal.butter(HP_FILTER_ORDER, HP_FILTER_CUTOFF / (0.5 * SAMPLE_RATE), 'highpass', output='sos')
zi_accel = np.zeros((sos.shape[0], 2, 3))  
data_lock = Lock()

def initialize_csv():
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Pos_X(m)", "Pos_Y(m)", "Pos_Z(m)", "Quat_W", "Quat_X", "Quat_Y", "Quat_Z", "Status"])
    print(f"📝 Linear KF CSV Initialized: {CSV_FILENAME}")

def append_to_csv(ts, pos, q, status_str):
    try:
        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"{ts:.4f}", f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}", f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}", status_str])
    except Exception as e: print(f"CSV Error: {e}")

def parse_packet(line):
    pattern = re.compile(r"Accel\[(.*?)\] Gyro\[(.*?)\] Quat\[(.*?)\]")
    match = pattern.search(line)
    if not match: return None
    accel = np.array([float(x) for x in match.group(1).split(',')], dtype=float)
    gyro = np.array([float(x) for x in match.group(2).split(',')], dtype=float)
    quat = np.array([float(x) for x in match.group(3).split(',')], dtype=float)
    return accel, gyro, quat

def quaternion_rotate(v, q):
    w, x, y, z = q
    qv = np.array([0, v[0], v[1], v[2]])
    q_conj = np.array([w, -x, -y, -z])
    
    def q_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    return q_mul(q_mul(q, qv), q_conj)[1:]

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y**2 + z**2)],
        [2*(x*z - w*y), 2*(y**2 + w*x), 1 - 2*(x**2 + y**2)]
    ])

def udp_server_processor():
    global is_calibrating, calibration_counter, gyro_bias, accel_bias
    global position, velocity, current_quat, stat_window, is_stationary, zi_accel, path_x, path_y, path_z
    
    initialize_csv()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"UDP Linear KF Processor Active on Port {UDP_PORT}...")
    
    while True:
        try:
            data, _ = sock.recvfrom(BUFFER_SIZE)
            line = data.decode('utf-8', errors='ignore').strip()
            parsed = parse_packet(line)
            if parsed is None: continue
            
            accel_raw, gyro_raw, quat_raw = parsed
            current_time = time.time()
            
            if is_calibrating:
                calibration_buffer_accel.append(accel_raw)
                calibration_buffer_gyro.append(gyro_raw)
                calibration_counter += 1
                if calibration_counter >= CALIBRATION_SAMPLES:
                    gyro_bias = np.mean(calibration_buffer_gyro, axis=0)
                    accel_bias = np.mean(calibration_buffer_accel, axis=0)
                    is_calibrating = False
                    print("\n✓ LINEAR KF CALIBRATION COMPLETE\n")
                continue
            
            with data_lock:
                accel = accel_raw - accel_bias
                gyro = np.radians(gyro_raw - gyro_bias)
                current_quat = quat_raw
                
                acc_earth = quaternion_rotate(accel, current_quat)
                acc_earth[0] = -acc_earth[0]; acc_earth[1] = -acc_earth[1]
                
                acc_earth_reshaped = acc_earth.reshape(1, 3)
                acc_earth_filtered, zi_accel = signal.sosfilt(sos, acc_earth_reshaped, zi=zi_accel, axis=0)
                acc_filtered = acc_earth_filtered[0]
                
                acc_filtered[0] *= SCALE_X; acc_filtered[1] *= SCALE_Y; acc_filtered[2] *= SCALE_Z
                
                acc_mag = np.linalg.norm(acc_filtered)
                gyro_mag = np.linalg.norm(gyro)
                sample_stat = (acc_mag < STATIONARY_ACCEL_THRESHOLD) and (gyro_mag < np.radians(STATIONARY_GYRO_THRESHOLD))
                
                stat_window.append(sample_stat)
                if len(stat_window) > ZUPT_WINDOW: stat_window.pop(0)
                is_stationary = len(stat_window) == ZUPT_WINDOW and all(stat_window)
                
                # --- 🧠 LINEAR KF STEP ---
                kf.predict(acc_filtered)
                if is_stationary:
                    kf.update_zupt()
                
                position = kf.x[0:3]
                velocity = kf.x[3:6]
                
                path_x.append(position[0]); path_y.append(position[1]); path_z.append(position[2])
                if len(path_x) > int(20 * SAMPLE_RATE):
                    path_x.pop(0); path_y.pop(0); path_z.pop(0)
                
                append_to_csv(current_time, position, current_quat, "STATIONARY" if is_stationary else "MOVING")
                    
        except Exception as e: print(f"Processing Error: {e}")

# VISUALIZATION
fig = plt.figure(figsize=(14, 10))
ax_3d = fig.add_subplot(221, projection='3d')
ax_top = fig.add_subplot(222); ax_side = fig.add_subplot(223); ax_front = fig.add_subplot(224)

line_3d, = ax_3d.plot([], [], [], 'b-', linewidth=2, label='Linear KF Path')
scat_3d = ax_3d.scatter([], [], [], c='red', s=100)
quiver_x = ax_3d.quiver([],[],[],[],[],[], color='red')
quiver_y = ax_3d.quiver([],[],[],[],[],[], color='green')
quiver_z = ax_3d.quiver([],[],[],[],[],[], color='blue')

line_top, = ax_top.plot([], [], 'b-', linewidth=2); scat_top = ax_top.scatter([], [], c='red', s=100)
line_side, = ax_side.plot([], [], 'b-', linewidth=2); scat_side = ax_side.scatter([], [], c='red', s=100)
line_front, = ax_front.plot([], [], 'b-', linewidth=2); scat_front = ax_front.scatter([], [], c='red', s=100)

ax_3d.scatter([0], [0], [0], c='green', s=150, marker='*', label='Origin')
ax_top.scatter([0], [0], c='green', s=150, marker='*')
ax_side.scatter([0], [0], c='green', s=150, marker='*')
ax_front.scatter([0], [0], c='green', s=150, marker='*')

def init_plot():
    ax_3d.set_title('3D Trajectory (Standard Linear Kalman Filter)')
    ax_top.set_title('Top View (XY)'); ax_top.grid(True); ax_top.set_aspect('equal')
    ax_side.set_title('Side View (XZ)'); ax_side.grid(True); ax_side.set_aspect('equal')
    ax_front.set_title('Front View (YZ)'); ax_front.grid(True); ax_front.set_aspect('equal')
    return [line_3d, scat_3d]

def update_plot(frame):
    global quiver_x, quiver_y, quiver_z
    with data_lock:
        if is_calibrating:
            fig.suptitle(f"🔄 CALIBRATING... ({calibration_counter}/{CALIBRATION_SAMPLES})", color='orange', fontsize=14, fontweight='bold')
            return [line_3d, scat_3d]
        
        motion_state = "STATIONARY (KF Lock)" if is_stationary else "🔴 MOVING"
        fig.suptitle(f"x-IMU Linear KF | Status: {motion_state} | Pos: {position} m", fontsize=13, fontweight='bold', color="green" if is_stationary else "red")
        xs, ys, zs = np.array(path_x), np.array(path_y), np.array(path_z)
        q_now = current_quat.copy()
    
    if len(xs) > 1:
        line_3d.set_data(xs, ys); line_3d.set_3d_properties(zs)
        scat_3d._offsets3d = ([xs[-1]], [ys[-1]], [zs[-1]])
        
        R = quaternion_to_rotation_matrix(q_now)
        arr_len = 0.15
        quiver_x.remove(); quiver_y.remove(); quiver_z.remove()
        quiver_x = ax_3d.quiver(xs[-1], ys[-1], zs[-1], R[0,0]*arr_len, R[1,0]*arr_len, R[2,0]*arr_len, color='red', linewidth=2)
        quiver_y = ax_3d.quiver(xs[-1], ys[-1], zs[-1], R[0,1]*arr_len, R[1,1]*arr_len, R[2,1]*arr_len, color='green', linewidth=2)
        quiver_z = ax_3d.quiver(xs[-1], ys[-1], zs[-1], R[0,2]*arr_len, R[1,2]*arr_len, R[2,2]*arr_len, color='blue', linewidth=2)
        
        line_top.set_data(xs, ys); scat_top.set_offsets([[xs[-1], ys[-1]]])
        line_side.set_data(xs, zs); scat_side.set_offsets([[xs[-1], zs[-1]]])
        line_front.set_data(ys, zs); scat_front.set_offsets([[ys[-1], zs[-1]]])
        
        margin = 0.2
        ax_3d.set_xlim(xs.min()-margin, xs.max()+margin); ax_3d.set_ylim(ys.min()-margin, ys.max()+margin); ax_3d.set_zlim(zs.min()-margin, zs.max()+margin)
        ax_top.set_xlim(xs.min()-margin, xs.max()+margin); ax_top.set_ylim(ys.min()-margin, ys.max()+margin)
        ax_side.set_xlim(xs.min()-margin, xs.max()+margin); ax_side.set_ylim(zs.min()-margin, zs.max()+margin)
        ax_front.set_xlim(ys.min()-margin, ys.max()+margin); ax_front.set_ylim(zs.min()-margin, zs.max()+margin)
    return [line_3d, scat_3d]

def on_key(event):
    global position, velocity, path_x, path_y, path_z, zi_accel
    if event.key in ['c', 'C']:
        with data_lock:
            kf.reset()
            position, velocity = np.zeros(3), np.zeros(3)
            path_x, path_y, path_z = [0.0], [0.0], [0.0]
            zi_accel = np.zeros_like(zi_accel)
            initialize_csv()

fig.canvas.mpl_connect('key_press_event', on_key)
p_thread = Thread(target=udp_server_processor, daemon=True)
p_thread.start()
ani = FuncAnimation(fig, update_plot, init_func=init_plot, interval=33, cache_frame_data=False)
plt.tight_layout()
plt.show()