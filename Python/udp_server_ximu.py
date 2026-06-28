"""
[โค้ดรวม] x-IMU Real-time Motion Tracking - Live KF vs EKF Comparison
- รัน Standard Kalman Filter (KF) และ Extended Kalman Filter (EKF) ขนานพร้อมกัน
- แสดงผลพล็อตเปรียบเทียบแบบเรียลไทม์:
  * เส้นสีฟ้า = Standard Linear KF
  * เส้นสีส้ม = Extended Kalman Filter (EKF)
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
UDP_IP = "192.168.1.64"  
UDP_PORT = 5000
BUFFER_SIZE = 1024
CSV_FILENAME = "kf_vs_ekf_comparison_log.csv"

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
# 🧠 1. STANDARD LINEAR KALMAN FILTER CLASS
# ==========================================
class StandardKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(6) # [px, py, pz, vx, vy, vz]
        
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt
        
        self.B = np.zeros((6, 3))
        self.B[0:3, 0:3] = 0.5 * (dt**2) * np.eye(3)
        self.B[3:6, 0:3] = dt * np.eye(3)
        
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.01
        self.Q[0:3, 0:3] *= 0.1  
        
        self.H = np.zeros((3, 6))
        self.H[:, 3:6] = np.eye(3)
        self.R = np.eye(3) * 0.001

    def predict(self, acc):
        self.x = self.F @ self.x + self.B @ acc
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_zupt(self):
        z = np.zeros(3)  
        y = z - self.H @ self.x  
        S = self.H @ self.P @ self.H.T + self.R  
        K = self.P @ self.H.T @ np.linalg.inv(S)  
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.1

# ==========================================
# 🧠 2. EXTENDED KALMAN FILTER (EKF) CLASS
# ==========================================
class IMUExtendedKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(10) # [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
        self.x[6] = 1.0  
        
        self.P = np.eye(10) * 0.1
        self.P[6:10, 6:10] *= 0.01  
        
        self.Q = np.eye(10) * 1e-4
        self.Q[0:3, 0:3] *= 0.1     
        self.Q[3:6, 3:6] *= 1.0     
        self.Q[6:10, 6:10] *= 0.1   
        
        self.H = np.zeros((3, 10))
        self.H[:, 3:6] = np.eye(3)
        self.R = np.eye(3) * 1e-4

    def _quaternion_to_R(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def predict(self, acc_raw, gyro_raw):
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        
        R_matrix = self._quaternion_to_R(q)
        acc_earth = R_matrix @ acc_raw
        
        p_next = p + v * self.dt + 0.5 * acc_earth * (self.dt**2)
        v_next = v + acc_earth * self.dt
        
        qw, qx, qy, qz = q
        gx, gy, gz = gyro_raw
        dq = 0.5 * np.array([
            -qx*gx - qy*gy - qz*gz,
             qw*gx + qy*gz - qz*gy,
             qw*gy - qx*gz + qz*gx,
             qw*gz + qx*gy - qy*gx
        ]) * self.dt
        q_next = q + dq
        q_next /= np.linalg.norm(q_next) 
        
        self.x[0:3] = p_next
        self.x[3:6] = v_next
        self.x[6:10] = q_next
        
        w, x, y, z = q
        ax, ay, az = acc_raw
        
        Fj = np.eye(10)
        Fj[0:3, 3:6] = np.eye(3) * self.dt  
        
        dR_dq = 2 * np.array([
            [  y*ay + z*az,  y*ax - 2*x*ay - w*az,  x*ax + w*az - 2*y*az, -w*ay + x*az],
            [ -z*ax + w*az,  y*ay + w*az,           x*ay - 2*y*ax + z*az, -w*ax + z*ay],
            [  y*ax - w*ay,  z*ax - w*az,          -w*ax + z*ay,          x*az + y*ay]
        ])
        
        Fj[0:3, 6:10] = 0.5 * (self.dt**2) * dR_dq  
        Fj[3:6, 6:10] = self.dt * dR_dq             
        
        Fj[6:10, 6:10] += 0.5 * self.dt * np.array([
            [0,   -gx,  -gy,  -gz],
            [gx,   0,    gz,  -gy],
            [gy,  -gz,   0,    gx],
            [gz,   gy,  -gx,   0]
        ])
        self.P = Fj @ self.P @ Fj.T + self.Q

    def update_zupt(self):
        z = np.zeros(3)  
        y_residual = z - self.H @ self.x  
        S = self.H @ self.P @ self.H.T + self.R  
        K = self.P @ self.H.T @ np.linalg.inv(S)  
        self.x = self.x + K @ y_residual
        self.P = (np.eye(10) - K @ self.H) @ self.P
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

    def reset(self):
        self.x = np.zeros(10)
        self.x[6] = 1.0
        self.P = np.eye(10) * 0.1

# สร้าง Object ตัวกรองทั้ง 2 รูปแบบ
kf = StandardKalmanFilter(DT)
ekf = IMUExtendedKalmanFilter(DT)

# ==========================================
# GLOBAL STATE & LIVE BUFFERS
# ==========================================
is_calibrating = True
calibration_counter = 0
calibration_buffer_accel, calibration_buffer_gyro = [], []
gyro_bias, accel_bias = np.zeros(3), np.zeros(3)

# แยกบัฟเฟอร์เก็บตำแหน่งของแต่ละตัวกรองสำหรับนำไปพล็อต
path_kf_x, path_kf_y, path_kf_z = [0.0], [0.0], [0.0]
path_ekf_x, path_ekf_y, path_ekf_z = [0.0], [0.0], [0.0]

pos_kf, pos_ekf = np.zeros(3), np.zeros(3)
current_quat_ekf = np.array([1.0, 0.0, 0.0, 0.0])
is_stationary = True
stat_window = []

sos = signal.butter(HP_FILTER_ORDER, HP_FILTER_CUTOFF / (0.5 * SAMPLE_RATE), 'highpass', output='sos')
zi_accel_kf = np.zeros((sos.shape[0], 2, 3))  
zi_accel_ekf = np.zeros((sos.shape[0], 2, 3))  
data_lock = Lock()

def initialize_csv():
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "KF_X", "KF_Y", "KF_Z", "EKF_X", "EKF_Y", "EKF_Z", "Status"])
    print(f"📝 Dual Log CSV Initialized: {CSV_FILENAME}")

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

# ==========================================
# REAL-TIME DUAL PROCESSING ENGINE
# ==========================================
def udp_server_processor():
    global is_calibrating, calibration_counter, gyro_bias, accel_bias
    global pos_kf, pos_ekf, current_quat_ekf, is_stationary, stat_window
    global zi_accel_kf, zi_accel_ekf, path_kf_x, path_kf_y, path_kf_z, path_ekf_x, path_ekf_y, path_ekf_z
    
    initialize_csv()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Dual KF vs EKF Real-time Processor Running on Port {UDP_PORT}...")
    
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
                    print("\n✓ DUAL FILTERS CALIBRATION COMPLETE\n")
                continue
            
            with data_lock:
                accel = accel_raw - accel_bias
                gyro = np.radians(gyro_raw - gyro_bias)
                
                # --- [รันฝั่ง Standard KF] ---
                acc_earth_kf = quaternion_rotate(accel, quat_raw)
                acc_earth_kf[0] = -acc_earth_kf[0]; acc_earth_kf[1] = -acc_earth_kf[1]
                acc_kf_reshaped = acc_earth_kf.reshape(1, 3)
                acc_kf_filtered, zi_accel_kf = signal.sosfilt(sos, acc_kf_reshaped, zi=zi_accel_kf, axis=0)
                acc_input_kf = acc_kf_filtered[0]
                acc_input_kf[0] *= SCALE_X; acc_input_kf[1] *= SCALE_Y; acc_input_kf[2] *= SCALE_Z
                
                # --- [รันฝั่ง EKF] ---
                accel_ekf = accel.copy()
                accel_ekf[0] *= SCALE_X; accel_ekf[1] *= SCALE_Y; accel_ekf[2] *= SCALE_Z
                acc_ekf_reshaped = accel_ekf.reshape(1, 3)
                acc_ekf_filtered, zi_accel_ekf = signal.sosfilt(sos, acc_ekf_reshaped, zi=zi_accel_ekf, axis=0)
                acc_input_ekf = acc_ekf_filtered[0]
                
                # ZUPT Condition เช็คจากสัญญาณดิบที่ปรับสเกลแล้ว
                acc_mag = np.linalg.norm(acc_input_ekf)
                gyro_mag = np.linalg.norm(gyro)
                sample_stat = (acc_mag < STATIONARY_ACCEL_THRESHOLD) and (gyro_mag < np.radians(STATIONARY_GYRO_THRESHOLD))
                
                stat_window.append(sample_stat)
                if len(stat_window) > ZUPT_WINDOW: stat_window.pop(0)
                is_stationary = len(stat_window) == ZUPT_WINDOW and all(stat_window)
                
                # ทำนายและอัปเดตทั้งคู่
                kf.predict(acc_input_kf)
                ekf.predict(acc_input_ekf, gyro)
                
                if is_stationary:
                    kf.update_zupt()
                    ekf.update_zupt()
                
                pos_kf = kf.x[0:3]
                pos_ekf = ekf.x[0:3]
                current_quat_ekf = ekf.x[6:10]
                
                # บันทึกประวัติลงอาเรย์พล็อต
                path_kf_x.append(pos_kf[0]); path_kf_y.append(pos_kf[1]); path_kf_z.append(pos_kf[2])
                path_ekf_x.append(pos_ekf[0]); path_ekf_y.append(pos_ekf[1]); path_ekf_z.append(pos_ekf[2])
                
                max_len = int(20 * SAMPLE_RATE)
                if len(path_kf_x) > max_len:
                    path_kf_x.pop(0); path_kf_y.pop(0); path_kf_z.pop(0)
                    path_ekf_x.pop(0); path_ekf_y.pop(0); path_ekf_z.pop(0)
                
                # เขียนลง CSV
                with open(CSV_FILENAME, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f"{current_time:.4f}", f"{pos_kf[0]:.4f}", f"{pos_kf[1]:.4f}", f"{pos_kf[2]:.4f}", f"{pos_ekf[0]:.4f}", f"{pos_ekf[1]:.4f}", f"{pos_ekf[2]:.4f}", "STATIONARY" if is_stationary else "MOVING"])
                    
        except Exception as e: print(f"Processor Error: {e}")

# ==========================================
# VISUALIZATION (พล็อตเทียบ 2 ตัวกรองในจอเดียว)
# ==========================================
fig = plt.figure(figsize=(14, 10))
ax_3d = fig.add_subplot(221, projection='3d')
ax_top = fig.add_subplot(222); ax_side = fig.add_subplot(223); ax_front = fig.add_subplot(224)

# สร้างเส้นพล็อตของ Standard KF (สีฟ้า) และ EKF (สีส้ม/แดง)
line_kf_3d, = ax_3d.plot([], [], [], 'b-', linewidth=1.5, label='Standard KF (Linear)')
line_ekf_3d, = ax_3d.plot([], [], [], 'r-', linewidth=2.5, label='Extended KF (EKF)')
scat_kf_3d = ax_3d.scatter([], [], [], c='blue', s=60)
scat_ekf_3d = ax_3d.scatter([], [], [], c='red', s=80)

line_kf_top, = ax_top.plot([], [], 'b--', linewidth=1.5, label='Standard KF')
line_ekf_top, = ax_top.plot([], [], 'r-', linewidth=2, label='EKF')
scat_kf_top = ax_top.scatter([], [], c='blue', s=50)
scat_ekf_top = ax_top.scatter([], [], c='red', s=70)

line_kf_side, = ax_side.plot([], [], 'b--', linewidth=1.5)
line_ekf_side, = ax_side.plot([], [], 'r-', linewidth=2)
scat_kf_side = ax_side.scatter([], [], c='blue', s=50)
scat_ekf_side = ax_side.scatter([], [], c='red', s=70)

line_front_kf, = ax_front.plot([], [], 'b--', linewidth=1.5)
line_front_ekf, = ax_front.plot([], [], 'r-', linewidth=2)
scat_kf_front = ax_front.scatter([], [], c='blue', s=50)
scat_ekf_front = ax_front.scatter([], [], c='red', s=70)

# จุดกำเนิด (Origin)
ax_3d.scatter([0], [0], [0], c='green', s=150, marker='*', label='Origin')
ax_top.scatter([0], [0], c='green', s=150, marker='*')

def init_plot():
    ax_3d.set_title('3D Path Comparison: KF (Blue) vs EKF (Red)')
    ax_top.set_title('Top View (XY)'); ax_top.grid(True); ax_top.set_aspect('equal'); ax_top.legend(loc='lower left')
    ax_side.set_title('Side View (XZ)'); ax_side.grid(True); ax_side.set_aspect('equal')
    ax_front.set_title('Front View (YZ)'); ax_front.grid(True); ax_front.set_aspect('equal')
    ax_3d.legend(loc='upper right')
    return [line_kf_3d, line_ekf_3d]

def update_plot(frame):
    with data_lock:
        if is_calibrating:
            fig.suptitle(f"🔄 CALIBRATING... ({calibration_counter}/{CALIBRATION_SAMPLES})", color='orange', fontsize=14, fontweight='bold')
            return [line_kf_3d, line_ekf_3d]
        
        motion_state = "STATIONARY (ZUPT Active)" if is_stationary else "🔴 MOVING"
        fig.suptitle(f"Live Comparison | Status: {motion_state}\nKF Pos: {pos_kf} | EKF Pos: {pos_ekf}", fontsize=11, fontweight='bold')
        
        kf_x, kf_y, kf_z = np.array(path_kf_x), np.array(path_kf_y), np.array(path_kf_z)
        ekf_x, ekf_y, ekf_z = np.array(path_ekf_x), np.array(path_ekf_y), np.array(path_ekf_z)

    if len(kf_x) > 1:
        # อัปเดตกราฟ 3D
        line_kf_3d.set_data(kf_x, kf_y); line_kf_3d.set_3d_properties(kf_z)
        line_ekf_3d.set_data(ekf_x, ekf_y); line_ekf_3d.set_3d_properties(ekf_z)
        scat_kf_3d._offsets3d = ([kf_x[-1]], [kf_y[-1]], [kf_z[-1]])
        scat_ekf_3d._offsets3d = ([ekf_x[-1]], [ekf_y[-1]], [ekf_z[-1]])
        
        # อัปเดตระนาบ 2D ต่างๆ
        line_kf_top.set_data(kf_x, kf_y); scat_kf_top.set_offsets([[kf_x[-1], kf_y[-1]]])
        line_ekf_top.set_data(ekf_x, ekf_y); scat_ekf_top.set_offsets([[ekf_x[-1], ekf_y[-1]]])
        
        line_kf_side.set_data(kf_x, kf_z); scat_kf_side.set_offsets([[kf_x[-1], kf_z[-1]]])
        line_ekf_side.set_data(ekf_x, ekf_z); scat_ekf_side.set_offsets([[ekf_x[-1], ekf_z[-1]]])
        
        line_front_kf.set_data(kf_y, kf_z); scat_kf_front.set_offsets([[kf_y[-1], kf_z[-1]]])
        line_front_ekf.set_data(ekf_y, ekf_z); scat_ekf_front.set_offsets([[ekf_y[-1], ekf_z[-1]]])
        
        # ปรับสเกลหน้าจออัตโนมัติอิงตามขอบเขตที่กว้างที่สุดของทั้งสองฝั่ง
        margin = 0.2
        x_min = min(kf_x.min(), ekf_x.min()) - margin
        x_max = max(kf_x.max(), ekf_x.max()) + margin
        y_min = min(kf_y.min(), ekf_y.min()) - margin
        y_max = max(kf_y.max(), ekf_y.max()) + margin
        z_min = min(kf_z.min(), ekf_z.min()) - margin
        z_max = max(kf_z.max(), ekf_z.max()) + margin
        
        ax_3d.set_xlim(x_min, x_max); ax_3d.set_ylim(y_min, y_max); ax_3d.set_zlim(z_min, z_max)
        ax_top.set_xlim(x_min, x_max); ax_top.set_ylim(y_min, y_max)
        ax_side.set_xlim(x_min, x_max); ax_side.set_ylim(z_min, z_max)
        ax_front.set_xlim(y_min, y_max); ax_front.set_ylim(z_min, z_max)
        
    return [line_kf_3d, line_ekf_3d]

def on_key(event):
    global path_kf_x, path_kf_y, path_kf_z, path_ekf_x, path_ekf_y, path_ekf_z, zi_accel_kf, zi_accel_ekf
    if event.key in ['c', 'C']:
        with data_lock:
            kf.reset()
            ekf.reset()
            path_kf_x, path_kf_y, path_kf_z = [0.0], [0.0], [0.0]
            path_ekf_x, path_ekf_y, path_ekf_z = [0.0], [0.0], [0.0]
            zi_accel_kf = np.zeros_like(zi_accel_kf)
            zi_accel_ekf = np.zeros_like(zi_accel_ekf)
            initialize_csv()
            print("\n🗑 Dual Filters Reset Successfully!\n")

fig.canvas.mpl_connect('key_press_event', on_key)
p_thread = Thread(target=udp_server_processor, daemon=True)
p_thread.start()
ani = FuncAnimation(fig, update_plot, init_func=init_plot, interval=33, cache_frame_data=False)
plt.tight_layout()
plt.show()