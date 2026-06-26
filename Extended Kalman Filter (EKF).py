"""
[ชุดที่ 2] x-IMU Real-time Motion Tracking - Extended Kalman Filter (EKF)
- ใช้สมการไม่เป็นเส้นตรง ร่วมกับการคำนวณหา Jacobian Matrix (Fj) แบบเรียลไทม์
- State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (10 มิติเต็มรูปแบบ)
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
CSV_FILENAME = "motion_tracking_extended_kf.csv"

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
# 🧠 EXTENDED KALMAN FILTER (EKF) CLASS
# ==========================================
class IMUExtendedKalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        # State vector (10 มิติ): [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
        self.x = np.zeros(10)
        self.x[6] = 1.0  # ควอเทอร์เนียนเริ่มต้น W=1
        
        # State Covariance Matrix (P) ขนาด 10x10
        self.P = np.eye(10) * 0.1
        self.P[6:10, 6:10] *= 0.01  
        
        # Process Noise Covariance (Q) ขนาด 10x10
        self.Q = np.eye(10) * 1e-4
        self.Q[0:3, 0:3] *= 0.1     
        self.Q[3:6, 3:6] *= 1.0     
        self.Q[6:10, 6:10] *= 0.1   
        
        # Measurement Matrix สำหรับ ZUPT (3x10) ตรวจวัดเฉพาะความเร็ว
        self.H = np.zeros((3, 10))
        self.H[:, 3:6] = np.eye(3)
        
        # Measurement Noise Covariance (R)
        self.R = np.eye(3) * 1e-4

    def _quaternion_to_R(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    def predict(self, acc_raw, gyro_raw):
        """ ขั้นตอน Nonlinear Prediction และคำนวณ Jacobian Matrix (Fj) ไดนามิก """
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        
        # 1. หมุนความเร่งพิกัด Body -> Earth (Nonlinear Step)
        R_matrix = self._quaternion_to_R(q)
        acc_earth = R_matrix @ acc_raw
        
        # 2. อัปเดตสถานะถัดไปด้วยสมการ Nonlinear Kinematics
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
        
        # 3. 🛠️ ดึงค่าตัวแปรออกมารองรับ dR_dq เพื่อทำ Linearization (คำนวณ Fj 10x10)
        w, x, y, z = q
        ax, ay, az = acc_raw
        
        Fj = np.eye(10)
        Fj[0:3, 3:6] = np.eye(3) * self.dt  # dp/dv
        
        #คำนวณเมทริกซ์อนุพันธ์ย่อย (Jacobian) ของการหมุนเทียบกับ Quaternion
        dR_dq = 2 * np.array([
            [  y*ay + z*az,  y*ax - 2*x*ay - w*az,  x*ax + w*az - 2*y*az, -w*ay + x*az],
            [ -z*ax + w*az,  y*ay + w*az,           x*ay - 2*y*ax + z*az, -w*ax + z*ay],
            [  y*ax - w*ay,  z*ax - w*az,          -w*ax + z*ay,          x*az + y*ay]
        ])
        
        Fj[0:3, 6:10] = 0.5 * (self.dt**2) * dR_dq  # dp/dq
        Fj[3:6, 6:10] = self.dt * dR_dq             # dv/dq
        
        # อนุพันธ์ย่อยของ Quaternion เทียบตัวมันเองและ Gyro
        Fj[6:10, 6:10] += 0.5 * self.dt * np.array([
            [0,   -gx,  -gy,  -gz],
            [gx,   0,    gz,  -gy],
            [gy,  -gz,   0,    gx],
            [gz,   gy,  -gx,   0]
        ])
        
        # 4. อัปเดต Error Covariance ล่วงหน้า
        self.P = Fj @ self.P @ Fj.T + self.Q

    def update_zupt(self):
        """ ขั้นตอน Measurement Update ของ EKF """
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

ekf = IMUExtendedKalmanFilter(DT)

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
    print(f"📝 EKF CSV Initialized: {CSV_FILENAME}")

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
    print(f"UDP EKF Processor Active on Port {UDP_PORT}...")
    
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
                    print("\n✓ EXTENDED KALMAN FILTER CALIBRATION COMPLETE\n")
                continue
            
            with data_lock:
                accel = accel_raw - accel_bias
                gyro = np.radians(gyro_raw - gyro_bias)
                
                accel[0] *= SCALE_X; accel[1] *= SCALE_Y; accel[2] *= SCALE_Z
                
                acc_reshaped = accel.reshape(1, 3)
                acc_filtered_block, zi_accel = signal.sosfilt(sos, acc_reshaped, zi=zi_accel, axis=0)
                acc_input = acc_filtered_block[0]
                
                acc_mag = np.linalg.norm(acc_input)
                gyro_mag = np.linalg.norm(gyro)
                sample_stat = (acc_mag < STATIONARY_ACCEL_THRESHOLD) and (gyro_mag < np.radians(STATIONARY_GYRO_THRESHOLD))
                
                stat_window.append(sample_stat)
                if len(stat_window) > ZUPT_WINDOW: stat_window.pop(0)
                is_stationary = len(stat_window) == ZUPT_WINDOW and all(stat_window)
                
                # --- 🧠 EXTENDED KALMAN FILTER STEP ---
                ekf.predict(acc_input, gyro)
                if is_stationary:
                    ekf.update_zupt()
                
                position = ekf.x[0:3]
                velocity = ekf.x[3:6]
                current_quat = ekf.x[6:10]
                
                path_x.append(position[0]); path_y.append(position[1]); path_z.append(position[2])
                if len(path_x) > int(20 * SAMPLE_RATE):
                    path_x.pop(0); path_y.pop(0); path_z.pop(0)
                
                append_to_csv(current_time, position, current_quat, "STATIONARY" if is_stationary else "MOVING")
                    
        except Exception as e: print(f"EKF Error: {e}")

# VISUALIZATION
fig = plt.figure(figsize=(14, 10))
ax_3d = fig.add_subplot(221, projection='3d')
ax_top = fig.add_subplot(222); ax_side = fig.add_subplot(223); ax_front = fig.add_subplot(224)

line_3d, = ax_3d.plot([], [], [], 'b-', linewidth=2, label='EKF Live Path')
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
    ax_3d.set_title('3D Live Trajectory (Extended Kalman Filter)')
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
        
        motion_state = "STATIONARY (EKF ZUPT Lock)" if is_stationary else "🔴 MOVING"
        fig.suptitle(f"x-IMU EKF | Status: {motion_state} | Pos: {position} m", fontsize=13, fontweight='bold', color="green" if is_stationary else "red")
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
            ekf.reset()
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