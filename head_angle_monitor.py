"""
═══════════════════════════════════════════════════════════════════
  head_angle_monitor.py
  Đo góc cúi đầu lâm sàng — 2 MPU6050 (T1-T3 + Chỏm đầu)
  
  Pipeline:
    Serial (ESP32) → Parse → Madgwick + EKF (mỗi sensor) →
    Góc tương đối → Real-time plot + Lưu CSV

  Cài đặt:
    pip install pyserial numpy pandas matplotlib scipy

  Chạy:
    python head_angle_monitor.py
═══════════════════════════════════════════════════════════════════
"""

import serial
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from collections import deque
from datetime import datetime
import threading
import queue
import os
import sys

# ══════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════
SERIAL_PORT    = "COM5"       # Đổi theo máy (Linux: "/dev/ttyUSB0")
BAUD_RATE      = 115200
SAMPLE_RATE_HZ = 100

CALIB_SECONDS  = 3            # Thời gian calibrate (giây)
PLOT_WINDOW_S  = 15           # Cửa sổ hiển thị real-time (giây)

OUTPUT_DIR     = "recordings" # Thư mục lưu CSV

# ── Ngưỡng lâm sàng tham chiếu (độ) ─────────────────────────
# Nguồn: APTA clinical guidelines cho ROM cổ
CLINICAL_REF = {
    "pitch": {"normal_min": -50, "normal_max":  50,
              "label_fwd": "Cúi trước ~50°", "label_back": "Ngửa sau ~50°"},
    "roll":  {"normal_min": -45, "normal_max":  45,
              "label_left": "Nghiêng trái ~45°", "label_right": "Nghiêng phải ~45°"},
}

# ══════════════════════════════════════════════════════════════
# BỘ LỌC: ADAPTIVE MADGWICK
# ══════════════════════════════════════════════════════════════
class AdaptiveMadgwick:
    """
    Madgwick AHRS với beta thích nghi theo nhiễu gia tốc.
    Dùng cho MPU6050 (6-DOF, không có magnetometer).
    Yaw sẽ drift — không sử dụng cho lâm sàng.
    """
    def __init__(self, freq=100.0, beta_base=0.033,
                 beta_min=0.005, beta_max=0.12,
                 accel_threshold=0.15):
        self.dt        = 1.0 / freq
        self.beta_base = beta_base
        self.beta_min  = beta_min
        self.beta_max  = beta_max
        self.accel_thr = accel_threshold   # Tuned cho chuyển động đầu (nhẹ hơn tay)
        self.q         = np.array([1.0, 0.0, 0.0, 0.0])
        self.beta      = beta_base

    def _adapt_beta(self, ax, ay, az):
        a_mag = math.sqrt(ax*ax + ay*ay + az*az)
        dev   = abs(a_mag - 1.0)
        # Khi chuyển động tịnh tiến (dev cao) → tin gyro hơn → beta nhỏ
        alpha       = min(dev / self.accel_thr, 1.0)
        self.beta   = self.beta_min + (1.0 - alpha) * (self.beta_max - self.beta_min)
        return self.beta

    def update(self, gx, gy, gz, ax, ay, az):
        beta = self._adapt_beta(ax, ay, az)
        q0, q1, q2, q3 = self.q

        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm < 1e-6:
            return self.q
        ax, ay, az = ax/norm, ay/norm, az/norm

        _2q0 = 2*q0; _2q1 = 2*q1; _2q2 = 2*q2; _2q3 = 2*q3
        q0q0 = q0*q0; q1q1 = q1*q1; q2q2 = q2*q2; q3q3 = q3*q3

        s0 = (4*q0*q2q2 + _2q2*ax + 4*q0*q1q1 - _2q1*ay)
        s1 = (4*q1*q3q3 - _2q3*ax + 4*q0q0*q1 - _2q0*ay
              - 4*q1 + 8*q1*q1q1 + 8*q1*q2q2 + 4*q1*az)
        s2 = (4*q0q0*q2 + _2q0*ax + 4*q2*q3q3 - _2q3*ay
              - 4*q2 + 8*q2*q1q1 + 8*q2*q2q2 + 4*q2*az)
        s3 = (4*q1q1*q3 - _2q1*ax + 4*q2q2*q3 - _2q2*ay)

        sn = math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        if sn < 1e-6:
            sn = 1.0
        s0 /= sn; s1 /= sn; s2 /= sn; s3 /= sn

        qd0 = 0.5*(-q1*gx - q2*gy - q3*gz) - beta*s0
        qd1 = 0.5*( q0*gx + q2*gz - q3*gy) - beta*s1
        qd2 = 0.5*( q0*gy - q1*gz + q3*gx) - beta*s2
        qd3 = 0.5*( q0*gz + q1*gy - q2*gx) - beta*s3

        q0 += qd0*self.dt; q1 += qd1*self.dt
        q2 += qd2*self.dt; q3 += qd3*self.dt

        qn = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        self.q = np.array([q0/qn, q1/qn, q2/qn, q3/qn])
        return self.q

    def get_pitch_roll(self):
        q0, q1, q2, q3 = self.q
        pitch = math.asin(max(-1.0, min(1.0, 2*(q0*q2 - q3*q1)))) * 180/math.pi
        roll  = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2)) * 180/math.pi
        return pitch, roll

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])


# ══════════════════════════════════════════════════════════════
# BỘ LỌC: EKF 2D (khử bias gyro)
# ══════════════════════════════════════════════════════════════
class EKF2D:
    """
    Extended Kalman Filter 2D: state = [angle, gyro_bias]
    Nhận đầu vào: raw gyro (°/s) + góc đo từ Madgwick (°)
    """
    def __init__(self, dt,
                 sigma_q_angle=0.003,    # Tuned cho MPU6050
                 sigma_q_bias=0.0003,
                 sigma_r=0.03):
        self.dt = dt
        self.x  = np.zeros(2)            # [angle, bias]
        self.P  = np.eye(2) * 0.1
        self.Q  = np.diag([sigma_q_angle**2, sigma_q_bias**2])
        self.R  = np.array([[sigma_r**2]])
        self.F  = np.array([[1.0, -dt], [0.0, 1.0]])
        self.H  = np.array([[1.0, 0.0]])

    def predict(self, gyro_deg_s):
        ang, bias = self.x
        self.x[0] = ang + (gyro_deg_s - bias) * self.dt
        self.P    = self.F @ self.P @ self.F.T + self.Q

    def update(self, angle_meas):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0, 0]
        y = angle_meas - self.x[0]
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - np.outer(K.flatten(), self.H)) @ self.P
        return self.x[0]

    def reset(self):
        self.x = np.zeros(2)
        self.P = np.eye(2) * 0.1


# ══════════════════════════════════════════════════════════════
# PIPELINE MỘT SENSOR (Madgwick + EKF, pitch và roll)
# ══════════════════════════════════════════════════════════════
class SensorPipeline:
    def __init__(self, name, freq=100.0):
        self.name     = name
        self.dt       = 1.0 / freq
        self.madgwick = AdaptiveMadgwick(freq=freq)
        self.ekf_p    = EKF2D(dt=self.dt)   # EKF cho pitch
        self.ekf_r    = EKF2D(dt=self.dt)   # EKF cho roll
        self.pitch    = 0.0
        self.roll     = 0.0

    def process(self, gx, gy, gz, ax, ay, az):
        """
        gx,gy,gz: rad/s  |  ax,ay,az: g
        Trả về (pitch_deg, roll_deg)
        """
        self.madgwick.update(gx, gy, gz, ax, ay, az)
        pitch_m, roll_m = self.madgwick.get_pitch_roll()

        # EKF dùng gyro theo trục tương ứng (chuyển rad/s → °/s)
        gy_dps = gy * (180.0 / math.pi)
        gx_dps = gx * (180.0 / math.pi)

        self.ekf_p.predict(gy_dps)
        self.ekf_r.predict(gx_dps)

        self.pitch = self.ekf_p.update(pitch_m)
        self.roll  = self.ekf_r.update(roll_m)
        return self.pitch, self.roll

    def reset(self):
        self.madgwick.reset()
        self.ekf_p.reset()
        self.ekf_r.reset()
        self.pitch = 0.0
        self.roll  = 0.0


# ══════════════════════════════════════════════════════════════
# SERIAL READER (chạy trên thread riêng)
# ══════════════════════════════════════════════════════════════
class SerialReader(threading.Thread):
    def __init__(self, port, baud, data_queue):
        super().__init__(daemon=True)
        self.port    = port
        self.baud    = baud
        self.queue   = data_queue
        self.running = False
        self.ser     = None
        self.ready   = threading.Event()
        self.error   = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"[Serial] Đã kết nối {self.port}")
        except Exception as e:
            self.error = str(e)
            self.ready.set()
            return

        self.running = True
        # Chờ dòng READY từ ESP32
        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("READY:"):
                    print(f"[Serial] ESP32 sẵn sàng: {line}")
                    self.ready.set()
                    break
                elif line.startswith("ERROR:"):
                    self.error = line
                    self.ready.set()
                    return
            except Exception:
                pass

        # Đọc dữ liệu
        while self.running:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("HEADER") or line.startswith("ERROR"):
                    continue
                parts = line.split(",")
                if len(parts) == 15:
                    vals = list(map(float, parts))
                    self.queue.put(vals)
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


# ══════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════
def calibrate(pipeline_t13, pipeline_head, data_queue, calib_s=3):
    """
    Lấy N mẫu đầu tiên khi bệnh nhân đứng thẳng để lấy offset.
    Trả về (offset_pitch_t13, offset_roll_t13,
             offset_pitch_head, offset_roll_head)
    """
    n         = int(calib_s * SAMPLE_RATE_HZ)
    samples   = []
    print(f"\n[Calibrate] Yêu cầu bệnh nhân đứng/ngồi thẳng, giữ yên {calib_s}s...")

    for i in range(n):
        # Chờ có dữ liệu
        while data_queue.empty():
            time.sleep(0.001)
        vals = data_queue.get()
        # vals: [t_ms, ax1,ay1,az1, gx1,gy1,gz1, ax2,ay2,az2, gx2,gy2,gz2, temp1, temp2]
        ax1,ay1,az1 = vals[1], vals[2], vals[3]
        gx1,gy1,gz1 = vals[4], vals[5], vals[6]
        ax2,ay2,az2 = vals[7], vals[8], vals[9]
        gx2,gy2,gz2 = vals[10],vals[11],vals[12]

        p1, r1 = pipeline_t13.process(gx1,gy1,gz1, ax1,ay1,az1)
        p2, r2 = pipeline_head.process(gx2,gy2,gz2, ax2,ay2,az2)
        samples.append((p1, r1, p2, r2))

        pct = int((i+1)/n * 30)
        print(f"\r  [{'█'*pct}{'░'*(30-pct)}] {i+1}/{n}", end="", flush=True)

    arr = np.array(samples)
    offsets = arr.mean(axis=0)
    print(f"\n[Calibrate] Xong! Offset → Pitch_T13={offsets[0]:.2f}°, "
          f"Roll_T13={offsets[1]:.2f}°, "
          f"Pitch_Head={offsets[2]:.2f}°, "
          f"Roll_Head={offsets[3]:.2f}°")

    # Reset filter sau calibrate để bắt đầu sạch
    pipeline_t13.reset()
    pipeline_head.reset()
    return tuple(offsets)


# ══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(OUTPUT_DIR, f"head_angle_{session_id}.csv")

    # ── Khởi tạo pipeline ──────────────────────────────────────
    pipeline_t13  = SensorPipeline("T1-T3",       freq=SAMPLE_RATE_HZ)
    pipeline_head = SensorPipeline("Chom_dau",    freq=SAMPLE_RATE_HZ)

    # ── Khởi tạo serial reader ─────────────────────────────────
    data_queue = queue.Queue(maxsize=500)
    reader     = SerialReader(SERIAL_PORT, BAUD_RATE, data_queue)
    reader.start()

    print("[*] Chờ ESP32 khởi động...")
    reader.ready.wait(timeout=10)

    if reader.error:
        print(f"[LỖI] {reader.error}")
        print("Gợi ý: Kiểm tra cổng COM, tắt Arduino Serial Monitor nếu đang mở.")
        return

    # ── Calibrate ──────────────────────────────────────────────
    off_p1, off_r1, off_p2, off_r2 = calibrate(
        pipeline_t13, pipeline_head, data_queue, calib_s=CALIB_SECONDS
    )

    # ── Buffer real-time plot ──────────────────────────────────
    N_plot = PLOT_WINDOW_S * SAMPLE_RATE_HZ
    t_buf  = deque(maxlen=N_plot)
    p_buf  = deque(maxlen=N_plot)   # Relative pitch
    r_buf  = deque(maxlen=N_plot)   # Relative roll

    # ── Lưu CSV ────────────────────────────────────────────────
    csv_rows = []
    csv_header = [
        "timestamp_s",
        "pitch_t13_deg", "roll_t13_deg",
        "pitch_head_deg","roll_head_deg",
        "rel_pitch_deg", "rel_roll_deg",
        "temp1_C","temp2_C"
    ]
    t0 = time.time()

    # ── Setup plot ─────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, (ax_pitch, ax_roll) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(
        f"ĐO GÓC CÚI ĐẦU LÂM SÀNG — Session: {session_id}",
        fontsize=13, fontweight="bold", color="#e0e0e0"
    )

    # Pitch plot
    ax_pitch.set_ylabel("Góc Pitch (°)\n[+ = Cúi trước]", color="#e0e0e0")
    ax_pitch.set_ylim(-70, 70)
    ax_pitch.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax_pitch.axhspan(CLINICAL_REF["pitch"]["normal_min"],
                     CLINICAL_REF["pitch"]["normal_max"],
                     alpha=0.08, color="#00c896", label="Vùng bình thường (±50°)")
    ax_pitch.axhline(CLINICAL_REF["pitch"]["normal_max"],
                     color="#00c896", linewidth=0.6, linestyle=":")
    ax_pitch.axhline(CLINICAL_REF["pitch"]["normal_min"],
                     color="#00c896", linewidth=0.6, linestyle=":")
    line_pitch, = ax_pitch.plot([], [], color="#4fc3f7", linewidth=1.8,
                                label="Góc cúi trước/sau")
    ax_pitch.legend(loc="upper right", fontsize=8)
    ax_pitch.grid(True, alpha=0.15)

    # Roll plot
    ax_roll.set_ylabel("Góc Roll (°)\n[+ = Nghiêng phải]", color="#e0e0e0")
    ax_roll.set_xlabel("Thời gian (s)", color="#e0e0e0")
    ax_roll.set_ylim(-60, 60)
    ax_roll.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax_roll.axhspan(CLINICAL_REF["roll"]["normal_min"],
                    CLINICAL_REF["roll"]["normal_max"],
                    alpha=0.08, color="#ffb74d", label="Vùng bình thường (±45°)")
    ax_roll.axhline(CLINICAL_REF["roll"]["normal_max"],
                    color="#ffb74d", linewidth=0.6, linestyle=":")
    ax_roll.axhline(CLINICAL_REF["roll"]["normal_min"],
                    color="#ffb74d", linewidth=0.6, linestyle=":")
    line_roll,  = ax_roll.plot([], [], color="#ff8a65", linewidth=1.8,
                               label="Góc nghiêng ngang")
    ax_roll.legend(loc="upper right", fontsize=8)
    ax_roll.grid(True, alpha=0.15)

    # Textbox hiển thị góc hiện tại
    txt_current = fig.text(0.01, 0.01,
                           "Pitch: --°  |  Roll: --°",
                           fontsize=10, color="#e0e0e0",
                           bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))
    txt_status  = fig.text(0.99, 0.01, "● LIVE",
                           fontsize=9, color="#00e676", ha="right",
                           bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    # ── Animation update ───────────────────────────────────────
    def update_plot(frame):
        # Xử lý tất cả mẫu có trong queue
        processed = 0
        while not data_queue.empty() and processed < 20:
            vals = data_queue.get()
            processed += 1

            t_ms = vals[0]
            ax1,ay1,az1 = vals[1], vals[2], vals[3]
            gx1,gy1,gz1 = vals[4], vals[5], vals[6]
            ax2,ay2,az2 = vals[7], vals[8], vals[9]
            gx2,gy2,gz2 = vals[10],vals[11],vals[12]
            temp1, temp2 = vals[13], vals[14]

            p1, r1 = pipeline_t13.process(gx1,gy1,gz1, ax1,ay1,az1)
            p2, r2 = pipeline_head.process(gx2,gy2,gz2, ax2,ay2,az2)

            # Góc tương đối = sensor đầu − sensor cột sống
            rel_p = (p2 - off_p2) - (p1 - off_p1)
            rel_r = (r2 - off_r2) - (r1 - off_r1)

            t_now = time.time() - t0
            t_buf.append(t_now)
            p_buf.append(rel_p)
            r_buf.append(rel_r)

            # Lưu vào CSV buffer
            csv_rows.append({
                "timestamp_s":   round(t_now, 4),
                "pitch_t13_deg": round(p1, 3),
                "roll_t13_deg":  round(r1, 3),
                "pitch_head_deg":round(p2, 3),
                "roll_head_deg": round(r2, 3),
                "rel_pitch_deg": round(rel_p, 3),
                "rel_roll_deg":  round(rel_r, 3),
                "temp1_C":       round(temp1, 2),
                "temp2_C":       round(temp2, 2),
            })

        if not t_buf:
            return line_pitch, line_roll

        t_arr = np.array(t_buf)
        p_arr = np.array(p_buf)
        r_arr = np.array(r_buf)

        # Cập nhật đường vẽ
        line_pitch.set_data(t_arr, p_arr)
        line_roll.set_data(t_arr, r_arr)

        # Cập nhật trục X theo thời gian thực
        t_min = max(0, t_arr[-1] - PLOT_WINDOW_S)
        ax_pitch.set_xlim(t_min, t_min + PLOT_WINDOW_S)
        ax_roll.set_xlim(t_min, t_min + PLOT_WINDOW_S)

        # Cập nhật text góc hiện tại
        cur_p = p_arr[-1]; cur_r = r_arr[-1]
        p_warn = "⚠ NGOÀI NGƯỠNG" if abs(cur_p) > 50 else ""
        r_warn = "⚠ NGOÀI NGƯỠNG" if abs(cur_r) > 45 else ""
        txt_current.set_text(
            f"Pitch: {cur_p:+.1f}°  {p_warn}   |   Roll: {cur_r:+.1f}°  {r_warn}"
        )

        return line_pitch, line_roll

    # ── Chạy animation ─────────────────────────────────────────
    ani = FuncAnimation(fig, update_plot, interval=50,   # 20 FPS
                        blit=False, cache_frame_data=False)

    print(f"\n[*] Đang hiển thị real-time. Đóng cửa sổ đồ thị để lưu file và thoát.")
    print(f"[*] File CSV sẽ được lưu vào: {csv_filename}")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    # ── Dừng đọc và lưu CSV ────────────────────────────────────
    reader.stop()

    if csv_rows:
        df = pd.DataFrame(csv_rows, columns=csv_header)
        df.to_csv(csv_filename, index=False)
        print(f"\n[*] Đã lưu {len(df)} mẫu → {csv_filename}")
        print(f"    Thời gian đo: {df['timestamp_s'].iloc[-1]:.1f}s")
        print(f"    Pitch max: {df['rel_pitch_deg'].max():.1f}°  "
              f"min: {df['rel_pitch_deg'].min():.1f}°")
        print(f"    Roll  max: {df['rel_roll_deg'].max():.1f}°  "
              f"min: {df['rel_roll_deg'].min():.1f}°")
    else:
        print("\n[!] Không có dữ liệu để lưu.")


if __name__ == "__main__":
    main()
