"""
head_angle_monitor_v3.py
Đo góc cúi đầu lâm sàng — 2 MPU6050 (T1-T3 + Chỏm đầu)

v3 Fix so với v2:
  [F1] Auto-detect COM port — không cần hardcode
  [F2] Serial reconnect tự động khi mất kết nối
  [F3] CRC8 validation — drop packet corrupt
  [F4] dt thực tế từ micros() thay vì hardcode 1/100
  [F5] Warmup hội tụ adaptive thay vì 2 vòng cố định
  [F6] Bỏ temperature compensation không đáng tin — chỉ dùng bias static
  [F7] Queue overflow tracking — log dropped packets
  [F8] Graceful shutdown: lưu CSV ngay cả khi crash

Cài: pip install pyserial numpy pandas matplotlib
"""

import serial
import serial.tools.list_ports
import time, math, os, threading, queue, signal, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from collections import deque
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════
BAUD_RATE       = 921600
SAMPLE_RATE_HZ  = 100
CALIB_SECONDS   = 5          # Tăng lên 5s để bias ổn định hơn
PLOT_WINDOW_S   = 15
OUTPUT_DIR      = "recordings"

# Ngưỡng lâm sàng (CROM — Cervical Range of Motion)
CLINICAL_PITCH_MAX =  50.0   # Flexion
CLINICAL_PITCH_MIN = -50.0   # Extension
CLINICAL_ROLL_MAX  =  45.0   # Lateral flexion phải
CLINICAL_ROLL_MIN  = -45.0   # Lateral flexion trái

# Adaptive Madgwick parameters — tuned cho chuyển động đầu người
BETA_MIN   = 0.006   # Tin gyro nhiều hơn khi đang chuyển động
BETA_MAX   = 0.08    # Kéo về accel khi đứng yên
ACCEL_THR  = 0.08    # g — ngưỡng detect chuyển động


# ══════════════════════════════════════════════════════════════
# CRC8 VALIDATION
# ══════════════════════════════════════════════════════════════
def crc8(data: str) -> int:
    """CRC8 polynomial 0x07 — khớp với ESP32."""
    crc = 0x00
    for b in data.encode("ascii"):
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
        crc &= 0xFF
    return crc


def validate_and_parse(line: str):
    """
    Parse 1 dòng CSV. Trả về list[float] 15 phần tử nếu hợp lệ, None nếu không.
    Format: t_us,ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2,t1,t2,crc8
    """
    parts = line.split(",")
    if len(parts) != 16:
        return None
    try:
        payload = ",".join(parts[:15])
        expected_crc = int(parts[15])
        actual_crc   = crc8(payload)
        if expected_crc != actual_crc:
            return None
        return list(map(float, parts[:15]))
    except (ValueError, IndexError):
        return None


# ══════════════════════════════════════════════════════════════
# [F1] AUTO-DETECT COM PORT
# ══════════════════════════════════════════════════════════════
ESP32_VID_PID = [
    (0x10C4, 0xEA60),   # Silicon Labs CP2102 (phổ biến nhất trên board ESP32)
    (0x1A86, 0x7523),   # CH340 (board clone rẻ)
    (0x1A86, 0x55D4),   # CH9102
    (0x0403, 0x6001),   # FTDI FT232RL
    (0x239A, 0x8089),   # Adafruit ESP32-S2
]

def find_esp32_port() -> str | None:
    """Tự động tìm ESP32 theo VID:PID hoặc description string."""
    ports = serial.tools.list_ports.comports()
    # Ưu tiên match VID:PID chính xác
    for p in ports:
        if p.vid and p.pid:
            if (p.vid, p.pid) in ESP32_VID_PID:
                print(f"[Port] Tìm thấy ESP32 tại {p.device} ({p.description})")
                return p.device
    # Fallback: match theo description string
    keywords = ["cp210", "ch340", "ch9102", "ftdi", "esp32", "usb serial", "usb-serial"]
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in keywords):
            print(f"[Port] Tìm thấy (fallback) tại {p.device} ({p.description})")
            return p.device
    return None


# ══════════════════════════════════════════════════════════════
# ADAPTIVE MADGWICK — 6-DOF
# ══════════════════════════════════════════════════════════════
class AdaptiveMadgwick:
    """
    Madgwick 6-DOF với beta thích nghi theo mức độ acceleration noise.

    Khi |accel| ≈ 1g → đứng yên → beta cao → tin accel, kéo về gravity.
    Khi |accel| lệch khỏi 1g → đang chuyển động → beta thấp → tin gyro.
    """
    def __init__(self, freq=100.0, beta_min=BETA_MIN, beta_max=BETA_MAX, accel_thr=ACCEL_THR):
        self.freq   = freq
        self.bmin   = beta_min
        self.bmax   = beta_max
        self.athr   = accel_thr
        self.q      = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.beta   = beta_max

    def _adapt_beta(self, ax, ay, az):
        dev       = abs(math.sqrt(ax*ax + ay*ay + az*az) - 1.0)
        alpha     = min(dev / self.athr, 1.0)
        self.beta = self.bmin + (1.0 - alpha) * (self.bmax - self.bmin)

    def update(self, gx, gy, gz, ax, ay, az, dt=None):
        """
        [F4] dt thực tế thay vì hardcode.
        dt = None → dùng 1/freq mặc định.
        """
        if dt is None:
            dt = 1.0 / self.freq

        self._adapt_beta(ax, ay, az)
        beta = self.beta
        q0, q1, q2, q3 = self.q

        n = math.sqrt(ax*ax + ay*ay + az*az)
        if n < 1e-6:
            # Accel quá yếu — không thể dùng làm reference, chỉ integrate gyro
            q0 += 0.5 * (-q1*gx - q2*gy - q3*gz) * dt
            q1 += 0.5 * ( q0*gx + q2*gz - q3*gy) * dt
            q2 += 0.5 * ( q0*gy - q1*gz + q3*gx) * dt
            q3 += 0.5 * ( q0*gz + q1*gy - q2*gx) * dt
        else:
            ax /= n; ay /= n; az /= n

            q0q0 = q0*q0; q1q1 = q1*q1; q2q2 = q2*q2; q3q3 = q3*q3

            s0 = 4*q0*q2q2 + 2*q2*ax + 4*q0*q1q1 - 2*q1*ay
            s1 = (4*q1*q3q3 - 2*q3*ax + 4*q0q0*q1 - 2*q0*ay
                  - 4*q1 + 8*q1*q1q1 + 8*q1*q2q2 + 4*q1*az)
            s2 = (4*q0q0*q2 + 2*q0*ax + 4*q2*q3q3 - 2*q3*ay
                  - 4*q2 + 8*q2*q1q1 + 8*q2*q2q2 + 4*q2*az)
            s3 = 4*q1q1*q3 - 2*q1*ax + 4*q2q2*q3 - 2*q2*ay

            sn = math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
            if sn > 1e-6:
                s0 /= sn; s1 /= sn; s2 /= sn; s3 /= sn

            q0 += (0.5*(-q1*gx - q2*gy - q3*gz) - beta*s0) * dt
            q1 += (0.5*( q0*gx + q2*gz - q3*gy) - beta*s1) * dt
            q2 += (0.5*( q0*gy - q1*gz + q3*gx) - beta*s2) * dt
            q3 += (0.5*( q0*gz + q1*gy - q2*gx) - beta*s3) * dt

        qn = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        self.q = np.array([q0/qn, q1/qn, q2/qn, q3/qn])
        return self.q.copy()

    def get_q(self):
        return self.q.copy()

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])


# ══════════════════════════════════════════════════════════════
# QUATERNION UTILITIES
# ══════════════════════════════════════════════════════════════
def q_inv(q):
    """Nghịch đảo quaternion đơn vị."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_mul(q, r):
    """Hamilton product q ⊗ r."""
    q0,q1,q2,q3 = q; r0,r1,r2,r3 = r
    return np.array([
        q0*r0 - q1*r1 - q2*r2 - q3*r3,
        q0*r1 + q1*r0 + q2*r3 - q3*r2,
        q0*r2 - q1*r3 + q2*r0 + q3*r1,
        q0*r3 + q1*r2 - q2*r1 + q3*r0,
    ])

def q_to_euler(q):
    """Quaternion → pitch (flexion) và roll (lateral flexion) theo convention Y-up."""
    q0,q1,q2,q3 = q
    pitch = math.asin(max(-1.0, min(1.0, 2*(q0*q2 - q3*q1)))) * 180.0/math.pi
    roll  = math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2)) * 180.0/math.pi
    return pitch, roll

def relative_pitch_roll(q_base, q_head):
    """
    Tính góc tương đối ĐÚNG trong không gian 3D.
    q_rel = q_base^{-1} ⊗ q_head
    """
    q_rel = q_mul(q_inv(q_base), q_head)
    if q_rel[0] < 0:
        q_rel = -q_rel
    return q_to_euler(q_rel)


# ══════════════════════════════════════════════════════════════
# SENSOR PIPELINE
# ══════════════════════════════════════════════════════════════
class SensorPipeline:
    def __init__(self, freq=100.0):
        self.madgwick   = AdaptiveMadgwick(freq=freq)
        self.gyro_bias  = np.zeros(3)  # Static bias đo khi calibrate
        self.last_ts_us = None         # [F4] Timestamp µs lần trước

    def process(self, gx, gy, gz, ax, ay, az, ts_us: float):
        """
        [F6] Không còn temperature compensation — chỉ trừ static bias.
        [F4] dt tính từ timestamp µs thực tế.
        """
        # Trừ gyro bias đo được lúc đứng yên
        gx -= self.gyro_bias[0]
        gy -= self.gyro_bias[1]
        gz -= self.gyro_bias[2]

        # [F4] dt thực tế
        dt = None
        if self.last_ts_us is not None:
            raw_dt = (ts_us - self.last_ts_us) * 1e-6
            # Sanity check: nếu dt quá lạ (mất gói, overflow micros) → dùng nominal
            if 0.005 <= raw_dt <= 0.05:
                dt = raw_dt
        self.last_ts_us = ts_us

        return self.madgwick.update(gx, gy, gz, ax, ay, az, dt)

    def get_q(self):
        return self.madgwick.get_q()

    def reset(self):
        self.madgwick.reset()
        self.last_ts_us = None


# ══════════════════════════════════════════════════════════════
# [F1][F2] SERIAL READER VỚI AUTO-DETECT VÀ RECONNECT
# ══════════════════════════════════════════════════════════════
class SerialReader(threading.Thread):
    def __init__(self, data_queue: queue.Queue):
        super().__init__(daemon=True)
        self.dq          = data_queue
        self.running     = False
        self.ser         = None
        self.ready       = threading.Event()
        self.error       = None
        self.port        = None
        self.dropped     = 0    # [F7] Đếm số packet dropped
        self.crc_errors  = 0    # Đếm số CRC fail

    def _open_serial(self) -> bool:
        """Tìm port và mở kết nối. Trả về True nếu thành công."""
        port = find_esp32_port()
        if port is None:
            self.error = "Không tìm thấy ESP32. Kiểm tra kết nối USB."
            return False
        try:
            self.ser  = serial.Serial(port, BAUD_RATE, timeout=1.0)
            self.port = port
            print(f"[Serial] Mở {port} @ {BAUD_RATE}")
            return True
        except serial.SerialException as e:
            self.error = str(e)
            return False

    def _wait_ready(self) -> bool:
        """Chờ ESP32 gửi READY hoặc bắt được data stream. Timeout 15s."""
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                line = self.ser.readline().decode("utf-8", "ignore").strip()
                if line.startswith("READY:"):
                    print(f"[ESP32] {line}")
                    return True
                if line.startswith("ERROR:"):
                    self.error = line
                    return False
                # Bắt được data stream mà không có READY (board đang chạy rồi)
                if len(line.split(",")) == 16:
                    print("[ESP32] Bắt được data stream (bỏ qua READY)")
                    return True
            except Exception:
                pass
        self.error = "Timeout: Không nhận READY từ ESP32 sau 15s"
        return False

    def run(self):
        if not self._open_serial():
            self.ready.set()
            return

        if not self._wait_ready():
            self.ready.set()
            return

        self.running = True
        self.ready.set()

        while self.running:
            try:
                if not self.ser.is_open:
                    raise serial.SerialException("Port đã đóng")

                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", "ignore").strip()
                if not line or line.startswith(("HEADER", "ERROR", "WARN", "READY")):
                    if line.startswith("WARN"):
                        print(f"[ESP32] {line}")
                    continue

                # [F3] CRC validation
                parsed = validate_and_parse(line)
                if parsed is None:
                    self.crc_errors += 1
                    continue

                # [F7] Tracking dropped packets
                if self.dq.full():
                    self.dropped += 1
                    try:
                        self.dq.get_nowait()   # Drop oldest
                    except queue.Empty:
                        pass

                self.dq.put_nowait(parsed)

            except serial.SerialException:
                # [F2] Reconnect tự động
                print(f"\n[Serial] Mất kết nối. Thử reconnect sau 2s...")
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                time.sleep(2.0)
                if self._open_serial() and self._wait_ready():
                    print("[Serial] Reconnect thành công!")
                else:
                    print("[Serial] Reconnect thất bại. Thử lại...")
                    time.sleep(3.0)
            except Exception:
                pass

    def stop(self):
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# [F5] CALIBRATION VỚI ADAPTIVE WARMUP
# ══════════════════════════════════════════════════════════════
def calibrate(p1: SensorPipeline, p2: SensorPipeline,
              dq: queue.Queue, calib_s: float):
    """
    Thu thập data tĩnh → tính gyro bias → warmup Madgwick đến hội tụ.
    """
    n = int(calib_s * SAMPLE_RATE_HZ)
    print(f"\n{'═'*55}")
    print(f"  CALIBRATION: YÊU CẦU BỆNH NHÂN GIỮ YÊN TRONG {calib_s}s")
    print(f"{'═'*55}")

    samples = []
    for i in range(n):
        while dq.empty():
            time.sleep(0.001)
        samples.append(dq.get())
        bar = int((i+1)/n * 40)
        print(f"\r  [{'█'*bar}{'░'*(40-bar)}] {i+1}/{n}", end="", flush=True)

    print()
    arr = np.array(samples)
    # Cột: 0:t_us, 1-3:a1, 4-6:g1, 7-9:a2, 10-12:g2, 13:T1, 14:T2

    # Static gyro bias (trung bình khi đứng yên)
    p1.gyro_bias = np.mean(arr[:, 4:7], axis=0)
    p2.gyro_bias = np.mean(arr[:, 10:13], axis=0)

    # Kiểm tra chất lượng calibration — std dev quá cao → sensor bị rung
    std1 = np.std(arr[:, 4:7], axis=0)
    std2 = np.std(arr[:, 10:13], axis=0)
    BIAS_STD_WARN = 0.005  # rad/s
    if np.any(std1 > BIAS_STD_WARN) or np.any(std2 > BIAS_STD_WARN):
        print(f"\n  ⚠ CẢNH BÁO: Sensor bị rung trong lúc calibrate.")
        print(f"    Std T13:  {np.round(std1, 4)} rad/s")
        print(f"    Std Head: {np.round(std2, 4)} rad/s")
        print(f"    Nên yêu cầu bệnh nhân giữ yên hơn và calibrate lại.\n")

    print(f"\n  Bias T13 : {np.round(p1.gyro_bias, 4)} rad/s")
    print(f"  Bias Head: {np.round(p2.gyro_bias, 4)} rad/s")

    # [F5] Warmup đến hội tụ — kiểm tra norm(q_new - q_old) < tol
    print("  Warmup Madgwick filter...")
    p1.reset(); p2.reset()

    CONVERGE_TOL   = 0.0005   # Tiêu chí hội tụ
    MAX_ROUNDS     = 15       # Giới hạn vòng lặp
    for rnd in range(MAX_ROUNDS):
        q1_prev = p1.get_q().copy()
        q2_prev = p2.get_q().copy()
        for v in samples:
            p1.process(v[4],v[5],v[6], v[1],v[2],v[3], v[0])
            p2.process(v[10],v[11],v[12], v[7],v[8],v[9], v[0])
        d1 = np.linalg.norm(p1.get_q() - q1_prev)
        d2 = np.linalg.norm(p2.get_q() - q2_prev)
        print(f"    Round {rnd+1:2d}: ΔQ = [{d1:.6f}, {d2:.6f}]", end="")
        if d1 < CONVERGE_TOL and d2 < CONVERGE_TOL:
            print(" ✓ HỘI TỤ")
            break
        print()
    else:
        print(f"\n  ⚠ Chưa hội tụ sau {MAX_ROUNDS} vòng — tiếp tục với state hiện tại.")

    qr1 = p1.get_q()
    qr2 = p2.get_q()
    print(f"\n  Quaternion ref T13 : {np.round(qr1, 5)}")
    print(f"  Quaternion ref Head: {np.round(qr2, 5)}")
    print(f"{'═'*55}\n")
    return qr1, qr2


# ══════════════════════════════════════════════════════════════
# [F8] GRACEFUL SHUTDOWN
# ══════════════════════════════════════════════════════════════
_rows_ref    = []
_csv_path_ref = ""

def _emergency_save(sig, frame):
    """Lưu CSV khi nhận Ctrl+C hoặc signal."""
    print("\n\n[!] Nhận tín hiệu dừng — đang lưu dữ liệu...")
    _do_save()
    sys.exit(0)

def _do_save():
    if _rows_ref:
        df = pd.DataFrame(_rows_ref)
        df.to_csv(_csv_path_ref, index=False)
        print(f"[*] Đã lưu {len(df)} mẫu → {_csv_path_ref}")
        print(f"    Pitch: {df.rel_pitch_deg.max():.1f}° / {df.rel_pitch_deg.min():.1f}°")
        print(f"    Roll:  {df.rel_roll_deg.max():.1f}° / {df.rel_roll_deg.min():.1f}°")
    else:
        print("[!] Không có dữ liệu để lưu.")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    global _rows_ref, _csv_path_ref

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sid      = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"head_angle_{sid}.csv")
    _csv_path_ref = csv_path

    # [F8] Đăng ký signal handler
    signal.signal(signal.SIGINT,  _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    pipe_t13  = SensorPipeline(freq=SAMPLE_RATE_HZ)
    pipe_head = SensorPipeline(freq=SAMPLE_RATE_HZ)

    dq     = queue.Queue(maxsize=1000)   # Buffer lớn hơn v2
    reader = SerialReader(dq)
    reader.start()

    print("[*] Đang tìm ESP32...")
    reader.ready.wait(timeout=20)

    if reader.error:
        print(f"[LỖI] {reader.error}")
        reader.stop()
        return

    qr1, qr2 = calibrate(pipe_t13, pipe_head, dq, CALIB_SECONDS)

    rows = _rows_ref   # Reference để _emergency_save có thể truy cập
    t0   = time.time()

    N   = PLOT_WINDOW_S * SAMPLE_RATE_HZ
    t_b = deque(maxlen=N)
    p_b = deque(maxlen=N)
    r_b = deque(maxlen=N)

    # ── PLOT ─────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, (axP, axR) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for ax in (axP, axR):
        ax.set_facecolor("#161b22")

    fig.suptitle(f"GÓC CÚI ĐẦU LÂM SÀNG  ·  {sid}",
                 fontsize=12, fontweight="bold", color="#c9d1d9",
                 y=0.98)

    def mk_axis(ax, ylo, yhi, rlo, rhi, ylabel, zone_color, line_color):
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
        ax.set_ylim(ylo, yhi)
        ax.axhline(0, color="#30363d", lw=1.0, ls="--", zorder=1)
        ax.axhspan(rlo, rhi, alpha=0.06, color=zone_color, zorder=0)
        ax.axhline(rhi, color=zone_color, lw=0.8, ls=":", alpha=0.7, zorder=1)
        ax.axhline(rlo, color=zone_color, lw=0.8, ls=":", alpha=0.7, zorder=1)
        ln, = ax.plot([], [], color=line_color, lw=1.6, zorder=2)
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_color("#30363d")
        ax.grid(True, alpha=0.08, color="#8b949e")
        return ln

    lP = mk_axis(axP, -80, 80, CLINICAL_PITCH_MIN, CLINICAL_PITCH_MAX,
                 "Pitch °\n(+ Flexion)", "#3fb950", "#58a6ff")
    lR = mk_axis(axR, -60, 60, CLINICAL_ROLL_MIN,  CLINICAL_ROLL_MAX,
                 "Roll °\n(+ Phải)", "#d29922", "#ff7b72")
    axR.set_xlabel("Thời gian (s)", color="#8b949e", fontsize=9)

    # Status bar phía dưới
    status_text = fig.text(
        0.02, 0.01,
        "Pitch: --°  |  Roll: --°  |  Dropped: 0  |  CRC Err: 0",
        fontsize=9, color="#c9d1d9",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                  edgecolor="#30363d", alpha=0.95)
    )

    # Hướng dẫn tare
    fig.text(0.5, 0.01,
             "Nhấn  Z  để đặt lại góc 0° (Tare)  |  Nhấn  S  để lưu snapshot",
             fontsize=8, color="#8b949e", ha="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                       edgecolor="#30363d", alpha=0.8))

    # Legend vùng normal
    norm_patch_p = mpatches.Patch(color="#3fb950", alpha=0.15, label=f"Normal ±{CLINICAL_PITCH_MAX}°")
    norm_patch_r = mpatches.Patch(color="#d29922", alpha=0.15, label=f"Normal ±{CLINICAL_ROLL_MAX}°")
    axP.legend(handles=[norm_patch_p], loc="upper right", fontsize=7,
               facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
    axR.legend(handles=[norm_patch_r], loc="upper right", fontsize=7,
               facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")

    # ── Tare (Z) và Snapshot (S) ──────────────────────────────
    def on_key_press(event):
        nonlocal qr1, qr2
        if event.key.lower() == "z":
            qr1 = pipe_t13.get_q()
            qr2 = pipe_head.get_q()
            print("\n[Tare] Góc 0° đã được đặt lại.")
        elif event.key.lower() == "s" and rows:
            snap_path = os.path.join(OUTPUT_DIR, f"snapshot_{sid}_{int(time.time())}.csv")
            pd.DataFrame(rows).to_csv(snap_path, index=False)
            print(f"\n[Snapshot] Lưu {len(rows)} mẫu → {snap_path}")

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    # ── Animation update ──────────────────────────────────────
    def update(_):
        done = 0
        while not dq.empty() and done < 30:
            v = dq.get(); done += 1

            ts_us = v[0]
            ax1, ay1, az1 = v[1], v[2], v[3]
            gx1, gy1, gz1 = v[4], v[5], v[6]
            ax2, ay2, az2 = v[7], v[8], v[9]
            gx2, gy2, gz2 = v[10], v[11], v[12]
            T1, T2        = v[13], v[14]

            q1 = pipe_t13.process(gx1, gy1, gz1, ax1, ay1, az1, ts_us)
            q2 = pipe_head.process(gx2, gy2, gz2, ax2, ay2, az2, ts_us)

            # Tính góc tương đối so với reference quaternion (sau Tare)
            q1_adj = q_mul(q_inv(qr1), q1)
            q2_adj = q_mul(q_inv(qr2), q2)
            rp, rr = relative_pitch_roll(q1_adj, q2_adj)

            tn = time.time() - t0
            t_b.append(tn); p_b.append(rp); r_b.append(rr)

            rows.append({
                "timestamp_s"   : round(tn, 4),
                "rel_pitch_deg" : round(rp, 3),
                "rel_roll_deg"  : round(rr, 3),
                "q1w": round(q1[0],5), "q1x": round(q1[1],5),
                "q1y": round(q1[2],5), "q1z": round(q1[3],5),
                "q2w": round(q2[0],5), "q2x": round(q2[1],5),
                "q2y": round(q2[2],5), "q2z": round(q2[3],5),
                "temp1_C"       : round(T1, 2),
                "temp2_C"       : round(T2, 2),
                "madgwick_beta1": round(pipe_t13.madgwick.beta, 4),
                "madgwick_beta2": round(pipe_head.madgwick.beta, 4),
            })

        if not t_b:
            return lP, lR

        ta = np.array(t_b); pa = np.array(p_b); ra = np.array(r_b)
        lP.set_data(ta, pa); lR.set_data(ta, ra)
        xm = max(0.0, ta[-1] - PLOT_WINDOW_S)
        axP.set_xlim(xm, xm + PLOT_WINDOW_S)
        axR.set_xlim(xm, xm + PLOT_WINDOW_S)

        cp = pa[-1]; cr = ra[-1]
        wp = " ⚠" if abs(cp) > abs(CLINICAL_PITCH_MAX) else ""
        wr = " ⚠" if abs(cr) > abs(CLINICAL_ROLL_MAX)  else ""

        status_text.set_text(
            f"Pitch: {cp:+6.1f}°{wp}   |   Roll: {cr:+6.1f}°{wr}"
            f"   |   Dropped: {reader.dropped}   |   CRC Err: {reader.crc_errors}"
        )
        # Đổi màu khi vượt ngưỡng
        lP.set_color("#ff7b72" if abs(cp) > abs(CLINICAL_PITCH_MAX) else "#58a6ff")
        lR.set_color("#ff7b72" if abs(cr) > abs(CLINICAL_ROLL_MAX)  else "#ff7b72"
                     if abs(cr) > abs(CLINICAL_ROLL_MAX) else "#e3b341")

        return lP, lR

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    print(f"[*] Đang đo. Nhấn Z để Tare, S để Snapshot.")
    print(f"[*] Đóng cửa sổ hoặc Ctrl+C để lưu và thoát.")
    print(f"[*] CSV sẽ lưu tại: {csv_path}\n")

    try:
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        plt.show()
    except KeyboardInterrupt:
        pass

    reader.stop()
    _do_save()


if __name__ == "__main__":
    main()
