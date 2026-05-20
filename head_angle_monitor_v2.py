cat > /home/claude/head_angle_monitor.py << 'PYEOF'
"""
head_angle_monitor.py  v2.0
Đo góc cúi đầu lâm sàng — 2 MPU6050 (T1-T3 + Chom dau)

Fix so với v1:
  [Fix 1] Quaternion relative angle: q_rel = q1_inv x q2
  [Fix 2] Bỏ EKF cascade, chỉ dùng Adaptive Madgwick đã tune
  [Fix 3] Baudrate 921600
  [Fix 4] Temperature compensation gyro trước khi lọc

Cài: pip install pyserial numpy pandas matplotlib
"""

import serial, time, math, os, threading, queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from datetime import datetime

# ── Cấu hình ─────────────────────────────────────────────────
SERIAL_PORT    = "COM5"
BAUD_RATE      = 921600        # Fix 3
SAMPLE_RATE_HZ = 100
CALIB_SECONDS  = 3
PLOT_WINDOW_S  = 15
OUTPUT_DIR     = "recordings"

CLINICAL_PITCH_MAX =  50.0
CLINICAL_PITCH_MIN = -50.0
CLINICAL_ROLL_MAX  =  45.0
CLINICAL_ROLL_MIN  = -45.0

GYRO_TEMP_COEFF = 0.02         # (deg/s) / degC — MPU6050 datasheet


# ── Fix 4: Temperature Compensator ───────────────────────────
class TempCompensator:
    """
    Bù tuyến tính: bias_corrected = gyro_raw - k*(T - T_ref)
    T_ref lưu lại khi calibrate.
    """
    def __init__(self, coeff=GYRO_TEMP_COEFF):
        self.k     = coeff * (math.pi / 180.0)  # rad/s / degC
        self.T_ref = None

    def set_ref(self, T):
        self.T_ref = T

    def apply(self, gx, gy, gz, T):
        if self.T_ref is None:
            return gx, gy, gz
        c = self.k * (T - self.T_ref)
        return gx - c, gy - c, gz - c


# ── Fix 2: Adaptive Madgwick (không cascade EKF) ─────────────
class AdaptiveMadgwick:
    """
    Madgwick 6-DOF với beta thích nghi.
    Tuned cho chuyển động đầu người:
      accel_threshold = 0.10g (tay=0.30g, drone=0.50g)
      beta_min = 0.008  (tin gyro khi cử động)
      beta_max = 0.10   (kéo về accel khi đứng yên)
    """
    def __init__(self, freq=100.0, beta_min=0.008, beta_max=0.10, accel_thr=0.10):
        self.dt      = 1.0 / freq
        self.bmin    = beta_min
        self.bmax    = beta_max
        self.athr    = accel_thr
        self.q       = np.array([1.0, 0.0, 0.0, 0.0])
        self.beta    = beta_max

    def _adapt(self, ax, ay, az):
        dev      = abs(math.sqrt(ax*ax + ay*ay + az*az) - 1.0)
        alpha    = min(dev / self.athr, 1.0)
        self.beta = self.bmin + (1.0 - alpha) * (self.bmax - self.bmin)

    def update(self, gx, gy, gz, ax, ay, az):
        self._adapt(ax, ay, az)
        beta = self.beta
        q0,q1,q2,q3 = self.q

        n = math.sqrt(ax*ax + ay*ay + az*az)
        if n < 1e-6:
            return self.q.copy()
        ax/=n; ay/=n; az/=n

        q0q0=q0*q0; q1q1=q1*q1; q2q2=q2*q2; q3q3=q3*q3

        s0 = 4*q0*q2q2 + 2*q2*ax + 4*q0*q1q1 - 2*q1*ay
        s1 = (4*q1*q3q3 - 2*q3*ax + 4*q0q0*q1 - 2*q0*ay
              - 4*q1 + 8*q1*q1q1 + 8*q1*q2q2 + 4*q1*az)
        s2 = (4*q0q0*q2 + 2*q0*ax + 4*q2*q3q3 - 2*q3*ay
              - 4*q2 + 8*q2*q1q1 + 8*q2*q2q2 + 4*q2*az)
        s3 = 4*q1q1*q3 - 2*q1*ax + 4*q2q2*q3 - 2*q2*ay

        sn = math.sqrt(s0*s0+s1*s1+s2*s2+s3*s3)
        if sn > 1e-6:
            s0/=sn; s1/=sn; s2/=sn; s3/=sn

        q0 += (0.5*(-q1*gx-q2*gy-q3*gz) - beta*s0)*self.dt
        q1 += (0.5*( q0*gx+q2*gz-q3*gy) - beta*s1)*self.dt
        q2 += (0.5*( q0*gy-q1*gz+q3*gx) - beta*s2)*self.dt
        q3 += (0.5*( q0*gz+q1*gy-q2*gx) - beta*s3)*self.dt

        qn = math.sqrt(q0*q0+q1*q1+q2*q2+q3*q3)
        self.q = np.array([q0/qn, q1/qn, q2/qn, q3/qn])
        return self.q.copy()

    def get_q(self):
        return self.q.copy()

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])


# ── Fix 1: Quaternion arithmetic ─────────────────────────────
def q_inv(q):
    """Nghịch đảo quaternion đơn vị."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_mul(q, r):
    """Phép nhân Hamilton q ⊗ r."""
    q0,q1,q2,q3 = q; r0,r1,r2,r3 = r
    return np.array([
        q0*r0 - q1*r1 - q2*r2 - q3*r3,
        q0*r1 + q1*r0 + q2*r3 - q3*r2,
        q0*r2 - q1*r3 + q2*r0 + q3*r1,
        q0*r3 + q1*r2 - q2*r1 + q3*r0,
    ])

def q_to_pitch_roll(q):
    q0,q1,q2,q3 = q
    pitch = math.asin(max(-1.0, min(1.0, 2*(q0*q2 - q3*q1)))) * 180/math.pi
    roll  = math.atan2(2*(q0*q1+q2*q3), 1-2*(q1*q1+q2*q2)) * 180/math.pi
    return pitch, roll

def relative_pitch_roll(q_base, q_head):
    """
    Góc tương đối ĐÚNG trong không gian 3D:
      q_rel = q_base^{-1} ⊗ q_head
    Tránh cross-axis error khi đồng thời Pitch + Roll.
    """
    q_rel = q_mul(q_inv(q_base), q_head)
    if q_rel[0] < 0:
        q_rel = -q_rel
    return q_to_pitch_roll(q_rel)


# ── Pipeline một sensor ───────────────────────────────────────
class SensorPipeline:
    def __init__(self, freq=100.0):
        self.madgwick = AdaptiveMadgwick(freq=freq)
        self.tc       = TempCompensator()

    def process(self, gx, gy, gz, ax, ay, az, T):
        gx, gy, gz = self.tc.apply(gx, gy, gz, T)  # Fix 4
        return self.madgwick.update(gx, gy, gz, ax, ay, az)

    def get_q(self):
        return self.madgwick.get_q()

    def reset(self):
        self.madgwick.reset()


# ── Serial Reader ─────────────────────────────────────────────
class SerialReader(threading.Thread):
    def __init__(self, port, baud, q):
        super().__init__(daemon=True)
        self.port=port; self.baud=baud; self.q=q
        self.running=False; self.ser=None
        self.ready=threading.Event(); self.error=None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"[Serial] {self.port} @ {self.baud}")
        except Exception as e:
            self.error=str(e); self.ready.set(); return

        self.running=True
        deadline = time.time() + 15
        while self.running and time.time() < deadline:
            try:
                line = self.ser.readline().decode("utf-8","ignore").strip()
                if line.startswith("READY:"):
                    print(f"[ESP32] {line}"); self.ready.set(); break
                elif line.startswith("ERROR:"):
                    self.error=line; self.ready.set(); return
            except Exception:
                pass

        if not self.ready.is_set():
            self.error="Timeout: Không nhận READY từ ESP32"
            self.ready.set(); return

        while self.running:
            try:
                line = self.ser.readline().decode("utf-8","ignore").strip()
                if not line or line.startswith(("HEADER","ERROR")):
                    continue
                parts = line.split(",")
                if len(parts) == 15 and not self.q.full():
                    self.q.put_nowait(list(map(float, parts)))
            except Exception:
                pass

    def stop(self):
        self.running=False
        if self.ser and self.ser.is_open:
            self.ser.close()


# ── Calibration ───────────────────────────────────────────────
def calibrate(p1, p2, dq, calib_s):
    n = int(calib_s * SAMPLE_RATE_HZ)
    print(f"\n[Calibrate] Giữ thẳng, không cử động trong {calib_s}s...")

    qa1 = np.zeros(4); qa2 = np.zeros(4)
    Ts1 = 0.0;         Ts2 = 0.0

    for i in range(n):
        while dq.empty():
            time.sleep(0.001)
        v = dq.get()
        ax1,ay1,az1 = v[1],v[2],v[3]; gx1,gy1,gz1 = v[4],v[5],v[6]
        ax2,ay2,az2 = v[7],v[8],v[9]; gx2,gy2,gz2 = v[10],v[11],v[12]
        T1,T2 = v[13],v[14]

        q_1 = p1.process(gx1,gy1,gz1, ax1,ay1,az1, T1)
        q_2 = p2.process(gx2,gy2,gz2, ax2,ay2,az2, T2)

        if np.dot(qa1, q_1) < 0: q_1 = -q_1
        if np.dot(qa2, q_2) < 0: q_2 = -q_2
        qa1 += q_1; qa2 += q_2
        Ts1 += T1;  Ts2 += T2

        bar = int((i+1)/n*30)
        print(f"\r  [{'█'*bar}{'░'*(30-bar)}] {i+1}/{n}", end="", flush=True)

    qr1 = qa1/np.linalg.norm(qa1); qr2 = qa2/np.linalg.norm(qa2)
    Tr1 = Ts1/n; Tr2 = Ts2/n

    p1.tc.set_ref(Tr1); p2.tc.set_ref(Tr2)
    print(f"\n[Calibrate] Xong — T_ref: {Tr1:.1f}°C / {Tr2:.1f}°C")

    p1.reset(); p2.reset()
    return qr1, qr2


# ── Main ──────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"head_angle_{sid}.csv")

    pipe_t13  = SensorPipeline(freq=SAMPLE_RATE_HZ)
    pipe_head = SensorPipeline(freq=SAMPLE_RATE_HZ)

    dq = queue.Queue(maxsize=500)
    reader = SerialReader(SERIAL_PORT, BAUD_RATE, dq)
    reader.start()
    print("[*] Chờ ESP32...")
    reader.ready.wait(timeout=15)

    if reader.error:
        print(f"[LỖI] {reader.error}")
        return

    qr1, qr2 = calibrate(pipe_t13, pipe_head, dq, CALIB_SECONDS)

    N   = PLOT_WINDOW_S * SAMPLE_RATE_HZ
    t_b = deque(maxlen=N); p_b = deque(maxlen=N); r_b = deque(maxlen=N)
    rows = []; t0 = time.time()

    # ── Plot ──────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, (axP, axR) = plt.subplots(2,1, figsize=(13,8), sharex=True)
    fig.suptitle(f"GÓC CÚI ĐẦU LÂM SÀNG — {sid}",
                 fontsize=13, fontweight="bold", color="#e0e0e0")

    def mk_axis(ax, ylo, yhi, rlo, rhi, ylabel, rc, lc, lbl):
        ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=9)
        ax.set_ylim(ylo, yhi)
        ax.axhline(0, color="#444", lw=0.8, ls="--")
        ax.axhspan(rlo, rhi, alpha=0.07, color=rc)
        ax.axhline(rhi, color=rc, lw=0.7, ls=":")
        ax.axhline(rlo, color=rc, lw=0.7, ls=":")
        ln, = ax.plot([], [], color=lc, lw=1.8, label=lbl)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.12)
        return ln

    lP = mk_axis(axP,-70,70, CLINICAL_PITCH_MIN,CLINICAL_PITCH_MAX,
                 "Pitch (°)\n[+=Cúi trước]","#00c896","#4fc3f7",
                 "Pitch — Quaternion relative")
    lR = mk_axis(axR,-60,60, CLINICAL_ROLL_MIN,CLINICAL_ROLL_MAX,
                 "Roll (°)\n[+=Nghiêng phải]","#ffb74d","#ff8a65",
                 "Roll — Quaternion relative")
    axR.set_xlabel("Thời gian (s)", color="#e0e0e0")

    txt = fig.text(0.01,0.01,"Pitch: --°  |  Roll: --°",
                   fontsize=10, color="#e0e0e0",
                   bbox=dict(boxstyle="round",facecolor="#111",alpha=0.85))
    fig.text(0.99,0.01,"● LIVE v2.0",fontsize=9,color="#00e676",ha="right",
             bbox=dict(boxstyle="round",facecolor="#111",alpha=0.85))
    plt.tight_layout(rect=[0,0.04,1,0.95])

    def update(_):
        done = 0
        while not dq.empty() and done < 20:
            v = dq.get(); done += 1
            ax1,ay1,az1=v[1],v[2],v[3]; gx1,gy1,gz1=v[4],v[5],v[6]
            ax2,ay2,az2=v[7],v[8],v[9]; gx2,gy2,gz2=v[10],v[11],v[12]
            T1,T2=v[13],v[14]

            q1 = pipe_t13.process(gx1,gy1,gz1, ax1,ay1,az1, T1)
            q2 = pipe_head.process(gx2,gy2,gz2, ax2,ay2,az2, T2)

            # Fix 1: tính góc tương đối đúng trên Quaternion
            q1_adj = q_mul(q_inv(qr1), q1)
            q2_adj = q_mul(q_inv(qr2), q2)
            rp, rr = relative_pitch_roll(q1_adj, q2_adj)

            tn = time.time()-t0
            t_b.append(tn); p_b.append(rp); r_b.append(rr)
            rows.append({"timestamp_s":round(tn,4),
                         "rel_pitch_deg":round(rp,3),
                         "rel_roll_deg":round(rr,3),
                         "q1w":round(q1[0],5),"q1x":round(q1[1],5),
                         "q1y":round(q1[2],5),"q1z":round(q1[3],5),
                         "q2w":round(q2[0],5),"q2x":round(q2[1],5),
                         "q2y":round(q2[2],5),"q2z":round(q2[3],5),
                         "temp1_C":round(T1,2),"temp2_C":round(T2,2)})

        if not t_b:
            return lP, lR

        ta=np.array(t_b); pa=np.array(p_b); ra=np.array(r_b)
        lP.set_data(ta,pa); lR.set_data(ta,ra)
        xm = max(0.0, ta[-1]-PLOT_WINDOW_S)
        axP.set_xlim(xm, xm+PLOT_WINDOW_S)
        axR.set_xlim(xm, xm+PLOT_WINDOW_S)

        cp=pa[-1]; cr=ra[-1]
        wp=" ⚠ NGOÀI NGƯỠNG" if abs(cp)>abs(CLINICAL_PITCH_MAX) else ""
        wr=" ⚠ NGOÀI NGƯỠNG" if abs(cr)>abs(CLINICAL_ROLL_MAX)  else ""
        txt.set_text(f"Pitch: {cp:+.1f}°{wp}   |   Roll: {cr:+.1f}°{wr}")
        return lP, lR

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    print(f"\n[*] Đóng cửa sổ để lưu và thoát.\n[*] CSV: {csv_path}")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    reader.stop()
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"\n[*] Lưu {len(df)} mẫu → {csv_path}")
        print(f"    Pitch: {df.rel_pitch_deg.max():.1f}° / {df.rel_pitch_deg.min():.1f}°")
        print(f"    Roll:  {df.rel_roll_deg.max():.1f}°  / {df.rel_roll_deg.min():.1f}°")
    else:
        print("\n[!] Không có dữ liệu.")

if __name__ == "__main__":
    main()
PYEOF