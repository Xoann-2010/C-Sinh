/*
 * ═══════════════════════════════════════════════════════════════
 * esp32_dual_mpu6050_v3_Final.ino
 * Đọc 2 MPU6050 qua I2C (Single Bus), stream dữ liệu qua Serial
 * Tối ưu cho ESP32-C3 (SDA=8, SCL=9)
 * ═══════════════════════════════════════════════════════════════
 */

#include <Wire.h>

// ── Cấu hình chân I2C cho ESP32-C3 ───────────────────────────
#define I2C_SDA 4
#define I2C_SCL 5

// ── Địa chỉ I2C ──────────────────────────────────────────────
#define MPU_ADDR_T13   0x68
#define MPU_ADDR_HEAD  0x69

// ── Thanh ghi MPU6050 ────────────────────────────────────────
#define REG_PWR_MGMT_1  0x6B
#define REG_SMPLRT_DIV  0x19
#define REG_CONFIG      0x1A
#define REG_GYRO_CFG    0x1B
#define REG_ACCEL_CFG   0x1C
#define REG_ACCEL_XOUT  0x3B
#define REG_TEMP_OUT    0x41
#define REG_GYRO_XOUT   0x43
#define REG_WHO_AM_I    0x75
#define REG_INT_STATUS  0x3A
#define REG_USER_CTRL   0x6A
#define REG_FIFO_EN     0x23

// ── Scale factors ─────────────────────────────────────────────
#define ACCEL_SCALE  (1.0f / 16384.0f)
#define GYRO_SCALE   (1.0f / 131.0f * 0.017453293f)   // rad/s

#define SAMPLE_RATE_HZ  100
#define SAMPLE_US       (1000000UL / SAMPLE_RATE_HZ)

// ── Định nghĩa Struct ĐẶT Ở ĐÂY ĐỂ TRÁNH LỖI BIÊN DỊCH ───────
struct RawData {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  int16_t temp;
};

// ── CRC8 (polynomial 0x07) ───────────────────────────────────
uint8_t crc8(const uint8_t *data, size_t len) {
  uint8_t crc = 0x00;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int j = 0; j < 8; j++)
      crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : (crc << 1);
  }
  return crc;
}

// ─────────────────────────────────────────────────────────────
bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

// ─────────────────────────────────────────────────────────────
bool initMPU(uint8_t addr) {
  Serial.print("  -> [DEBUG] Dang init MPU o dia chi 0x"); 
  Serial.println(addr, HEX);

  if (!writeReg(addr, REG_PWR_MGMT_1, 0x80)) {
      Serial.println("    [FAIL] Khong the gui lenh Reset (0x80)");
      return false;
  }
  delay(150);

  if (!writeReg(addr, REG_PWR_MGMT_1, 0x01)) {
      Serial.println("    [FAIL] Khong the set Clock PLL (0x01)");
      return false;
  }
  delay(50);

  if (!writeReg(addr, REG_SMPLRT_DIV, 0x09)) { Serial.println("    [FAIL] Set Rate"); return false; }
  if (!writeReg(addr, REG_CONFIG, 0x04)) { Serial.println("    [FAIL] Set DLPF"); return false; }
  if (!writeReg(addr, REG_GYRO_CFG, 0x00)) { Serial.println("    [FAIL] Set Gyro"); return false; }
  if (!writeReg(addr, REG_ACCEL_CFG, 0x00)) { Serial.println("    [FAIL] Set Accel"); return false; }
  if (!writeReg(addr, REG_USER_CTRL, 0x00)) { Serial.println("    [FAIL] Tat FIFO"); return false; }
  if (!writeReg(addr, REG_FIFO_EN,   0x00)) { Serial.println("    [FAIL] Tat FIFO_EN"); return false; }

  // Kiểm tra WHO_AM_I (Dùng Stop bit bình thường, KHÔNG dùng Repeated Start)
  Wire.beginTransmission(addr);
  Wire.write(REG_WHO_AM_I);
  if (Wire.endTransmission() != 0) { // Đã bỏ 'false'
      Serial.println("    [FAIL] Khong the ghi vao thanh ghi WHO_AM_I");
      return false;
  }

  Wire.requestFrom(addr, (uint8_t)1);
  if (!Wire.available()) {
      Serial.println("    [FAIL] Khong the doc du lieu WHO_AM_I");
      return false;
  }

  uint8_t whoami = Wire.read();
  Serial.print("    [OK] WHO_AM_I doc duoc = 0x"); 
  Serial.println(whoami, HEX);

  if (whoami != 0x68) {
      Serial.println("    [WARN] Phat hien chip clone, nhung van tiep tuc chay!");
      // Không return false nữa, ép nó Pass luôn
  }

  return true;
}

// ─────────────────────────────────────────────────────────────
bool readMPU(uint8_t addr, RawData &d) {
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_XOUT);
  if (Wire.endTransmission(false) != 0) return false;
  
  uint8_t n = Wire.requestFrom(addr, (uint8_t)14);
  if (n < 14) return false;

  d.ax   = (int16_t)(Wire.read() << 8 | Wire.read());
  d.ay   = (int16_t)(Wire.read() << 8 | Wire.read());
  d.az   = (int16_t)(Wire.read() << 8 | Wire.read());
  d.temp = (int16_t)(Wire.read() << 8 | Wire.read());
  d.gx   = (int16_t)(Wire.read() << 8 | Wire.read());
  d.gy   = (int16_t)(Wire.read() << 8 | Wire.read());
  d.gz   = (int16_t)(Wire.read() << 8 | Wire.read());
  return true;
}

// ─────────────────────────────────────────────────────────────
void sendData(uint32_t ts_us,
              float ax1, float ay1, float az1, float gx1, float gy1, float gz1,
              float ax2, float ay2, float az2, float gx2, float gy2, float gz2,
              float t1,  float t2) {
  char buf[256];
  int len = snprintf(buf, sizeof(buf),
    "%lu,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.2f,%.2f",
    (unsigned long)ts_us,
    ax1, ay1, az1, gx1, gy1, gz1,
    ax2, ay2, az2, gx2, gy2, gz2,
    t1, t2);
  uint8_t crc = crc8((uint8_t*)buf, len);
  Serial.print(buf);
  Serial.print(",");
  Serial.println(crc);
}

// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(921600);
  delay(2000); // Chờ cổng Serial mở

 Wire.begin(I2C_SDA, I2C_SCL);
 Wire.setClock(100000);    // Đã hạ xuống Standard mode 100kHz
 Wire.setTimeOut(20);

  delay(250);

  bool ok1 = initMPU(MPU_ADDR_T13);
  bool ok2 = initMPU(MPU_ADDR_HEAD);
  
  if (!ok1 || !ok2) {
    Serial.print("ERROR:SENSOR_INIT:");
    Serial.print(ok1 ? "T13_OK" : "T13_FAIL");
    Serial.print(",");
    Serial.println(ok2 ? "HEAD_OK" : "HEAD_FAIL");
    while (1) delay(1000);
  }

  // Header cho Python nhận diện
  Serial.println("READY:DUAL_MPU6050:100HZ:GYRO250:DLPF20:CRC8");
  Serial.println("HEADER:t_us,ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2,temp1,temp2,crc8");
}

// ─────────────────────────────────────────────────────────────
static uint8_t err_streak = 0;
#define MAX_ERR_STREAK 5

void loop() {
  static uint32_t lastUs = 0;
  uint32_t now = micros();
  
  if (now - lastUs < SAMPLE_US) return;
  lastUs = now;
  
  RawData d1, d2;
  bool ok1 = readMPU(MPU_ADDR_T13,  d1);
  bool ok2 = readMPU(MPU_ADDR_HEAD, d2);
  
  if (!ok1 || !ok2) {
    err_streak++;
    if (err_streak >= MAX_ERR_STREAK) {
      Serial.println("WARN:READ_FAIL:REINIT");
      initMPU(MPU_ADDR_T13);
      initMPU(MPU_ADDR_HEAD);
      err_streak = 0;
    }
    return;
  }
  err_streak = 0;
  
  float ax1 = d1.ax * ACCEL_SCALE, ay1 = d1.ay * ACCEL_SCALE, az1 = d1.az * ACCEL_SCALE;
  float gx1 = d1.gx * GYRO_SCALE,  gy1 = d1.gy * GYRO_SCALE,  gz1 = d1.gz * GYRO_SCALE;
  float ax2 = d2.ax * ACCEL_SCALE, ay2 = d2.ay * ACCEL_SCALE, az2 = d2.az * ACCEL_SCALE;
  float gx2 = d2.gx * GYRO_SCALE,  gy2 = d2.gy * GYRO_SCALE,  gz2 = d2.gz * GYRO_SCALE;
  
  float temp1 = d1.temp / 340.0f + 36.53f;
  float temp2 = d2.temp / 340.0f + 36.53f;

  sendData(now,
           ax1, ay1, az1, gx1, gy1, gz1,
           ax2, ay2, az2, gx2, gy2, gz2,
           temp1, temp2);
}