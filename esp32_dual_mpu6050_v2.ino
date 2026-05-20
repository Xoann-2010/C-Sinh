/*
 * ═══════════════════════════════════════════════════════════════
 *  esp32_dual_mpu6050.ino
 *  Đọc 2 MPU6050 qua I2C, stream dữ liệu qua Serial
 *
 *  Wiring:
 *    Sensor 1 (T1-T3)    : AD0 → GND  → địa chỉ 0x68
 *    Sensor 2 (Chỏm đầu) : AD0 → 3.3V → địa chỉ 0x69
 *    SDA → GPIO21, SCL → GPIO22 (ESP32 mặc định)
 *
 *  Serial format (CSV, 921600 baud):
 *    timestamp_ms, ax1,ay1,az1, gx1,gy1,gz1, ax2,ay2,az2, gx2,gy2,gz2, temp1, temp2
 *    Đơn vị: accel = g, gyro = rad/s, temp = °C
 * ═══════════════════════════════════════════════════════════════
 */

#include <Wire.h>

// ── Địa chỉ I2C ──────────────────────────────────────────────
#define MPU_ADDR_T13   0x68   // Sensor T1-T3  (AD0 = LOW)
#define MPU_ADDR_HEAD  0x69   // Sensor chỏm đầu (AD0 = HIGH)

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

// ── Scale factors ─────────────────────────────────────────────
// Accel FS = ±2g  → 16384 LSB/g
// Gyro  FS = ±500°/s → 65.5 LSB/(°/s) → /65.5 * π/180 để ra rad/s
#define ACCEL_SCALE  (1.0f / 16384.0f)
#define GYRO_SCALE   (1.0f / 65.5f * 0.017453293f)   // rad/s

// ── Tần số lấy mẫu ───────────────────────────────────────────
#define SAMPLE_RATE_HZ  100
#define SAMPLE_US       (1000000 / SAMPLE_RATE_HZ)

// ─────────────────────────────────────────────────────────────
struct RawData {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
  int16_t temp;
};

// ─────────────────────────────────────────────────────────────
bool initMPU(uint8_t addr) {
  // Wake up
  Wire.beginTransmission(addr);
  Wire.write(REG_PWR_MGMT_1);
  Wire.write(0x00);   // Clock source = internal 8MHz, wake
  if (Wire.endTransmission() != 0) return false;
  delay(100);

  // Sample rate divider: SMPLRT_DIV = 7 → 1kHz/(7+1) = 125Hz (lấy dư)
  Wire.beginTransmission(addr);
  Wire.write(REG_SMPLRT_DIV);
  Wire.write(0x07);
  Wire.endTransmission();

  // DLPF bandwidth = 42Hz (tránh aliasing ở 100Hz)
  Wire.beginTransmission(addr);
  Wire.write(REG_CONFIG);
  Wire.write(0x03);
  Wire.endTransmission();

  // Gyro FS = ±500°/s
  Wire.beginTransmission(addr);
  Wire.write(REG_GYRO_CFG);
  Wire.write(0x08);
  Wire.endTransmission();

  // Accel FS = ±2g
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_CFG);
  Wire.write(0x00);
  Wire.endTransmission();

  // Kiểm tra WHO_AM_I (phải trả về 0x68)
  Wire.beginTransmission(addr);
  Wire.write(REG_WHO_AM_I);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, (uint8_t)1);
  uint8_t whoami = Wire.read();
  return (whoami == 0x68);
}

// ─────────────────────────────────────────────────────────────
bool readMPU(uint8_t addr, RawData &d) {
  Wire.beginTransmission(addr);
  Wire.write(REG_ACCEL_XOUT);
  if (Wire.endTransmission(false) != 0) return false;

  // Đọc 14 byte: accel(6) + temp(2) + gyro(6)
  Wire.requestFrom(addr, (uint8_t)14);
  if (Wire.available() < 14) return false;

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
void setup() {
  Serial.begin(921600);
  Wire.begin(21, 22);       // SDA=21, SCL=22
  Wire.setClock(400000);    // Fast mode 400kHz

  delay(200);

  bool ok1 = initMPU(MPU_ADDR_T13);
  bool ok2 = initMPU(MPU_ADDR_HEAD);

  if (!ok1 || !ok2) {
    // Báo lỗi qua Serial — Python sẽ bắt dòng bắt đầu bằng "ERROR:"
    Serial.print("ERROR:SENSOR_INIT:");
    Serial.print(ok1 ? "T13_OK" : "T13_FAIL");
    Serial.print(",");
    Serial.println(ok2 ? "HEAD_OK" : "HEAD_FAIL");
    while (1) delay(1000);
  }

  // Báo sẵn sàng — Python chờ dòng này trước khi bắt đầu đọc
  Serial.println("READY:DUAL_MPU6050:100HZ");
  // Header CSV để Python parse dễ hơn
  Serial.println("HEADER:t_ms,ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2,temp1,temp2");
}

// ─────────────────────────────────────────────────────────────
void loop() {
  static uint32_t lastUs = 0;
  uint32_t now = micros();

  if (now - lastUs < SAMPLE_US) return;
  lastUs = now;

  RawData d1, d2;
  bool ok1 = readMPU(MPU_ADDR_T13,  d1);
  bool ok2 = readMPU(MPU_ADDR_HEAD, d2);

  if (!ok1 || !ok2) {
    Serial.println("ERROR:READ_FAIL");
    return;
  }

  // Convert và stream
  // Format: timestamp_ms, 6 giá trị sensor1, 6 giá trị sensor2, 2 nhiệt độ
  float ax1 = d1.ax * ACCEL_SCALE,  ay1 = d1.ay * ACCEL_SCALE,  az1 = d1.az * ACCEL_SCALE;
  float gx1 = d1.gx * GYRO_SCALE,   gy1 = d1.gy * GYRO_SCALE,   gz1 = d1.gz * GYRO_SCALE;
  float ax2 = d2.ax * ACCEL_SCALE,  ay2 = d2.ay * ACCEL_SCALE,  az2 = d2.az * ACCEL_SCALE;
  float gx2 = d2.gx * GYRO_SCALE,   gy2 = d2.gy * GYRO_SCALE,   gz2 = d2.gz * GYRO_SCALE;
  float temp1 = d1.temp / 340.0f + 36.53f;
  float temp2 = d2.temp / 340.0f + 36.53f;

  // In ra CSV với 4 chữ số thập phân
  Serial.print(millis());        Serial.print(",");
  Serial.print(ax1, 4);          Serial.print(",");
  Serial.print(ay1, 4);          Serial.print(",");
  Serial.print(az1, 4);          Serial.print(",");
  Serial.print(gx1, 5);          Serial.print(",");
  Serial.print(gy1, 5);          Serial.print(",");
  Serial.print(gz1, 5);          Serial.print(",");
  Serial.print(ax2, 4);          Serial.print(",");
  Serial.print(ay2, 4);          Serial.print(",");
  Serial.print(az2, 4);          Serial.print(",");
  Serial.print(gx2, 5);          Serial.print(",");
  Serial.print(gy2, 5);          Serial.print(",");
  Serial.print(gz2, 5);          Serial.print(",");
  Serial.print(temp1, 2);        Serial.print(",");
  Serial.println(temp2, 2);
}
