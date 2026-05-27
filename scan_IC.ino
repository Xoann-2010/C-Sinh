#include <Wire.h>

void setup() {
  Wire.begin(4, 5); // Chân SDA=8, SCL=9 theo thiết kế NeckAngle 
  Serial.begin(115200);
  while (!Serial);
  Serial.println("\nI2C Scanner");
}

void loop() {
  byte error, address;
  int nDevices = 0;
  for(address = 1; address < 127; address++ ) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("Thiết bị tìm thấy tại địa chỉ 0x");
      if (address<16) Serial.print("0");
      Serial.println(address, HEX);
      nDevices++;
    }
  }
  if (nDevices == 0) Serial.println("Không tìm thấy thiết bị I2C nào\n");
  else Serial.println("Hoàn tất\n");
  delay(5000);
}