#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

// ==========================================
// CONFIGURATION (ตั้งค่าเครือข่าย)
// ==========================================
const char* ssid     = "test1234";       // ชื่อ Hotspot มือถือ (คลื่น 2.4GHz)
const char* password = "123456789";       // รหัสผ่าน Hotspot 
const char* udpAddress = "10.220.116.247"; // IP ของคอมพิวเตอร์ (วง Wi-Fi มือถือ)
const int udpPort = 5000;                // พอร์ตเชื่อมต่อ UDP ตรงกับฝั่ง Python

WiFiUDP udp;
// กำหนด Address เป็น 0x29 ตามฮาร์ดแวร์ของคุณ
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29, &Wire);

unsigned long lastTime = 0;
const int samplePeriodMs = 10;  // ความถี่ 10ms = 100 Hz

void setup() {
  Serial.begin(115200);
  delay(100);

  Serial.println("\n--- Starting ESP32 IMU Setup ---");
  
  // ล้างค่า Wi-Fi เก่า ป้องกันอาการค้างจุด ..........
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(500);

  // 1. เริ่มการเชื่อมต่อ Wi-Fi Hotspot
  Serial.printf("Connecting to %s ", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✓ Wi-Fi Connected successfully!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  // 2. เริ่มการใช้งานเซนเซอร์ IMU BNO055
  Serial.println("Initializing BNO055...");
  if (!bno.begin()) {
    Serial.println("Error: BNO055 not detected over I2C at address 0x29!");
    while (1);
  }
  bno.setExtCrystalUse(true);
  Serial.println("✓ BNO055 Initialized successfully!");
}

void loop() {
  // บังคับให้วนลูปด้วยความถี่คงที่ที่ 100Hz (ตรงกับฝั่ง Python)
  if (millis() - lastTime >= samplePeriodMs) {
    lastTime = millis();

    sensors_event_t gyroData, accelData;
    bno.getEvent(&gyroData, Adafruit_BNO055::VECTOR_GYROSCOPE);
    
    // [แก้ไขจุดสำคัญ] เปลี่ยนจาก VECTOR_ACCELEROMETER เป็น VECTOR_LINEARACCEL
    // ชิปจะคำนวณหักล้างแรงโน้มถ่วงโลก 9.81 ออกให้ในระดับฮาร์ดแวร์ ลดปัญหากราฟไหลเอง
    bno.getEvent(&accelData, Adafruit_BNO055::VECTOR_LINEARACCEL);

    // ดึงค่า Quaternion จากเซนเซอร์โดยตรงเพื่อความแม่นยำในการหมุน 3 มิติ
    imu::Quaternion quatData = bno.getQuat();

    // จัดฟอร์แมตข้อความให้ตรงกับเครื่องมือตรวจจับ Regex ของฝั่ง Python
    // (หมายเหตุ: เนื่องจากใช้ LINEARACCEL ค่าที่ได้จะเป็นหน่วย m/s^2 อยู่แล้ว จึงไม่ต้องหาร 9.81 ในโค้ดนี้)
    char packetBuffer[256];
    snprintf(packetBuffer, sizeof(packetBuffer),
             "Accel[%.2f,%.2f,%.2f] Gyro[%.2f,%.2f,%.2f] Quat[%.4f,%.4f,%.4f,%.4f]",
             accelData.acceleration.x, accelData.acceleration.y, accelData.acceleration.z,
             gyroData.gyro.x, gyroData.gyro.y, gyroData.gyro.z,
             quatData.w(), quatData.x(), quatData.y(), quatData.z());

    // ยิงข้อมูลไร้สายผ่านโปรโตคอล UDP ไปยังคอมพิวเตอร์
    udp.beginPacket(udpAddress, udpPort);
    udp.print(packetBuffer);
    udp.endPacket();
  }
}