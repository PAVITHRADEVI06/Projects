#include <Wire.h>

// Camera Pin Definitions
#define VSYNC_PIN 2
#define PCLK_PIN 3
#define HREF_PIN 4
#define SDA_PIN A4
#define SCL_PIN A5

void setup() {
  Serial.begin(115200);  // Initialize Serial communication
  pinMode(VSYNC_PIN, INPUT);
  pinMode(PCLK_PIN, INPUT);
  pinMode(HREF_PIN, INPUT);

  // Initialize I2C communication (for OV7670 setup)
  Wire.begin();
  
  delay(100);  // Wait for everything to settle
}

void loop() {
  captureImage();
}

void captureImage() {
  Serial.println("Capturing...");
  
  // Wait for the start of a frame (VSYNC)
  while (digitalRead(VSYNC_PIN) == HIGH);  // Wait for VSYNC HIGH
  while (digitalRead(VSYNC_PIN) == LOW);   // Wait for VSYNC LOW
  
  for (int y = 0; y < 120; y++) {  // QQVGA (160x120)
    for (int x = 0; x < 160; x++) {
      while (digitalRead(PCLK_PIN) == HIGH);  // Wait for PCLK low
      while (digitalRead(PCLK_PIN) == LOW);   // Wait for PCLK high

      uint8_t pixel = readByte();  // Read the pixel byte
      Serial.write(pixel);         // Send pixel data over serial

      // Debugging prints (Optional)
      Serial.print("Pixel: ");
      Serial.println(pixel, HEX);

      delayMicroseconds(150);  // Slow down to prevent buffer overflow
    }
  }
  
  Serial.println("Done");  // Notify when capture is complete
}

uint8_t readByte() {
  uint8_t value = 0;
  for (int i = 0; i < 8; i++) {
    value |= (digitalRead(HREF_PIN) << (7 - i));  // Shift in pixel bits from HREF
  }
  return value;
}
