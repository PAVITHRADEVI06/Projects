#include <WiFi.h>
#include <DHT.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP085.h>

// Wi-Fi credentials
const char *ssid = "Pavi";       // Replace with your Wi-Fi SSID
const char *password = "alohomora";  // Replace with your Wi-Fi password

// DHT and BMP180 sensor settings
#define DHT_PIN 2 // GPIO2 for DHT sensor
#define DHT_TYPE DHT11
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP085 bmp;

// Pin definitions
#define RAIN_SENSOR_PIN 34 // GPIO34 for Rain sensor
#define SOIL_SENSOR_PIN 35 // GPIO35 for Soil sensor

WiFiServer server(80); // Create a server on port 80

void setup() {
  Serial.begin(115200); // For debugging
  dht.begin();
  if (!bmp.begin()) {
    Serial.println("BMP180 initialization failed!");
    while (1);
  }

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin(); // Start the server
}

void loop() {
  WiFiClient client = server.available(); // Check for incoming clients

  if (client) {
    Serial.println("New client connected!");
    while (client.connected()) {
      if (client.available()) {
        String request = client.readStringUntil('\r');
        client.flush();

        // Read sensor data
        int rainSensorValue = analogRead(RAIN_SENSOR_PIN);
        int soilMoistureValue = analogRead(SOIL_SENSOR_PIN);
        float temperature = dht.readTemperature();
        float humidity = dht.readHumidity();
        float pressure = bmp.readPressure() / 100.0F;
        float altitude = bmp.readAltitude(1013.25);

        // Ensure valid readings
        if (isnan(temperature) || isnan(humidity)) {
          temperature = 0;
          humidity = 0;
        }

        // Construct JSON string
        String json = "{";
        json += "\"Rain\":" + String(rainSensorValue) + ",";
        json += "\"SoilMoisture\":" + String(soilMoistureValue) + ",";
        json += "\"Temperature\":" + String(temperature, 2) + ",";
        json += "\"Humidity\":" + String(humidity, 2) + ",";
        json += "\"Pressure\":" + String(pressure, 2) + ",";
        json += "\"Altitude\":" + String(altitude, 2) + ",";
        json += "\"SoilLatitude\": 12.97078,"; // Replace with actual latitude
        json += "\"SoilLongitude\": 79.15869"; // Replace with actual longitude
        json += "}";

        // Send the response
        client.println("HTTP/1.1 200 OK");
        client.println("Content-Type: application/json");
        client.println("Connection: close");
        client.println();
        client.println(json);

        Serial.println("Sent data to client:");
        Serial.println(json);
        break;
      }
    }
    client.stop();
    Serial.println("Client disconnected.");
  }
}
