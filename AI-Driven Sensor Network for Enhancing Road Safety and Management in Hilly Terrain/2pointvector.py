import requests
import json
import time
from datetime import datetime

NODEMCU_IP = "192.168.20.154"
PORT = 80

soil_moisture_path = "C:/Users/06dev/PycharmProjects/newweather/soil_moisture_data.geojson"
temperature_path = "C:/Users/06dev/PycharmProjects/newweather/temperature_data.geojson"

SOIL_MOISTURE_THRESHOLD = 800

last_temp_save_time = time.time()

try:
    while True:
        try:
            url = f"http://{NODEMCU_IP}:{PORT}/"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                sensor_data = response.json()
                print("Received data:", sensor_data)

                if sensor_data["SoilMoisture"] > SOIL_MOISTURE_THRESHOLD:
                    soil_latitude = sensor_data["SoilLatitude"]
                    soil_longitude = sensor_data["SoilLongitude"]

                    soil_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [soil_longitude, soil_latitude]
                        },
                        "properties": {
                            "Label": f"Soil Moisture: {sensor_data['SoilMoisture']}"
                        },
                    }

                    soil_geojson = {
                        "type": "FeatureCollection",
                        "features": [soil_feature]
                    }

                    with open(soil_moisture_path, "w") as file:
                        json.dump(soil_geojson, file, indent=4)
                    print(f"Saved soil moisture data to {soil_moisture_path}")

                current_time = time.time()
                if current_time - last_temp_save_time >= 300:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    temperature_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [79.16502, 12.96914]
                        },
                        "properties": {
                            "Label": (
                                f"Timestamp: {timestamp}\n"
                                f"Temperature: {sensor_data['Temperature']}°C\n"
                                f"Humidity: {sensor_data['Humidity']}%\n"
                                f"Pressure: {sensor_data['Pressure']} hPa\n"
                                f"Altitude: {sensor_data['Altitude']} m\n"
                                f"Rain: {sensor_data['Rain']}"
                            )
                        },
                    }

                    temperature_geojson = {
                        "type": "FeatureCollection",
                        "features": [temperature_feature]
                    }

                    with open(temperature_path, "w") as file:
                        json.dump(temperature_geojson, file, indent=4)
                    print(f"Saved temperature data to {temperature_path}")
                    last_temp_save_time = current_time

            else:
                print(f"Error: Received status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Error: Could not fetch data: {e}")
        time.sleep(5)
except KeyboardInterrupt:
    print("Script interrupted by user.")
