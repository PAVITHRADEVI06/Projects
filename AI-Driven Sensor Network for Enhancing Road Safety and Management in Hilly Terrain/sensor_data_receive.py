import requests
import json
import time
from datetime import datetime

NODEMCU_IP = "192.168.20.154"
PORT = 80

LATITUDE = 12.96914
LONGITUDE = 79.16502

output_geojson_path = "sensor_data.geojson"

hourly_backup_path = "sensor_data_backup.geojson"

def save_latest_to_geojson(output_path, sensor_data):
    label = (
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Rain: {sensor_data['Rain']}\n"
        f"Soil Moisture: {sensor_data['SoilMoisture']}\n"
        f"Temperature: {sensor_data['Temperature']}°C\n"
        f"Humidity: {sensor_data['Humidity']}%\n"
        f"Pressure: {sensor_data['Pressure']} hPa\n"
        f"Altitude: {sensor_data['Altitude']} m"
    )

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [LONGITUDE, LATITUDE]
        },
        "properties": {
            "Label": label
        },
    }

    geojson_data = {
        "type": "FeatureCollection",
        "features": [feature]
    }

    with open(output_path, "w") as geojson_file:
        json.dump(geojson_data, geojson_file, indent=4)

    print(f"Latest data saved to GeoJSON file: {output_path}")

def save_hourly_backup(output_path, backup_path):
    try:
        with open(output_path, "r") as geojson_file:
            data = geojson_file.read()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path.replace(".geojson", f"_{timestamp}.geojson")

        with open(backup_file, "w") as backup:
            backup.write(data)

        print(f"Backup saved: {backup_file}")
    except FileNotFoundError:
        print(f"No GeoJSON file found at {output_path} to back up.")

try:
    last_backup_time = time.time()
    while True:
        try:
            url = f"http://{NODEMCU_IP}:{PORT}/"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                sensor_data = response.json()
                print("Received data:", sensor_data)

                save_latest_to_geojson(output_geojson_path, sensor_data)

                current_time = time.time()
                if current_time - last_backup_time >= 300:
                    save_hourly_backup(output_geojson_path, hourly_backup_path)
                    last_backup_time = current_time
            else:
                print(f"Error: Received status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Error: Could not fetch data: {e}")
        time.sleep(1)
except KeyboardInterrupt:
    print("Script interrupted by user.")
