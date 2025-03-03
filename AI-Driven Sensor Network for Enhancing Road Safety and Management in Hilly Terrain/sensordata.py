import serial
import json
import time

# Configure Bluetooth connection
try:
    bluetooth = serial.Serial("COM9", 9600, timeout=1)  # Adjust "COM8" to match your Bluetooth port
    print("Connected to Bluetooth!")
except serial.SerialException as e:
    print(f"Error connecting to Bluetooth: {e}")
    exit()

# Path to save the GeoJSON file
output_geojson_path = "C:/Users/06dev/PycharmProjects/newweather/sensor_data.geojson"


# Function to append GeoJSON data
def append_to_geojson(output_path, geojson_feature):
    """
    Append a GeoJSON feature to a GeoJSON file.

    Args:
        output_path (str): Path to save the GeoJSON file.
        geojson_feature (dict): The GeoJSON feature to append.
    """
    try:
        with open(output_path, "r") as geojson_file:
            geojson_data = json.load(geojson_file)
    except FileNotFoundError:
        geojson_data = {"type": "FeatureCollection", "features": []}

    geojson_data["features"].append(geojson_feature)

    with open(output_path, "w") as geojson_file:
        json.dump(geojson_data, geojson_file, indent=4)

    print(f"Data appended to GeoJSON file at: {output_path}")


try:
    while True:
        # Read data from Bluetooth
        data = bluetooth.readline().decode('utf-8').strip()
        if data:
            print("Received Data: ", data)
            try:
                # Parse the incoming data as JSON
                geojson_feature = json.loads(data)

                # Append the parsed data to the GeoJSON file
                append_to_geojson(output_geojson_path, geojson_feature)
            except json.JSONDecodeError:
                print("Error: Received data is not valid JSON!")
except KeyboardInterrupt:
    print("Process interrupted by user.")
finally:
    if bluetooth.is_open:
        bluetooth.close()
        print("Bluetooth connection closed.")
