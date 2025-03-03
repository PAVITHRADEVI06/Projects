"""import serial
import numpy as np
from PIL import Image
import time

# Set up serial communication (adjust COM port and baud rate)
ser = serial.Serial('COM12', 500000, timeout=2)  # Change COM port as needed

# Define frame dimensions
width = 320
height = 240
frame_size = width * height * 2  # RGB565 format (2 bytes per pixel)

def rgb565_to_rgb888(rgb565):
    # Convert RGB565 format to RGB888
    r = (rgb565 >> 11) & 0x1F
    g = (rgb565 >> 5) & 0x3F
    b = rgb565 & 0x1F
    return (r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)

while True:
    print("Waiting for Vsync...")

    # Wait until "Vsync" is detected
    line = ser.readline().decode(errors='ignore').strip()
    while "Vsync" not in line:
        line = ser.readline().decode(errors='ignore').strip()

    print("Vsync detected! Capturing frame...")

    frame_buffer = bytearray()

    # Collect data until the next "Vsync" appears
    while True:
        chunk = ser.read(frame_size - len(frame_buffer))
        frame_buffer.extend(chunk)

        # Check if we received a full frame
        if len(frame_buffer) >= frame_size:
            break

        # Also check for "Vsync" during data collection (next frame start)
        line = ser.readline().decode(errors='ignore').strip()
        if "Vsync" in line:
            print("Warning: Incomplete frame received!")
            frame_buffer.clear()
            continue

    print("Full frame received! Processing...")

    # Convert raw frame data to a numpy array of RGB565 values
    raw_frame = np.frombuffer(frame_buffer, dtype=np.uint16).reshape((height, width))

    # Convert to RGB888
    rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            rgb565 = raw_frame[y, x]
            rgb_frame[y, x] = rgb565_to_rgb888(rgb565)

    # Convert the numpy array to an image
    img = Image.fromarray(rgb_frame)
    img.save("frame.jpg", "JPEG")

    print("Image saved as frame.jpg. Waiting for next frame...\n")

    # Wait for 5 minutes before capturing the next frame
    time.sleep(300)

import serial
import numpy as np
from PIL import Image
import time
import re

# Serial Configuration
SERIAL_PORT = 'COM12'  # Change this to your port
BAUD_RATE = 500000
WIDTH = 320  # OV7670 resolution
HEIGHT = 240
FRAME_SIZE = WIDTH * HEIGHT * 2  # RGB565 (2 bytes per pixel)

# Open Serial Connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)


# Function to capture a single frame
def capture_frame():
    frame_buffer = bytearray()
    collecting = False

    while True:
        line = ser.readline().decode(errors='ignore').strip()

        if "Vsync" in line:
            if collecting:
                break  # Stop collecting at next Vsync
            collecting = True  # Start collecting
            frame_buffer.clear()

        if collecting and re.match(r"Frame \d+", line):
            continue  # Ignore frame number lines

        if collecting:
            frame_buffer.extend(ser.read(FRAME_SIZE - len(frame_buffer)))
            if len(frame_buffer) >= FRAME_SIZE:
                break

    if len(frame_buffer) != FRAME_SIZE:
        print("Warning: Incomplete frame received!")
        return None

    return frame_buffer


# Function to save frame as JPEG
def save_rgb565_as_jpeg(frame_buffer, filename="frame.jpg"):
    raw_frame = np.frombuffer(frame_buffer, dtype=np.uint16).reshape((HEIGHT, WIDTH))

    # Convert to RGB888 (PIL does not support RGB565 natively)
    rgb_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixel = raw_frame[y, x]
            r = (pixel >> 11) & 0x1F
            g = (pixel >> 5) & 0x3F
            b = pixel & 0x1F
            rgb_frame[y, x] = [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]

    img = Image.fromarray(rgb_frame, 'RGB')
    img.save(filename, "JPEG")
    print(f"Image saved as {filename}")


while True:
    print("Waiting for a new frame...")
    frame_data = capture_frame()
    if frame_data:
        save_rgb565_as_jpeg(frame_data)
    time.sleep(300)  # Wait for 5 minutes before next capture"""

import serial
import struct
from PIL import Image
import numpy as np

SERIAL_PORT = 'COMx'
BAUD_RATE = 500000
LINE_LENGTH = 320
LINE_COUNT = 240
PIXEL_SIZE = 2

START_MARKER = 0xFF
END_MARKER = 0xFE


def read_frame_from_serial():
    frame_data = bytearray()
    is_reading = False

    while True:
        byte = ser.read(1)
        if not byte:
            continue

        byte = ord(byte)
        if byte == START_MARKER and not is_reading:
            is_reading = True
            frame_data = bytearray()
        elif byte == END_MARKER and is_reading:
            break
        elif is_reading:
            frame_data.append(byte)

    return frame_data


def rgb565_to_rgb888(rgb565_data):
    rgb888_data = []
    for i in range(0, len(rgb565_data), 2):
        rgb565 = struct.unpack('H', rgb565_data[i:i + 2])[0]
        r = ((rgb565 >> 11) & 0x1F) << 3
        g = ((rgb565 >> 5) & 0x3F) << 2
        b = (rgb565 & 0x1F) << 3
        rgb888_data.extend([r, g, b])

    return np.array(rgb888_data, dtype=np.uint8).reshape((LINE_COUNT, LINE_LENGTH, 3))


def save_image(rgb_data, filename="output_image.jpg"):
    image = Image.fromarray(rgb_data)
    image.save(filename)
    print(f"Image saved as {filename}")


if __name__ == "__main__":
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Waiting for frame data...")

    while True:
        if ser.in_waiting > 0:
            command = ser.read(1)
            if command == b'\x01':
                print("Receiving new frame...")
                frame_data = read_frame_from_serial()

                rgb_data = rgb565_to_rgb888(frame_data)

                save_image(rgb_data)
