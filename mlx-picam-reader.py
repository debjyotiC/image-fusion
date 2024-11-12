import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import cv2
from picamzero import Camera

# Set up the I2C connection and the MLX90640 thermal camera
i2c = busio.I2C(board.SCL, board.SDA)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ  # Set a feasible refresh rate

# Define data and timing variables
frame = np.zeros((24 * 32,))
max_retries = 5
retry_count = 0
success = False

# Attempt to capture a single thermal frame
while retry_count < max_retries:
    try:
        mlx.getFrame(frame)
        data_array = np.reshape(frame, (24, 32))
        success = True
        break
    except (ValueError, RuntimeError):
        retry_count += 1
        if retry_count >= max_retries:
            print(f"Failed to capture frame after {max_retries} retries.")

# If successful, upscale, flip, and save the thermal image using OpenCV
if success:
    # Upscale thermal image to 320x240 using cubic interpolation
    data_array_upscaled = cv2.resize(data_array, (320, 240), interpolation=cv2.INTER_CUBIC)

    # Flip the image on the y-axis
    data_array_flipped = cv2.flip(data_array_upscaled, 1)

    # Normalize data for better color mapping
    normalized_data = cv2.normalize(data_array_flipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a colormap for visualization
    colored_image = cv2.applyColorMap(normalized_data, cv2.COLORMAP_INFERNO)

    # Save the colored thermal image
    cv2.imwrite('thermal_image_320x240.png', colored_image)
    print("Thermal image saved as 'thermal_image_320x240.png'.")
else:
    print("Thermal image capture failed.")

# Capture an additional image using picamzero
cam = Camera()
cam.still_size = (320, 240)
cam.take_photo("still_image.jpg")
cam.stop_preview()
print("Camera image saved as 'still_image.jpg'.")
