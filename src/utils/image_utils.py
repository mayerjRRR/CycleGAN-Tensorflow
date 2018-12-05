import numpy as np
import cv2

def uint8_to_float(image):
    float_image = np.array(image, dtype=np.float32)
    scaled_image = ((float_image/255) * 2) - 1
    return scaled_image

def float_to_unit8(image):
    float_image = np.array(image, dtype=np.float32)
    scaled_image = ((float_image + 1) / 2) * 255
    uint8_image = scaled_image.astype(np.uint8)
    return uint8_image

def load_image(image_path):
    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def load_float_image(image_path):
    uint8_image = load_image(image_path)
    float_image = uint8_to_float(uint8_image)
    return float_image

def save_float_image(float_image, file_name):
    uint8_image = float_to_unit8(float_image)
    bgr_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, bgr_image)