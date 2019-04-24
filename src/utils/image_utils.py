import numpy as np
import cv2
import os

video_formats = [".mp4"]
image_formats = [".jpg", ".jpeg",".png"]

def is_directory(path):
    return os.path.isdir(path)

def is_video(file_path):
    file_ending = get_file_format(file_path)
    return file_ending in video_formats

def is_image(file_path):
    file_ending = get_file_format(file_path)
    return file_ending in image_formats

def get_file_format(file_path):
    directory_name, file_name = os.path.split(file_path)
    input_name, file_ending = os.path.splitext(file_name)
    return file_ending

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

def load_all_video_float_frames(file_path):
    video_capture = cv2.VideoCapture(file_path)
    frames = []
    ret, frame = video_capture.read()
    while ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(uint8_to_float(rgb_frame))
        ret, frame = video_capture.read()
    video_capture.release()
    return frames

def iterate_all_video_float_frames(file_path):
    video_capture = cv2.VideoCapture(file_path)
    ret, frame = video_capture.read()
    while ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield rgb_frame
        ret, frame = video_capture.read()
    video_capture.release()

def save_frames_to_video(frames, file_name, framerate=30.0):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    frame_shape = np.array(frames[0]).shape
    video_writer = cv2.VideoWriter(file_name, fourcc, framerate, (frame_shape[1], frame_shape[0]))
    for frame in frames:
        uint8_frame = float_to_unit8(frame)
        bgr_frame = cv2.cvtColor(uint8_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
    video_writer.release()