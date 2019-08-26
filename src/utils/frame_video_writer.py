import cv2
import os

class FrameVideoWriter:
    def __init__(self, output_name):
        self.image_stem, self.image_format = os.path.splitext(output_name)
        if not os.path.isdir(self.image_stem):
            os.mkdir(self.image_stem)
        self.frame_counter = 0

    def release(self):
        pass

    def write(self, frame):
        cv2.imwrite(os.path.join(self.image_stem, f"{self.frame_counter:03d}{self.image_format}"), frame)
        self.frame_counter += 1