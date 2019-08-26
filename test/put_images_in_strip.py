import argparse
from glob import glob
import os
import cv2
import numpy as np

from src.utils.utils import get_subdirs

parser = argparse.ArgumentParser(description="Dataset Image Exporter")
parser.add_argument('--data', type=str, default=None, help='Location of the image data')
parser.add_argument('--inplace', type=bool, default=False, help='wether to embed the differences into the strip')
args, _ = parser.parse_known_args()


subdirs = get_subdirs(args.data)

big_strip = None
big_diff_strip = None

for current_directory in subdirs:
    images = glob(os.path.join(current_directory, '*.png'))
    if len(images) == 0:
        images = glob(os.path.join(current_directory, '*.jpg'))
    strip = None
    strip_diff = None
    last = None
    for image in images:
        bgr = cv2.imread(image,cv2.IMREAD_COLOR)
        if last is None:
            last = bgr
        else:
            diff = np.abs(bgr.astype(np.int)-last.astype(np.int))*4
            if strip_diff is None:
                strip_diff = diff
            else:
                strip_diff = np.concatenate((strip_diff, diff), axis=0)
            last = bgr
        if strip is None:
            strip = np.ndarray(shape=(0,)+bgr.shape[1:], dtype=bgr.dtype)
        else:
            strip = np.concatenate((strip, bgr), axis=0)
            if args.inplace and diff is not None:
                strip = np.concatenate((strip, diff), axis=0)

    save_name = os.path.join(current_directory, "strip.png")
    save_name_diff = os.path.join(current_directory, "strip_diff.png")
    #cv2.imwrite(save_name, strip)
    #cv2.imwrite(save_name_diff, strip_diff)

    if big_strip is None:
        big_strip = strip
    else:
        big_strip = np.concatenate((big_strip, strip), axis=1)

    if big_diff_strip is None:
        big_diff_strip = strip_diff
    else:
        big_diff_strip = np.concatenate((big_diff_strip, strip_diff), axis=1)

save_name = os.path.join(args.data, "strip.png")
save_name_diff = os.path.join(args.data, "strip_diff.png")
cv2.imwrite(save_name, big_strip)
cv2.imwrite(save_name_diff, big_diff_strip)

