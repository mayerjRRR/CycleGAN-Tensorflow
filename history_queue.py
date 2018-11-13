import random

import numpy as np


class HistoryQueue(object):
    def __init__(self, shape=[128,128,3], size=50):
        self._size = size
        self._shape = shape
        self._count = 0
        self._queue = []

    def query(self, image):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if self._size == 0:
            return image
        if self._count < self._size:
            self._count += 1
            self._queue.append(image)
            return image

        p = random.random()
        if p > 0.5:
            idx = random.randrange(0, self._size)
            ret = self._queue[idx]
            self._queue[idx] = image
            return ret
        else:
            return image