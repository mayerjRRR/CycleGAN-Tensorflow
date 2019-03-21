import random


class HistoryQueue(object):
    def __init__(self, size=50):
        self.buffer_size = size
        self.current_size = 0
        self.buffer = []

    def query(self, input_element):
        if self.buffer_size == 0:
            return input_element

        if self.current_size < self.buffer_size:
            return self.fill_buffer(input_element)

        p = random.random()
        if p > 0.5:
            return self.replace_random_element(input_element)
        else:
            return input_element

    def replace_random_element(self, input_element):
        random_index = random.randrange(0, self.buffer_size)
        history_element = self.buffer[random_index]
        self.buffer[random_index] = input_element
        return history_element

    def fill_buffer(self, input_element):
        self.current_size += 1
        self.buffer.append(input_element)
        return input_element
