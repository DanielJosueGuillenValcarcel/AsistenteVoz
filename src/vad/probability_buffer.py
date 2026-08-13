from collections import deque


class ProbabilityBuffer:
    def __init__(self, size):
        self.values = deque(maxlen=size)

    def add(self, probability):
        self.values.append(probability)

    def get_values(self):
        return list(self.values)

    def average(self):
        if not self.values:
            return 0.0

        return sum(self.values) / len(self.values)

    def count_above(self, threshold):
        return sum(value >= threshold for value in self.values)

    def is_full(self):
        return len(self.values) == self.values.maxlen