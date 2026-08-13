from collections import deque

import numpy as np


class PreRollBuffer:
    def __init__(self, max_chunks):
        self.chunks = deque(maxlen=max_chunks)

    def add(self, chunk):
        self.chunks.append(chunk.copy())

    def get_audio(self):
        if not self.chunks:
            return np.empty((0, 1), dtype=np.int16)

        return np.concatenate(self.chunks, axis=0)

    def get_chunk_count(self):
        return len(self.chunks)

    def clear(self):
        self.chunks.clear()