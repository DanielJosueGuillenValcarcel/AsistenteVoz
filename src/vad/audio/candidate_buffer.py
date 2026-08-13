import numpy as np


class CandidateBuffer:
    def __init__(self, end_silence_chunks, max_samples):
        self.end_silence_chunks = end_silence_chunks
        self.max_samples = max_samples
        self.active = False
        self.chunks = []
        self.silence_chunks = 0
        self.samples = 0

    def start(self, pre_roll_audio):
        self.active = True
        self.chunks = [pre_roll_audio.copy()]
        self.silence_chunks = 0
        self.samples = len(pre_roll_audio)

    def add(self, chunk, speech_flag):
        if not self.active:
            return None

        self.chunks.append(chunk.copy())
        self.samples += len(chunk)

        if speech_flag:
            self.silence_chunks = 0
        else:
            self.silence_chunks += 1

        if self.silence_chunks >= self.end_silence_chunks:
            return self._finish("silence")

        if self.samples >= self.max_samples:
            return self._finish("max_duration")

        return None

    def _finish(self, reason):
        audio = np.concatenate(self.chunks, axis=0)
        self.reset()
        return audio, reason

    def reset(self):
        self.active = False
        self.chunks = []
        self.silence_chunks = 0
        self.samples = 0