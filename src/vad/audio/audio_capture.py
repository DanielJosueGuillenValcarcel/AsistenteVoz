import queue

import sounddevice as sd

from ..config import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    CHANNELS,
    DTYPE,
    AUDIO_QUEUE_SIZE,
)


class AudioCapture:
    def __init__(self):
        self.stream = None
        self.queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)

    def _callback(self, indata, frames, time, status):
        chunk = indata.copy()

        try:
            self.queue.put_nowait(chunk)

        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(chunk)
            except queue.Full:
                pass

    def start(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback
        )

        self.stream.start()

    def read(self, timeout=0.1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def clear(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break