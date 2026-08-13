import queue
import threading

from .phonetic_verifier import PhoneticVerifier
from ..config import PHONETIC_QUEUE_SIZE


class PhoneticWorker:
    def __init__(self):
        self.queue = queue.Queue(
            maxsize=PHONETIC_QUEUE_SIZE
        )

        self.verifier = PhoneticVerifier()

        self.running = False
        self.thread = None
        self.on_result = None

    def start(self, on_result):
        self.on_result = on_result
        self.running = True

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    def submit(self, audio):
        try:
            self.queue.put_nowait(audio.copy())

        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(audio.copy())
            except queue.Full:
                pass

    def _run(self):
        while self.running:
            try:
                audio = self.queue.get(
                    timeout=0.1
                )

            except queue.Empty:
                continue

            result = self.verifier.analyze(audio)

            if self.on_result:
                self.on_result(
                    result,
                    audio
                )

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join()
            self.thread = None