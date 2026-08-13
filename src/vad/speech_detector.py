class SpeechDetector:
    def __init__(
        self,
        start_weak_chunks,
        start_strong_chunks
    ):
        self.start_weak_chunks = start_weak_chunks
        self.start_strong_chunks = start_strong_chunks

    def is_speech_start(self, weak, strong):
        return (
            weak >= self.start_weak_chunks
            or strong >= self.start_strong_chunks
        )