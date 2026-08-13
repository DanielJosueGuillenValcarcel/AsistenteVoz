import torch
from silero_vad import load_silero_vad

from .config import SAMPLE_RATE


class VADModel:
    def __init__(self):
        self.model = load_silero_vad()

    def predict(self, audio):
        tensor = torch.from_numpy(audio.flatten())
        probability = self.model(tensor, SAMPLE_RATE).item()
        return probability