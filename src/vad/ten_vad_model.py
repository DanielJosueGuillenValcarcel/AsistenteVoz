from ten_vad import TenVad


class TenVADModel:
    def __init__(self):
        self.model = None
        self.reset()

    def reset(self):
        self.model = TenVad(
            256,
            0.5
        )

    def predict(self, audio):
        probability, flag = self.model.process(
            audio.flatten()
        )

        return probability, flag