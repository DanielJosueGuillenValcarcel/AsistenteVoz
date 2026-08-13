import numpy as np
import torch

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

from .config import SAMPLE_RATE


class AudioEventVerifier:
    SPEECH_LABELS = (
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Whispering",
    )

    NON_SPEECH_LABELS = (
        "Sigh",
        "Breathing",
        "Gasp",
        "Snort",
        "Throat clearing",
        "Cough",
        "Sneeze",
        "Hiccup",
        "Wind noise (microphone)",
        "Computer keyboard",
        "Typing",
    )

    def __init__(self):
        self.model_name = (
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def analyze(self, audio, top_k=10):
        waveform = audio.reshape(-1).astype(
            np.float32
        )

        waveform /= 32768.0

        inputs = self.extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(
                **inputs
            ).logits[0]

        scores = torch.sigmoid(
            logits
        )

        label_scores = {}

        for index in range(
            len(scores)
        ):
            label = (
                self.model.config.id2label[
                    index
                ]
            )

            label_scores[label] = float(
                scores[index].item()
            )

        speech_score = max(
            label_scores.get(
                label,
                0.0
            )
            for label in self.SPEECH_LABELS
        )

        non_speech_score = max(
            label_scores.get(
                label,
                0.0
            )
            for label in self.NON_SPEECH_LABELS
        )

        top_k = min(
            top_k,
            len(scores)
        )

        values, indexes = torch.topk(
            scores,
            k=top_k
        )

        events = []

        for score, index in zip(
            values.tolist(),
            indexes.tolist()
        ):
            events.append(
                {
                    "label": (
                        self.model.config.id2label[
                            index
                        ]
                    ),
                    "score": score,
                }
            )

        return {
            "events": events,
            "label_scores": label_scores,
            "speech_score": speech_score,
            "non_speech_score": non_speech_score,
            "margin": (
                speech_score
                - non_speech_score
            ),
        }