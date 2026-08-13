import json
import wave

import numpy as np

from pathlib import Path

from .phonetic_verifier import PhoneticVerifier
from .speech_evidence_classifier import (
    SpeechEvidenceClassifier,
    SpeechDecision,
)


PATH = Path(
    "calibration_data_v2/records.jsonl"
)

SPEECH_LABELS = {
    "normal",
    "whisper",
}

verifier = PhoneticVerifier()
classifier = SpeechEvidenceClassifier()


def load_wav(path):
    with wave.open(
        str(path),
        "rb"
    ) as wav:
        frames = wav.readframes(
            wav.getnframes()
        )

    audio = np.frombuffer(
        frames,
        dtype=np.int16
    )

    return audio.reshape(
        -1,
        1
    )


records = []

with PATH.open(
    "r",
    encoding="utf-8"
) as file:
    for line in file:
        line = line.strip()

        if line:
            records.append(
                json.loads(line)
            )


for record in records:
    phonetic = record.get(
        "phonetic"
    )

    if phonetic is None:
        continue

    current = classifier.classify(
        phonetic
    )

    is_speech = (
        record["label"]
        in SPEECH_LABELS
    )

    interesting = (
        (
            is_speech
            and current["decision"]
            != SpeechDecision.ACCEPT
        )
        or
        (
            not is_speech
            and current["decision"]
            == SpeechDecision.UNCERTAIN
        )
    )

    if not interesting:
        continue

    window_audio = load_wav(
        record["window_audio_path"]
    )

    full_result = verifier.analyze(
        window_audio
    )

    full_classification = classifier.classify(
        full_result
    )

    print("=" * 75)

    print(
        f"REAL: {record['label'].upper()}"
    )

    print(
        f"ID: {record['id']}"
    )

    print()

    print("CANDIDATO TEN")

    print(
        f"Fonemas: "
        f"{phonetic['phonemes']}"
    )

    print(
        f"Cantidad: "
        f"{phonetic['phoneme_count']}"
    )

    print(
        f"Confianza: "
        f"{phonetic['phoneme_confidence']:.3f}"
    )

    print(
        f"Decision: "
        f"{current['decision'].value.upper()}"
    )

    print()

    print("VENTANA COMPLETA")

    print(
        f"Fonemas: "
        f"{full_result['phonemes']}"
    )

    print(
        f"Cantidad: "
        f"{full_result['phoneme_count']}"
    )

    print(
        f"Confianza: "
        f"{full_result['phoneme_confidence']:.3f}"
    )

    print(
        f"Decision: "
        f"{full_classification['decision'].value.upper()}"
    )

    print()