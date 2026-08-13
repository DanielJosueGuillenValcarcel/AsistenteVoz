import json
import wave

import numpy as np

from pathlib import Path

from .config import (
    CHUNK_SIZE,
    PRE_ROLL_CHUNKS,
    CANDIDATE_END_SILENCE_CHUNKS,
)

from .ten_vad_model import TenVADModel
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


phonetic_verifier = PhoneticVerifier()
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


def make_chunks(audio):
    chunks = []

    for start in range(
        0,
        len(audio) - CHUNK_SIZE + 1,
        CHUNK_SIZE
    ):
        end = start + CHUNK_SIZE

        chunks.append(
            audio[start:end].copy()
        )

    return chunks


def run_ten(chunks):
    vad = TenVADModel()

    probabilities = []
    flags = []

    for chunk in chunks:
        probability, flag = vad.predict(
            chunk
        )

        probabilities.append(
            probability
        )

        flags.append(
            flag
        )

    return probabilities, flags


def find_segments(flags):
    segments = []

    active = False
    start_index = 0
    silence = 0
    positive_frames = 0

    for index, flag in enumerate(flags):
        if not active:
            if flag == 1:
                active = True
                start_index = index
                silence = 0
                positive_frames = 1

            continue

        if flag == 1:
            silence = 0
            positive_frames += 1

        else:
            silence += 1

        if (
            silence
            >= CANDIDATE_END_SILENCE_CHUNKS
        ):
            segments.append(
                (
                    start_index,
                    index + 1,
                    positive_frames
                )
            )

            active = False
            silence = 0
            positive_frames = 0

    if active:
        segments.append(
            (
                start_index,
                len(flags),
                positive_frames
            )
        )

    return segments


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

    audio = load_wav(
        record["window_audio_path"]
    )

    chunks = make_chunks(
        audio
    )

    probabilities, flags = run_ten(
        chunks
    )

    segments = find_segments(
        flags
    )

    print()
    print("=" * 80)

    print(
        f"REAL: {record['label'].upper()}"
    )

    print(
        f"ID: {record['id']}"
    )

    print(
        f"SEGMENTOS TEN: {len(segments)}"
    )

    print()

    for number, segment in enumerate(
        segments,
        start=1
    ):
        start_index, end_index, positives = segment

        pre_start = max(
            0,
            start_index - PRE_ROLL_CHUNKS
        )

        selected = chunks[
            pre_start:end_index
        ]

        segment_audio = np.concatenate(
            selected,
            axis=0
        )

        result = phonetic_verifier.analyze(
            segment_audio
        )

        classification = classifier.classify(
            result
        )

        segment_probabilities = probabilities[
            start_index:end_index
        ]

        duration = (
            len(segment_audio)
            / 16000
        )

        print(
            f"SEGMENTO {number}"
        )

        print(
            f"  chunks: "
            f"{start_index}-{end_index}"
        )

        print(
            f"  positivos: "
            f"{positives}"
        )

        print(
            f"  max TEN: "
            f"{max(segment_probabilities):.3f}"
        )

        print(
            f"  duracion: "
            f"{duration:.3f}s"
        )

        print(
            f"  fonemas: "
            f"{result['phonemes']}"
        )

        print(
            f"  cantidad: "
            f"{result['phoneme_count']}"
        )

        print(
            f"  confianza: "
            f"{result['phoneme_confidence']:.3f}"
        )

        print(
            f"  decision: "
            f"{classification['decision'].value.upper()}"
        )

        print()