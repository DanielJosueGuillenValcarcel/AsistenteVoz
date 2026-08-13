import json
import wave

from pathlib import Path

import numpy as np

from .config import (
    CHUNK_SIZE,
    PRE_ROLL_CHUNKS,
)

from .phonetic_verifier import PhoneticVerifier
from .speech_evidence_classifier import (
    SpeechEvidenceClassifier,
)
from .audio_event_verifier import (
    AudioEventVerifier,
)


V2_PATH = Path(
    "calibration_data_v2/records.jsonl"
)

V3_PATH = Path(
    "calibration_data_v3/ten_reprocessed.jsonl"
)


phonetic_verifier = PhoneticVerifier()
classifier = SpeechEvidenceClassifier()
ast = AudioEventVerifier()


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
        chunks.append(
            audio[
                start:start + CHUNK_SIZE
            ].copy()
        )

    return chunks


v2 = {}

with V2_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    for line in file:
        if not line.strip():
            continue

        record = json.loads(line)

        v2[
            record["id"]
        ] = record


v3 = []

with V3_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    for line in file:
        if line.strip():
            v3.append(
                json.loads(line)
            )


for record in v3:
    source = v2[
        record["id"]
    ]

    audio = load_wav(
        source["window_audio_path"]
    )

    chunks = make_chunks(
        audio
    )

    best_uncertain = None
    best_rank = None

    has_accept = False

    for segment in record["segments"]:
        start = max(
            0,
            segment["start"]
            - PRE_ROLL_CHUNKS
        )

        end = segment["end"]

        selected = chunks[
            start:end
        ]

        if not selected:
            continue

        segment_audio = np.concatenate(
            selected,
            axis=0
        )

        phonetic = phonetic_verifier.analyze(
            segment_audio
        )

        classification = classifier.classify(
            phonetic
        )

        decision = classification[
            "decision"
        ].value

        if decision == "accept":
            has_accept = True
            break

        if decision != "uncertain":
            continue

        rank = (
            phonetic["phoneme_count"],
            phonetic["phoneme_confidence"],
        )

        if (
            best_rank is None
            or rank > best_rank
        ):
            best_rank = rank

            best_uncertain = (
                segment_audio,
                phonetic,
            )

    if has_accept:
        continue

    if best_uncertain is None:
        continue

    segment_audio, phonetic = (
        best_uncertain
    )

    result = ast.analyze(
        segment_audio,
        top_k=10
    )

    print()
    print("=" * 75)

    print(
        f"REAL: "
        f"{record['label'].upper()}"
    )

    print(
        f"ID: {record['id']}"
    )

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

    print()

    for event in result["events"]:
        print(
            f"{event['label']}: "
            f"{event['score']:.3f}"
        )