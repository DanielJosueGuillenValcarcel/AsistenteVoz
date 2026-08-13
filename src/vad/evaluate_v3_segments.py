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


V2_PATH = Path(
    "calibration_data_v2/records.jsonl"
)

V3_PATH = Path(
    "calibration_data_v3/ten_reprocessed.jsonl"
)


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


v2_records = {}

with V2_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        v2_records[
            record["id"]
        ] = record


v3_records = []

with V3_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    for line in file:
        line = line.strip()

        if line:
            v3_records.append(
                json.loads(line)
            )


summary = {}


for record in v3_records:
    label = record["label"]

    if label not in summary:
        summary[label] = {
            "samples": 0,
            "segments": 0,
            "accept_segments": 0,
            "uncertain_segments": 0,
            "reject_segments": 0,
            "samples_with_accept": 0,
            "samples_with_uncertain": 0,
            "samples_only_reject": 0,
        }

    data = summary[label]

    data["samples"] += 1

    source = v2_records[
        record["id"]
    ]

    audio = load_wav(
        source["window_audio_path"]
    )

    chunks = make_chunks(
        audio
    )

    sample_decisions = []

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

        phonetic = verifier.analyze(
            segment_audio
        )

        classification = classifier.classify(
            phonetic
        )

        decision = classification[
            "decision"
        ].value

        sample_decisions.append(
            decision
        )

        data["segments"] += 1

        if decision == "accept":
            data[
                "accept_segments"
            ] += 1

        elif decision == "uncertain":
            data[
                "uncertain_segments"
            ] += 1

        else:
            data[
                "reject_segments"
            ] += 1

    if "accept" in sample_decisions:
        data[
            "samples_with_accept"
        ] += 1

    elif "uncertain" in sample_decisions:
        data[
            "samples_with_uncertain"
        ] += 1

    else:
        data[
            "samples_only_reject"
        ] += 1


print()
print(
    "RESULTADOS POR CLASE"
)

print("=" * 95)

print(
    "CLASE      | MUESTRAS | SEG | "
    "SEG A/U/R | MUESTRA ACCEPT | "
    "MUESTRA UNCERTAIN | SOLO REJECT"
)

print("-" * 95)


for label, data in summary.items():
    print(
        f"{label:11}| "
        f"{data['samples']:8d} | "
        f"{data['segments']:3d} | "
        f"{data['accept_segments']:2d}/"
        f"{data['uncertain_segments']:2d}/"
        f"{data['reject_segments']:2d} | "
        f"{data['samples_with_accept']:14d} | "
        f"{data['samples_with_uncertain']:17d} | "
        f"{data['samples_only_reject']:11d}"
    )