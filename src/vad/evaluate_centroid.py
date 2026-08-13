import json
from pathlib import Path

from .speech_evidence_classifier import (
    SpeechEvidenceClassifier,
)


PATH = Path(
    "calibration_data_v2/records.jsonl"
)

LABELS = [
    "normal",
    "whisper",
    "sigh",
    "blow",
    "breathing",
    "keyboard",
]

THRESHOLDS = [
    600,
    650,
    700,
    750,
    800,
    850,
]

classifier = SpeechEvidenceClassifier()

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


print()
print("CENTROIDE POR CLASE")
print("=" * 70)


for label in LABELS:
    values = []

    for record in records:
        if record["label"] != label:
            continue

        articulation = record.get(
            "articulation"
        )

        if articulation is None:
            continue

        values.append(
            articulation["centroid_std"]
        )

    if not values:
        print(
            f"{label:10} | sin candidatos"
        )

        continue

    average = (
        sum(values) / len(values)
    )

    print(
        f"{label:10} | "
        f"n={len(values):2d} | "
        f"min={min(values):7.1f} | "
        f"prom={average:7.1f} | "
        f"max={max(values):7.1f}"
    )


print()
print("PRUEBA DE THRESHOLDS")
print("=" * 70)

print(
    "THRESHOLD | SPEECH <= | NOISE <= | "
    "SPEECH > | NOISE >"
)

print("-" * 70)


for threshold in THRESHOLDS:
    speech_low = 0
    noise_low = 0
    speech_high = 0
    noise_high = 0

    for record in records:
        articulation = record.get(
            "articulation"
        )

        if articulation is None:
            continue

        centroid = articulation[
            "centroid_std"
        ]

        is_speech = record["label"] in (
            "normal",
            "whisper",
        )

        if centroid <= threshold:
            if is_speech:
                speech_low += 1
            else:
                noise_low += 1
        else:
            if is_speech:
                speech_high += 1
            else:
                noise_high += 1

    print(
        f"{threshold:9.0f} | "
        f"{speech_low:9d} | "
        f"{noise_low:8d} | "
        f"{speech_high:8d} | "
        f"{noise_high:7d}"
    )


print()
print("CASOS NO-ACCEPT CON CENTROIDE")
print("=" * 70)


for record in records:
    phonetic = record.get(
        "phonetic"
    )

    articulation = record.get(
        "articulation"
    )

    if (
        phonetic is None
        or articulation is None
    ):
        continue

    classification = classifier.classify(
        phonetic
    )

    if (
        classification["decision"].value
        == "accept"
    ):
        continue

    ast = record.get(
        "ast"
    )

    print(
        f"{record['label']:10} | "
        f"{classification['decision'].value:9} | "
        f"centroid={articulation['centroid_std']:7.1f} | "
        f"phonemes={phonetic['phoneme_count']:2d} | "
        f"conf={phonetic['phoneme_confidence']:.3f} | "
        f"AST_S={ast['speech_score']:.3f} | "
        f"AST_NS={ast['non_speech_score']:.3f}"
    )