import json
from pathlib import Path


PATH = Path(
    "calibration_data_v2/records.jsonl"
)

SPEECH_LABELS = {
    "normal",
    "whisper",
}

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
]


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
print(
    "THRESHOLD | TP ACCEPT | FP ACCEPT | "
    "SPEECH UNCERTAIN | NOISE UNCERTAIN"
)
print("-" * 70)


for threshold in THRESHOLDS:
    tp_accept = 0
    fp_accept = 0

    speech_uncertain = 0
    noise_uncertain = 0

    for record in records:
        classification = record.get(
            "classification"
        )

        if classification is None:
            continue

        decision = classification[
            "decision"
        ]

        phonetic = record.get(
            "phonetic"
        )

        confidence = (
            phonetic[
                "phoneme_confidence"
            ]
            if phonetic
            else 0.0
        )

        if (
            decision == "accept"
            and confidence < threshold
        ):
            decision = "uncertain"

        is_speech = (
            record["label"]
            in SPEECH_LABELS
        )

        if decision == "accept":
            if is_speech:
                tp_accept += 1
            else:
                fp_accept += 1

        elif decision == "uncertain":
            if is_speech:
                speech_uncertain += 1
            else:
                noise_uncertain += 1

    print(
        f"{threshold:9.2f} | "
        f"{tp_accept:9d} | "
        f"{fp_accept:9d} | "
        f"{speech_uncertain:16d} | "
        f"{noise_uncertain:15d}"
    )


print()
print("ACCEPT AFECTADOS")
print("-" * 70)


for record in records:
    classification = record.get(
        "classification"
    )

    phonetic = record.get(
        "phonetic"
    )

    if (
        classification is None
        or phonetic is None
        or classification["decision"]
        != "accept"
    ):
        continue

    confidence = phonetic[
        "phoneme_confidence"
    ]

    if confidence < 0.55:
        print(
            f"{record['label']:10} | "
            f"conf={confidence:.3f} | "
            f"fonemas="
            f"{phonetic['phoneme_count']} | "
            f"{record['id']}"
        )