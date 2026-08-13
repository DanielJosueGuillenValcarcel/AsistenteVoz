import json
import statistics
from pathlib import Path


RECORDS_PATH = Path(
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


def mean(values):
    if not values:
        return 0.0

    return statistics.mean(
        values
    )


def range_text(values, decimals=3):
    if not values:
        return "-"

    return (
        f"{min(values):.{decimals}f}"
        f" / {mean(values):.{decimals}f}"
        f" / {max(values):.{decimals}f}"
    )


records = []

with RECORDS_PATH.open(
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
    f"TOTAL DE MUESTRAS: "
    f"{len(records)}"
)
print()


for label in LABELS:
    samples = [
        record
        for record in records
        if record["label"] == label
    ]

    ten_detected = sum(
        record["ten_detected"]
        for record in samples
    )

    decisions = []

    phoneme_counts = []
    phonetic_confidences = []

    ten_positive_frames = []
    ten_max_probabilities = []

    ast_speech = []
    ast_non_speech = []
    ast_margins = []

    for record in samples:
        ten_positive_frames.append(
            record["ten_positive_frames"]
        )

        ten_max_probabilities.append(
            record["ten_max_probability"]
        )

        classification = record.get(
            "classification"
        )

        if classification is None:
            decisions.append(
                "no_candidate"
            )
        else:
            decisions.append(
                classification["decision"]
            )

        phonetic = record.get(
            "phonetic"
        )

        if phonetic is not None:
            phoneme_counts.append(
                phonetic["phoneme_count"]
            )

            phonetic_confidences.append(
                phonetic[
                    "phoneme_confidence"
                ]
            )

        ast = record.get(
            "ast"
        )

        if ast is not None:
            ast_speech.append(
                ast["speech_score"]
            )

            ast_non_speech.append(
                ast["non_speech_score"]
            )

            ast_margins.append(
                ast["margin"]
            )

    accept = decisions.count(
        "accept"
    )

    uncertain = decisions.count(
        "uncertain"
    )

    reject = decisions.count(
        "reject"
    )

    no_candidate = decisions.count(
        "no_candidate"
    )

    print(
        "=" * 60
    )

    print(
        label.upper()
    )

    print(
        f"Muestras: "
        f"{len(samples)}"
    )

    print(
        f"TEN detecto: "
        f"{ten_detected}/{len(samples)}"
    )

    print(
        f"CTC ACCEPT: "
        f"{accept}"
    )

    print(
        f"CTC UNCERTAIN: "
        f"{uncertain}"
    )

    print(
        f"CTC REJECT: "
        f"{reject}"
    )

    print(
        f"Sin candidato: "
        f"{no_candidate}"
    )

    print(
        "Fonemas min/prom/max: "
        f"{range_text(phoneme_counts, 2)}"
    )

    print(
        "Confianza fonetica min/prom/max: "
        f"{range_text(phonetic_confidences)}"
    )

    print(
        "TEN frames positivos min/prom/max: "
        f"{range_text(ten_positive_frames, 1)}"
    )

    print(
        "TEN max prob min/prom/max: "
        f"{range_text(ten_max_probabilities)}"
    )

    print(
        "AST Speech min/prom/max: "
        f"{range_text(ast_speech)}"
    )

    print(
        "AST Non-speech min/prom/max: "
        f"{range_text(ast_non_speech)}"
    )

    print(
        "AST margen min/prom/max: "
        f"{range_text(ast_margins)}"
    )

    print()


print(
    "=" * 60
)

print(
    "CASOS IMPORTANTES"
)

print()


for index, record in enumerate(
    records,
    start=1
):
    label = record["label"]

    classification = record.get(
        "classification"
    )

    decision = (
        classification["decision"]
        if classification
        else "no_candidate"
    )

    is_speech = label in (
        "normal",
        "whisper",
    )

    problematic = False

    if is_speech:
        if (
            not record["ten_detected"]
            or decision != "accept"
        ):
            problematic = True

    else:
        if decision == "accept":
            problematic = True

    if not problematic:
        continue

    phonetic = record.get(
        "phonetic"
    )

    ast = record.get(
        "ast"
    )

    phonemes = (
        phonetic["phoneme_count"]
        if phonetic
        else None
    )

    confidence = (
        phonetic["phoneme_confidence"]
        if phonetic
        else None
    )

    speech_score = (
        ast["speech_score"]
        if ast
        else None
    )

    non_speech_score = (
        ast["non_speech_score"]
        if ast
        else None
    )

    print(
        f"{record['id']}"
    )

    print(
        f"  real={label}"
    )

    print(
        f"  TEN={record['ten_detected']}"
    )

    print(
        f"  decision={decision}"
    )

    print(
        f"  fonemas={phonemes}"
    )

    print(
        f"  confianza={confidence}"
    )

    print(
        f"  AST speech={speech_score}"
    )

    print(
        f"  AST non-speech={non_speech_score}"
    )

    print()