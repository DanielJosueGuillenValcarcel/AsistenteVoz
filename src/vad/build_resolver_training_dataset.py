import csv
from pathlib import Path


SEGMENTS_PATH = Path(
    "calibration_data_v3/resolver_segments.csv"
)

MANUAL_PATH = Path(
    "calibration_data_v3/manual_segment_labels.csv"
)

OUTPUT_PATH = Path(
    "calibration_data_v3/resolver_training.csv"
)


NON_SPEECH_LABELS = {
    "sigh",
    "blow",
    "breathing",
    "keyboard",
}


manual_labels = {}

with MANUAL_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    reader = csv.DictReader(
        file
    )

    for row in reader:
        key = (
            row["sample_id"],
            int(row["segment_index"])
        )

        manual_labels[key] = int(
            row["manual_target_speech"]
        )


segments = []

with SEGMENTS_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    reader = csv.DictReader(
        file
    )

    for row in reader:
        segments.append(
            row
        )


training_rows = []

manual_positive = 0
manual_negative = 0
noise_negative = 0


for row in segments:
    if row["ctc_decision"] == "accept":
        continue

    sample_id = row["sample_id"]

    segment_index = int(
        row["segment_index"]
    )

    label = row["label"]

    key = (
        sample_id,
        segment_index
    )

    target = None
    target_source = None

    if key in manual_labels:
        target = manual_labels[
            key
        ]

        target_source = "manual"

        if target == 1:
            manual_positive += 1
        else:
            manual_negative += 1

    elif label in NON_SPEECH_LABELS:
        target = 0
        target_source = "known_noise"

        noise_negative += 1

    else:
        continue

    output = {
        "sample_id": sample_id,
        "segment_index": segment_index,
        "original_label": label,
        "target_speech": target,
        "target_source": target_source,
    }

    excluded_columns = {
        "sample_id",
        "segment_index",
        "label",
        "target_speech",
        "ctc_decision",
    }

    for key_name, value in row.items():
        if key_name in excluded_columns:
            continue

        output[key_name] = value

    training_rows.append(
        output
    )


if not training_rows:
    raise RuntimeError(
        "No se generaron ejemplos."
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(
            training_rows[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        training_rows
    )


positive = sum(
    int(row["target_speech"]) == 1
    for row in training_rows
)

negative = sum(
    int(row["target_speech"]) == 0
    for row in training_rows
)


print()
print("=" * 70)

print(
    f"Ejemplos para resolver: "
    f"{len(training_rows)}"
)

print(
    f"Voz: "
    f"{positive}"
)

print(
    f"No voz: "
    f"{negative}"
)

print()

print(
    f"Positivos manuales: "
    f"{manual_positive}"
)

print(
    f"Negativos manuales: "
    f"{manual_negative}"
)

print(
    f"Negativos conocidos: "
    f"{noise_negative}"
)

print()

print(
    f"Archivo: "
    f"{OUTPUT_PATH}"
)

print()