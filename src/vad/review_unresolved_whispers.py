import csv
import json
import wave

from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    PRE_ROLL_CHUNKS,
)


V2_PATH = Path(
    "calibration_data_v2/records.jsonl"
)

V3_PATH = Path(
    "calibration_data_v3/ten_reprocessed.jsonl"
)

SEGMENTS_PATH = Path(
    "calibration_data_v3/resolver_segments.csv"
)

OUTPUT_PATH = Path(
    "calibration_data_v3/manual_segment_labels.csv"
)


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
        line = line.strip()

        if not line:
            continue

        record = json.loads(
            line
        )

        v2[
            record["id"]
        ] = record


v3 = {}

with V3_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        record = json.loads(
            line
        )

        v3[
            record["id"]
        ] = record


segment_rows = []

with SEGMENTS_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    reader = csv.DictReader(
        file
    )

    segment_rows = list(
        reader
    )


whisper_groups = {}

for row in segment_rows:
    if row["label"] != "whisper":
        continue

    whisper_groups.setdefault(
        row["sample_id"],
        []
    ).append(
        row
    )


unresolved_ids = set()

for sample_id, rows in whisper_groups.items():
    has_accept = any(
        row["ctc_decision"] == "accept"
        for row in rows
    )

    if not has_accept:
        unresolved_ids.add(
            sample_id
        )


already_labeled = {}

if OUTPUT_PATH.exists():
    with OUTPUT_PATH.open(
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

            already_labeled[
                key
            ] = row


pending = []

for row in segment_rows:
    sample_id = row[
        "sample_id"
    ]

    if sample_id not in unresolved_ids:
        continue

    segment_index = int(
        row["segment_index"]
    )

    key = (
        sample_id,
        segment_index
    )

    if key in already_labeled:
        continue

    pending.append(
        (
            sample_id,
            segment_index
        )
    )


print()
print(
    f"Susurros sin ACCEPT: "
    f"{len(unresolved_ids)}"
)

print(
    f"Segmentos pendientes: "
    f"{len(pending)}"
)

print()


for position, (
    sample_id,
    segment_index
) in enumerate(
    pending,
    start=1
):
    source = v2[
        sample_id
    ]

    ten_record = v3[
        sample_id
    ]

    segment = ten_record[
        "segments"
    ][
        segment_index - 1
    ]

    audio = load_wav(
        source["window_audio_path"]
    )

    chunks = make_chunks(
        audio
    )

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

    duration = (
        len(segment_audio)
        / SAMPLE_RATE
    )

    while True:
        print(
            "=" * 60
        )

        print(
            f"{position}/"
            f"{len(pending)}"
        )

        print(
            f"Muestra: "
            f"{sample_id}"
        )

        print(
            f"Segmento: "
            f"{segment_index}"
        )

        print(
            f"Duracion: "
            f"{duration:.3f}s"
        )

        print()
        print(
            "Reproduciendo..."
        )

        playback = (
            segment_audio
            .reshape(-1)
            .astype(np.float32)
            / 32768.0
        )

        sd.play(
            playback,
            SAMPLE_RATE
        )

        sd.wait()

        print()

        option = input(
            "s = contiene voz | "
            "n = no contiene voz | "
            "r = repetir | "
            "q = salir: "
        ).strip().lower()

        if option == "r":
            continue

        if option == "q":
            raise SystemExit

        if option not in (
            "s",
            "n"
        ):
            continue

        manual_target = (
            1
            if option == "s"
            else 0
        )

        row = {
            "sample_id": sample_id,
            "segment_index": segment_index,
            "manual_target_speech": (
                manual_target
            ),
        }

        write_header = (
            not OUTPUT_PATH.exists()
            or OUTPUT_PATH.stat().st_size == 0
        )

        with OUTPUT_PATH.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "sample_id",
                    "segment_index",
                    "manual_target_speech",
                ]
            )

            if write_header:
                writer.writeheader()

            writer.writerow(
                row
            )

        break


print()
print(
    "Revision terminada."
)

print(
    f"Etiquetas: "
    f"{OUTPUT_PATH}"
)