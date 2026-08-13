import json
import wave

from pathlib import Path

import numpy as np

from .config import (
    CHUNK_SIZE,
    CANDIDATE_END_SILENCE_CHUNKS,
)

from .ten_vad_model import TenVADModel


SOURCE = Path(
    "calibration_data_v2/records.jsonl"
)

OUTPUT_DIRECTORY = Path(
    "calibration_data_v3"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT = (
    OUTPUT_DIRECTORY
    / "ten_reprocessed.jsonl"
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
        end = (
            start
            + CHUNK_SIZE
        )

        chunks.append(
            audio[start:end].copy()
        )

    return chunks


def find_segments(flags):
    segments = []

    active = False
    start_index = 0
    silence = 0
    positives = 0

    for index, flag in enumerate(flags):
        if not active:
            if flag == 1:
                active = True
                start_index = index
                silence = 0
                positives = 1

            continue

        if flag == 1:
            silence = 0
            positives += 1
        else:
            silence += 1

        if (
            silence
            >= CANDIDATE_END_SILENCE_CHUNKS
        ):
            segments.append(
                {
                    "start": start_index,
                    "end": index + 1,
                    "positive_frames": positives,
                }
            )

            active = False
            silence = 0
            positives = 0

    if active:
        segments.append(
            {
                "start": start_index,
                "end": len(flags),
                "positive_frames": positives,
            }
        )

    return segments


def process(chunks):
    vad = TenVADModel()

    silence = np.zeros(
        (CHUNK_SIZE, 1),
        dtype=np.int16
    )

    for _ in range(16):
        vad.predict(
            silence
        )

    probabilities = []
    flags = []

    for chunk in chunks:
        probability, flag = vad.predict(
            chunk
        )

        probabilities.append(
            float(probability)
        )

        flags.append(
            int(flag)
        )

    return {
        "probabilities": probabilities,
        "flags": flags,
        "positive_frames": sum(flags),
        "max_probability": (
            max(probabilities)
            if probabilities
            else 0.0
        ),
        "segments": find_segments(
            flags
        ),
    }


records = []

with SOURCE.open(
    "r",
    encoding="utf-8"
) as file:
    for line in file:
        line = line.strip()

        if line:
            records.append(
                json.loads(line)
            )


results = []

mismatches = 0


for index, record in enumerate(
    records,
    start=1
):
    audio = load_wav(
        record["window_audio_path"]
    )

    chunks = make_chunks(
        audio
    )

    first = process(
        chunks
    )

    second = process(
        chunks
    )

    same_flags = (
        first["flags"]
        == second["flags"]
    )

    probability_difference = max(
        (
            abs(a - b)
            for a, b in zip(
                first["probabilities"],
                second["probabilities"]
            )
        ),
        default=0.0
    )

    deterministic = (
        same_flags
        and probability_difference < 1e-6
    )

    if not deterministic:
        mismatches += 1

    result = {
        "id": record["id"],
        "label": record["label"],
        "window_audio_path": record[
            "window_audio_path"
        ],
        "old_positive_frames": record[
            "ten_positive_frames"
        ],
        "old_max_probability": record[
            "ten_max_probability"
        ],
        "positive_frames": first[
            "positive_frames"
        ],
        "max_probability": first[
            "max_probability"
        ],
        "segments": first[
            "segments"
        ],
        "segment_count": len(
            first["segments"]
        ),
        "flags": first[
            "flags"
        ],
        "probabilities": first[
            "probabilities"
        ],
        "deterministic": deterministic,
        "max_repeat_difference": (
            probability_difference
        ),
    }

    results.append(
        result
    )

    print(
        f"{index:02d}/"
        f"{len(records)} | "
        f"{record['label']:10} | "
        f"old={record['ten_positive_frames']:3d} | "
        f"new={first['positive_frames']:3d} | "
        f"segments={len(first['segments'])} | "
        f"repeat={'OK' if deterministic else 'ERROR'}"
    )


with OUTPUT.open(
    "w",
    encoding="utf-8"
) as file:
    for result in results:
        file.write(
            json.dumps(
                result,
                ensure_ascii=False
            )
            + "\n"
        )


print()
print("=" * 70)

print(
    f"Muestras: "
    f"{len(results)}"
)

print(
    f"Repeticiones diferentes: "
    f"{mismatches}"
)

print(
    f"Guardado en: "
    f"{OUTPUT}"
)

print()


labels = sorted(
    set(
        result["label"]
        for result in results
    )
)


for label in labels:
    samples = [
        result
        for result in results
        if result["label"] == label
    ]

    detected = sum(
        result["positive_frames"] > 0
        for result in samples
    )

    average_positive = sum(
        result["positive_frames"]
        for result in samples
    ) / len(samples)

    average_segments = sum(
        result["segment_count"]
        for result in samples
    ) / len(samples)

    max_segments = max(
        result["segment_count"]
        for result in samples
    )

    print(
        f"{label:10} | "
        f"TEN={detected}/{len(samples)} | "
        f"frames prom={average_positive:.1f} | "
        f"segmentos prom={average_segments:.2f} | "
        f"max segmentos={max_segments}"
    )