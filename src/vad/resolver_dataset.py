import csv
import json
import wave

from pathlib import Path

import numpy as np

from .config import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    PRE_ROLL_CHUNKS,
)

from .phonetic_verifier import PhoneticVerifier
from .speech_evidence_classifier import (
    SpeechEvidenceClassifier,
)

from .articulation_analyzer import (
    ArticulationAnalyzer,
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

OUTPUT_PATH = Path(
    "calibration_data_v3/resolver_segments.csv"
)


SPEECH_LABELS = {
    "normal",
    "whisper",
}


AST_FEATURE_LABELS = [
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Whispering",
    "Sigh",
    "Breathing",
    "Gasp",
    "Snort",
    "Throat clearing",
    "Cough",
    "Sneeze",
    "Wheeze",
    "Computer keyboard",
    "Typing",
    "Silence",
    "Sound effect",
]


phonetic_verifier = PhoneticVerifier()

speech_classifier = (
    SpeechEvidenceClassifier()
)

articulation_analyzer = (
    ArticulationAnalyzer()
)

audio_event_verifier = (
    AudioEventVerifier()
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

        record = json.loads(
            line
        )

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


rows = []

total_segments = sum(
    len(record["segments"])
    for record in v3_records
)

processed = 0


for record in v3_records:
    sample_id = record["id"]
    label = record["label"]

    target = (
        1
        if label in SPEECH_LABELS
        else 0
    )

    source = v2_records[
        sample_id
    ]

    audio = load_wav(
        source["window_audio_path"]
    )

    chunks = make_chunks(
        audio
    )

    for segment_index, segment in enumerate(
        record["segments"],
        start=1
    ):
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

        phonetic = (
            phonetic_verifier.analyze(
                segment_audio
            )
        )

        classification = (
            speech_classifier.classify(
                phonetic
            )
        )

        articulation = (
            articulation_analyzer.analyze(
                segment_audio
            )
        )

        ast = (
            audio_event_verifier.analyze(
                segment_audio
            )
        )

        probabilities = record[
            "probabilities"
        ][
            segment["start"]:
            segment["end"]
        ]

        flags = record[
            "flags"
        ][
            segment["start"]:
            segment["end"]
        ]

        positive_frames = sum(
            flags
        )

        total_frames = len(
            flags
        )

        ten_positive_ratio = (
            positive_frames
            / total_frames
            if total_frames > 0
            else 0.0
        )

        ten_mean_probability = (
            sum(probabilities)
            / len(probabilities)
            if probabilities
            else 0.0
        )

        ten_max_probability = (
            max(probabilities)
            if probabilities
            else 0.0
        )

        duration = (
            len(segment_audio)
            / SAMPLE_RATE
        )

        row = {
            "sample_id": sample_id,
            "segment_index": segment_index,
            "label": label,
            "target_speech": target,

            "segment_duration": duration,

            "ten_positive_frames": (
                positive_frames
            ),
            "ten_total_frames": (
                total_frames
            ),
            "ten_positive_ratio": (
                ten_positive_ratio
            ),
            "ten_mean_probability": (
                ten_mean_probability
            ),
            "ten_max_probability": (
                ten_max_probability
            ),

            "phoneme_count": (
                phonetic[
                    "phoneme_count"
                ]
            ),
            "unique_phoneme_count": (
                phonetic[
                    "unique_phoneme_count"
                ]
            ),
            "transition_count": (
                phonetic[
                    "transition_count"
                ]
            ),
            "phoneme_diversity": (
                phonetic[
                    "phoneme_diversity"
                ]
            ),
            "phoneme_confidence": (
                phonetic[
                    "phoneme_confidence"
                ]
            ),
            "non_blank_ratio": (
                phonetic[
                    "non_blank_ratio"
                ]
            ),
            "active_non_blank_ratio": (
                phonetic[
                    "active_non_blank_ratio"
                ]
            ),
            "active_phoneme_rate": (
                phonetic[
                    "active_phoneme_rate"
                ]
            ),
            "phonetic_span_duration": (
                phonetic[
                    "active_span_duration"
                ]
            ),

            "ctc_score": (
                classification[
                    "score"
                ]
            ),
            "ctc_decision": (
                classification[
                    "decision"
                ].value
            ),

            "spectral_flux_mean": (
                articulation[
                    "spectral_flux_mean"
                ]
            ),
            "spectral_flux_p90": (
                articulation[
                    "spectral_flux_p90"
                ]
            ),
            "band_change_mean": (
                articulation[
                    "band_change_mean"
                ]
            ),
            "band_change_p90": (
                articulation[
                    "band_change_p90"
                ]
            ),
            "centroid_std": (
                articulation[
                    "centroid_std"
                ]
            ),
            "centroid_change": (
                articulation[
                    "centroid_change"
                ]
            ),
            "entropy_mean": (
                articulation[
                    "entropy_mean"
                ]
            ),
            "entropy_std": (
                articulation[
                    "entropy_std"
                ]
            ),
            "rms_variation": (
                articulation[
                    "rms_variation"
                ]
            ),
            "zcr_mean": (
                articulation[
                    "zcr_mean"
                ]
            ),
            "zcr_std": (
                articulation[
                    "zcr_std"
                ]
            ),

            "ast_speech_group": (
                ast[
                    "speech_score"
                ]
            ),
            "ast_non_speech_group": (
                ast[
                    "non_speech_score"
                ]
            ),
            "ast_margin": (
                ast[
                    "margin"
                ]
            ),
        }

        for ast_label in AST_FEATURE_LABELS:
            column = (
                "ast_"
                + ast_label.lower()
                .replace(" ", "_")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
            )

            row[column] = (
                ast["label_scores"].get(
                    ast_label,
                    0.0
                )
            )

        rows.append(
            row
        )

        processed += 1

        print(
            f"{processed:03d}/"
            f"{total_segments:03d} | "
            f"{label:10} | "
            f"segmento={segment_index} | "
            f"CTC="
            f"{classification['decision'].value.upper()}"
        )


if not rows:
    raise RuntimeError(
        "No se generaron segmentos."
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


fieldnames = list(
    rows[0].keys()
)


with OUTPUT_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print()
print("=" * 70)

print(
    f"Segmentos guardados: "
    f"{len(rows)}"
)

print(
    f"Archivo: "
    f"{OUTPUT_PATH}"
)

print()


speech_segments = sum(
    row["target_speech"] == 1
    for row in rows
)

noise_segments = sum(
    row["target_speech"] == 0
    for row in rows
)

accept = sum(
    row["ctc_decision"]
    == "accept"
    for row in rows
)

uncertain = sum(
    row["ctc_decision"]
    == "uncertain"
    for row in rows
)

reject = sum(
    row["ctc_decision"]
    == "reject"
    for row in rows
)


print(
    f"Segmentos de habla: "
    f"{speech_segments}"
)

print(
    f"Segmentos de ruido: "
    f"{noise_segments}"
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

print()