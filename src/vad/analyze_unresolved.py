import json
from pathlib import Path

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


def print_case(record, classification):
    phonetic = record.get(
        "phonetic"
    )

    ast = record.get(
        "ast"
    )

    articulation = record.get(
        "articulation"
    )

    print("=" * 75)

    print(
        f"REAL: {record['label'].upper()}"
    )

    print(
        f"ID: {record['id']}"
    )

    print(
        f"DECISION ACTUAL: "
        f"{classification['decision'].value.upper()}"
    )

    print()

    print(
        f"TEN positivos: "
        f"{record['ten_positive_frames']}/"
        f"{record['ten_total_frames']}"
    )

    print(
        f"TEN max: "
        f"{record['ten_max_probability']:.3f}"
    )

    print()

    if phonetic is not None:
        print(
            f"Fonemas: "
            f"{phonetic['phonemes']}"
        )

        print(
            f"Cantidad: "
            f"{phonetic['phoneme_count']}"
        )

        print(
            f"Unicos: "
            f"{phonetic['unique_phoneme_count']}"
        )

        print(
            f"Diversidad: "
            f"{phonetic['phoneme_diversity']:.3f}"
        )

        print(
            f"Confianza: "
            f"{phonetic['phoneme_confidence']:.3f}"
        )

        print(
            f"Non-blank activo: "
            f"{phonetic['active_non_blank_ratio']:.3f}"
        )

    print()

    if ast is not None:
        print(
            f"AST Speech: "
            f"{ast['speech_score']:.3f}"
        )

        print(
            f"AST Non-speech: "
            f"{ast['non_speech_score']:.3f}"
        )

        print(
            f"AST margen: "
            f"{ast['margin']:.3f}"
        )

        print("AST top:")

        for event in ast["events"][:5]:
            print(
                f"  {event['label']}: "
                f"{event['score']:.3f}"
            )

    print()

    if articulation is not None:
        print(
            f"Active frames: "
            f"{articulation['active_frame_ratio']:.3f}"
        )

        print(
            f"Flux: "
            f"{articulation['spectral_flux_mean']:.3f}"
        )

        print(
            f"Cambio bandas: "
            f"{articulation['band_change_mean']:.3f}"
        )

        print(
            f"Entropia: "
            f"{articulation['entropy_mean']:.3f}"
        )

        print(
            f"RMS variation: "
            f"{articulation['rms_variation']:.3f}"
        )

        print(
            f"ZCR: "
            f"{articulation['zcr_mean']:.3f}"
        )

        print(
            f"Centroide std: "
            f"{articulation['centroid_std']:.1f}"
        )

    print()


speech_unresolved = []
noise_uncertain = []


for record in records:
    phonetic = record.get(
        "phonetic"
    )

    if phonetic is None:
        continue

    classification = classifier.classify(
        phonetic
    )

    is_speech = (
        record["label"]
        in SPEECH_LABELS
    )

    if (
        is_speech
        and classification["decision"]
        != SpeechDecision.ACCEPT
    ):
        speech_unresolved.append(
            (
                record,
                classification
            )
        )

    if (
        not is_speech
        and classification["decision"]
        == SpeechDecision.UNCERTAIN
    ):
        noise_uncertain.append(
            (
                record,
                classification
            )
        )


print()
print(
    "HABLA REAL SIN RESOLVER:"
    f" {len(speech_unresolved)}"
)
print(
    "RUIDO UNCERTAIN:"
    f" {len(noise_uncertain)}"
)
print()


print(
    "#" * 75
)
print("HABLA REAL SIN RESOLVER")
print(
    "#" * 75
)
print()


for record, classification in speech_unresolved:
    print_case(
        record,
        classification
    )


print()
print(
    "#" * 75
)
print("RUIDO UNCERTAIN")
print(
    "#" * 75
)
print()


for record, classification in noise_uncertain:
    print_case(
        record,
        classification
    )