import csv
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path(
    "calibration_data_v3/resolver_segments.csv"
)

OUTPUT_PATH = Path(
    "calibration_data_v3/resolver_samples.csv"
)


def number(row, key):
    return float(
        row[key]
    )


def maximum(rows, key):
    return max(
        number(row, key)
        for row in rows
    )


def minimum(rows, key):
    return min(
        number(row, key)
        for row in rows
    )


def average(rows, key):
    values = [
        number(row, key)
        for row in rows
    ]

    return (
        sum(values)
        / len(values)
    )


groups = defaultdict(list)


with INPUT_PATH.open(
    "r",
    encoding="utf-8"
) as file:
    reader = csv.DictReader(
        file
    )

    for row in reader:
        groups[
            row["sample_id"]
        ].append(
            row
        )


samples = []


for sample_id, rows in groups.items():
    first = rows[0]

    decisions = [
        row["ctc_decision"]
        for row in rows
    ]

    sample = {
        "sample_id": sample_id,
        "label": first["label"],
        "target_speech": int(
            first["target_speech"]
        ),

        "segment_count": len(
            rows
        ),

        "ctc_accept_count": decisions.count(
            "accept"
        ),
        "ctc_uncertain_count": decisions.count(
            "uncertain"
        ),
        "ctc_reject_count": decisions.count(
            "reject"
        ),

        "max_ten_positive_ratio": maximum(
            rows,
            "ten_positive_ratio"
        ),
        "mean_ten_positive_ratio": average(
            rows,
            "ten_positive_ratio"
        ),
        "max_ten_probability": maximum(
            rows,
            "ten_max_probability"
        ),

        "max_phoneme_count": maximum(
            rows,
            "phoneme_count"
        ),
        "max_unique_phoneme_count": maximum(
            rows,
            "unique_phoneme_count"
        ),
        "max_transition_count": maximum(
            rows,
            "transition_count"
        ),
        "max_phoneme_confidence": maximum(
            rows,
            "phoneme_confidence"
        ),
        "max_active_non_blank_ratio": maximum(
            rows,
            "active_non_blank_ratio"
        ),
        "max_ctc_score": maximum(
            rows,
            "ctc_score"
        ),

        "max_ast_speech": maximum(
            rows,
            "ast_speech"
        ),
        "max_ast_male_speech_man_speaking": maximum(
            rows,
            "ast_male_speech_man_speaking"
        ),
        "max_ast_female_speech_woman_speaking": maximum(
            rows,
            "ast_female_speech_woman_speaking"
        ),
        "max_ast_child_speech_kid_speaking": maximum(
            rows,
            "ast_child_speech_kid_speaking"
        ),
        "max_ast_whispering": maximum(
            rows,
            "ast_whispering"
        ),

        "max_ast_sigh": maximum(
            rows,
            "ast_sigh"
        ),
        "max_ast_breathing": maximum(
            rows,
            "ast_breathing"
        ),
        "max_ast_gasp": maximum(
            rows,
            "ast_gasp"
        ),
        "max_ast_snort": maximum(
            rows,
            "ast_snort"
        ),
        "max_ast_throat_clearing": maximum(
            rows,
            "ast_throat_clearing"
        ),
        "max_ast_cough": maximum(
            rows,
            "ast_cough"
        ),
        "max_ast_sneeze": maximum(
            rows,
            "ast_sneeze"
        ),
        "max_ast_wheeze": maximum(
            rows,
            "ast_wheeze"
        ),

        "max_ast_computer_keyboard": maximum(
            rows,
            "ast_computer_keyboard"
        ),
        "max_ast_typing": maximum(
            rows,
            "ast_typing"
        ),

        "min_centroid_std": minimum(
            rows,
            "centroid_std"
        ),
        "max_centroid_std": maximum(
            rows,
            "centroid_std"
        ),
        "mean_centroid_std": average(
            rows,
            "centroid_std"
        ),

        "max_spectral_flux": maximum(
            rows,
            "spectral_flux_mean"
        ),
        "mean_spectral_flux": average(
            rows,
            "spectral_flux_mean"
        ),

        "max_band_change": maximum(
            rows,
            "band_change_mean"
        ),
        "mean_band_change": average(
            rows,
            "band_change_mean"
        ),

        "max_entropy": maximum(
            rows,
            "entropy_mean"
        ),
        "mean_entropy": average(
            rows,
            "entropy_mean"
        ),

        "max_rms_variation": maximum(
            rows,
            "rms_variation"
        ),

        "max_zcr": maximum(
            rows,
            "zcr_mean"
        ),
        "mean_zcr": average(
            rows,
            "zcr_mean"
        ),
    }

    samples.append(
        sample
    )


samples.sort(
    key=lambda item: (
        item["label"],
        item["sample_id"],
    )
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
            samples[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        samples
    )


print()
print("=" * 70)

print(
    f"Muestras originales: "
    f"{len(samples)}"
)

print(
    f"Archivo: "
    f"{OUTPUT_PATH}"
)

print()


labels = sorted(
    set(
        sample["label"]
        for sample in samples
    )
)


for label in labels:
    rows = [
        sample
        for sample in samples
        if sample["label"] == label
    ]

    accept = sum(
        sample["ctc_accept_count"] > 0
        for sample in rows
    )

    uncertain = sum(
        (
            sample["ctc_accept_count"] == 0
            and sample["ctc_uncertain_count"] > 0
        )
        for sample in rows
    )

    reject = sum(
        (
            sample["ctc_accept_count"] == 0
            and sample["ctc_uncertain_count"] == 0
        )
        for sample in rows
    )

    print(
        f"{label:10} | "
        f"muestras={len(rows):2d} | "
        f"accept={accept:2d} | "
        f"uncertain={uncertain:2d} | "
        f"reject={reject:2d}"
    )