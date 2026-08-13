import time

import numpy as np

from ..audio.audio_capture import AudioCapture
from ..audio.ten_vad_model import TenVADModel
from ..verification.phonetic_verifier import PhoneticVerifier
from ..decision.speech_evidence_classifier import SpeechEvidenceClassifier
from .articulation_analyzer import ArticulationAnalyzer
from ..verification.audio_event_verifier import AudioEventVerifier
from .calibration_recorder import CalibrationRecorder

from ..config import (
    CHUNK_SIZE,
    PRE_ROLL_CHUNKS,
    CANDIDATE_END_SILENCE_CHUNKS,
    CALIBRATION_WINDOW_CHUNKS,
)


LABELS = {
    "1": "normal",
    "2": "whisper",
    "3": "sigh",
    "4": "blow",
    "5": "breathing",
    "6": "keyboard",
}


audio = AudioCapture()
vad = TenVADModel()

phonetic_verifier = PhoneticVerifier()
speech_classifier = SpeechEvidenceClassifier()
articulation_analyzer = ArticulationAnalyzer()
audio_event_verifier = AudioEventVerifier()

recorder = CalibrationRecorder()


def prepare_capture():
    audio.clear()

    quiet_chunks = 0

    while quiet_chunks < 16:
        chunk = audio.read()

        if chunk is None:
            continue

        _, flag = vad.predict(
            chunk
        )

        if flag == 0:
            quiet_chunks += 1
        else:
            quiet_chunks = 0


def capture_window():
    chunks = []
    probabilities = []
    flags = []

    while len(chunks) < CALIBRATION_WINDOW_CHUNKS:
        chunk = audio.read(
            timeout=1.0
        )

        if chunk is None:
            continue

        probability, flag = vad.predict(
            chunk
        )

        chunks.append(
            chunk.copy()
        )

        probabilities.append(
            probability
        )

        flags.append(
            flag
        )

    window_audio = np.concatenate(
        chunks,
        axis=0
    )

    return (
        chunks,
        window_audio,
        probabilities,
        flags
    )


def extract_candidates(
    chunks,
    flags
):
    candidates = []

    active = False
    start_index = 0
    silence_chunks = 0
    speech_frames = 0

    for index, flag in enumerate(flags):
        if not active:
            if flag == 1:
                active = True
                start_index = index
                silence_chunks = 0
                speech_frames = 1

            continue

        if flag == 1:
            silence_chunks = 0
            speech_frames += 1
        else:
            silence_chunks += 1

        if (
            silence_chunks
            >= CANDIDATE_END_SILENCE_CHUNKS
        ):
            end_index = index + 1

            candidates.append(
                (
                    start_index,
                    end_index,
                    speech_frames
                )
            )

            active = False
            silence_chunks = 0
            speech_frames = 0

    if active:
        candidates.append(
            (
                start_index,
                len(chunks),
                speech_frames
            )
        )

    return candidates


def extract_best_candidate(
    chunks,
    flags
):
    candidates = extract_candidates(
        chunks,
        flags
    )

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: item[2]
    )

    start_index = max(
        0,
        best[0] - PRE_ROLL_CHUNKS
    )

    end_index = best[1]

    selected = chunks[
        start_index:end_index
    ]

    if not selected:
        return None

    return np.concatenate(
        selected,
        axis=0
    )


def analyze(
    candidate_audio
):
    phonetic = phonetic_verifier.analyze(
        candidate_audio
    )

    classification = speech_classifier.classify(
        phonetic
    )

    articulation = articulation_analyzer.analyze(
        candidate_audio
    )

    ast_result = audio_event_verifier.analyze(
        candidate_audio
    )

    return (
        phonetic,
        classification,
        articulation,
        ast_result
    )


def print_summary(
    probabilities,
    flags,
    phonetic=None,
    classification=None,
    ast_result=None
):
    positive = sum(
        flag == 1
        for flag in flags
    )

    print()
    print(
        f"TEN detectado: "
        f"{'SI' if positive > 0 else 'NO'}"
    )

    print(
        f"TEN frames positivos: "
        f"{positive}/{len(flags)}"
    )

    print(
        f"TEN max: "
        f"{max(probabilities):.3f}"
    )

    if phonetic is None:
        print(
            "Sin candidato para analisis secundario."
        )
        print()
        return

    print(
        f"Fonemas: "
        f"{phonetic['phonemes']}"
    )

    print(
        f"Cantidad: "
        f"{phonetic['phoneme_count']}"
    )

    print(
        f"Confianza fonetica: "
        f"{phonetic['phoneme_confidence']:.3f}"
    )

    print(
        f"Decision CTC: "
        f"{classification['decision'].value.upper()}"
    )

    print(
        f"Speech AST: "
        f"{ast_result['speech_score']:.4f}"
    )

    print(
        f"Non-speech AST: "
        f"{ast_result['non_speech_score']:.4f}"
    )

    print(
        f"Margen AST: "
        f"{ast_result['margin']:.4f}"
    )

    print()


try:
    audio.start()

    while True:
        print()
        print("CALIBRACION V2")
        print("1 - Voz normal")
        print("2 - Susurro")
        print("3 - Suspiro")
        print("4 - Soplo")
        print("5 - Respiracion")
        print("6 - Teclado")
        print("q - Salir")

        option = input(
            "Tipo: "
        ).strip().lower()

        if option == "q":
            break

        if option not in LABELS:
            continue

        label = LABELS[
            option
        ]

        amount_text = input(
            "Cantidad de muestras [10]: "
        ).strip()

        amount = (
            int(amount_text)
            if amount_text
            else 10
        )

        saved = 0

        while saved < amount:
            print()
            print(
                f"{label} "
                f"{saved + 1}/{amount}"
            )

            input(
                "Enter para preparar..."
            )

            vad.reset()
            prepare_capture()

            print("AHORA")
            print(
                "Tienes 4 segundos."
            )

            (
                chunks,
                window_audio,
                probabilities,
                flags
            ) = capture_window()

            candidate_audio = (
                extract_best_candidate(
                    chunks,
                    flags
                )
            )

            phonetic = None
            classification = None
            articulation = None
            ast_result = None

            if candidate_audio is not None:
                print(
                    "Analizando candidato..."
                )

                (
                    phonetic,
                    classification,
                    articulation,
                    ast_result
                ) = analyze(
                    candidate_audio
                )

            print_summary(
                probabilities,
                flags,
                phonetic,
                classification,
                ast_result
            )

            action = input(
                "Guardar [Enter] | "
                "r repetir | "
                "q terminar: "
            ).strip().lower()

            if action == "q":
                break

            if action == "r":
                continue

            path = recorder.save(
                label=label,
                window_audio=window_audio,
                ten_probabilities=probabilities,
                ten_flags=flags,
                candidate_audio=candidate_audio,
                phonetic=phonetic,
                classification=classification,
                articulation=articulation,
                ast_result=ast_result
            )

            saved += 1

            print(
                f"Guardado: {path}"
            )

finally:
    audio.stop()