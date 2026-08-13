from .audio.audio_capture import AudioCapture
from .audio.ten_vad_model import TenVADModel
from .audio.pre_roll_buffer import PreRollBuffer
from .audio.candidate_buffer import CandidateBuffer
from .verification.phonetic_worker import PhoneticWorker
from .verification.audio_event_verifier import AudioEventVerifier
from .decision.uncertain_resolver import UncertainResolver

from .decision.speech_evidence_classifier import (
    SpeechEvidenceClassifier,
    SpeechDecision,
)

from .config import (
    SAMPLE_RATE,
    PRE_ROLL_CHUNKS,
    CANDIDATE_END_SILENCE_CHUNKS,
    MAX_CANDIDATE_SAMPLES,
)


audio = AudioCapture()

vad = TenVADModel()

pre_roll = PreRollBuffer(
    PRE_ROLL_CHUNKS
)

candidate = CandidateBuffer(
    CANDIDATE_END_SILENCE_CHUNKS,
    MAX_CANDIDATE_SAMPLES
)

speech_classifier = SpeechEvidenceClassifier()

audio_event_verifier = AudioEventVerifier()

uncertain_resolver = UncertainResolver()

phonetic_worker = PhoneticWorker()


def on_phonetic_result(result, candidate_audio):
    print()
    print("ANALISIS FONETICO")
    print(f"Duracion total: {result['duration']:.3f}s")
    print(f"Fonemas: {result['phonemes']}")
    print(f"Cantidad: {result['phoneme_count']}")
    print(f"Unicos: {result['unique_phoneme_count']}")
    print(f"Transiciones: {result['transition_count']}")
    print(f"Diversidad: {result['phoneme_diversity']:.3f}")

    print(
        f"Duracion fonetica: "
        f"{result['active_span_duration']:.3f}s"
    )

    print(
        f"Non-blank global: "
        f"{result['non_blank_ratio']:.3f}"
    )

    print(
        f"Non-blank activo: "
        f"{result['active_non_blank_ratio']:.3f}"
    )

    print(
        f"Tasa fonetica activa: "
        f"{result['active_phoneme_rate']:.3f}/s"
    )

    print(
        f"Confianza fonetica: "
        f"{result['phoneme_confidence']:.3f}"
    )

    classification = speech_classifier.classify(
        result
    )

    print()

    print(
        f"Decision: "
        f"{classification['decision'].value.upper()}"
    )

    print(
        f"Speech score: "
        f"{classification['score']}"
    )

    print(
        f"Evidencia: "
        f"{', '.join(classification['evidence'])}"
    )

    print()

    final_decision = classification["decision"]

    if classification["decision"] == SpeechDecision.UNCERTAIN:
        ast_result = audio_event_verifier.analyze(
            candidate_audio
        )

        print("ANALISIS AST")

        for event in ast_result["events"]:
            print(
                f"{event['label']}: "
                f"{event['score']:.4f}"
            )

        print(
            f"Speech group: "
            f"{ast_result['speech_score']:.4f}"
        )

        print(
            f"Non-speech group: "
            f"{ast_result['non_speech_score']:.4f}"
        )

        print(
            f"Margen: "
            f"{ast_result['margin']:.4f}"
        )

        final_decision = uncertain_resolver.resolve(
            ast_result
        )

    print()
    print(
        f"DECISION FINAL: "
        f"{final_decision.value.upper()}"
    )
    print()

try:
    phonetic_worker.start(
        on_phonetic_result
    )

    audio.start()

    while True:
        chunk = audio.read()

        if chunk is None:
            continue

        pre_roll.add(chunk)

        probability, flag = vad.predict(
            chunk
        )

        if not candidate.active:
            if flag == 1:
                candidate.start(
                    pre_roll.get_audio()
                )

                print(
                    f"CANDIDATO INICIADO | "
                    f"prob={probability:.3f}"
                )

            continue

        result = candidate.add(
            chunk,
            flag
        )

        if result is not None:
            candidate_audio, reason = result

            duration = (
                len(candidate_audio)
                / SAMPLE_RATE
            )

            print(
                f"CANDIDATO CERRADO | "
                f"dur={duration:.3f}s | "
                f"reason={reason}"
            )

            phonetic_worker.submit(
                candidate_audio
            )

finally:
    audio.stop()
    phonetic_worker.stop()