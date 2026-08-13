from enum import Enum

from ..config import (
    EVIDENCE_ACCEPT_SCORE,
    EVIDENCE_REJECT_SCORE,
    EVIDENCE_MIN_DIVERSITY,
    EVIDENCE_MIN_ACTIVE_NON_BLANK,
    EVIDENCE_MIN_PHONETIC_CONFIDENCE,
    CTC_ACCEPT_MIN_CONFIDENCE,
)


class SpeechDecision(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class SpeechEvidenceClassifier:
    def classify(self, result):
        count = result["phoneme_count"]
        unique = result["unique_phoneme_count"]
        transitions = result["transition_count"]
        diversity = result["phoneme_diversity"]
        active_non_blank = result["active_non_blank_ratio"]
        confidence = result["phoneme_confidence"]

        if count == 0:
            return {
                "decision": SpeechDecision.REJECT,
                "score": 0,
                "evidence": []
            }
            
        if count == 1:
            return {
                "decision": SpeechDecision.UNCERTAIN,
                "score": 2,
                "evidence": ["single_phonetic_event"]
            }
        
        if (
            count == 2
            and confidence < EVIDENCE_MIN_PHONETIC_CONFIDENCE
        ):
            return {
                "decision": SpeechDecision.UNCERTAIN,
                "score": 4,
                "evidence": [
                    "short_phonetic_sequence",
                    "low_phonetic_confidence"
                ]
            }
        
        if (
            count <= 4
            and (
                diversity < EVIDENCE_MIN_DIVERSITY
                or confidence < EVIDENCE_MIN_PHONETIC_CONFIDENCE
            )
        ):
            return {
                "decision": SpeechDecision.UNCERTAIN,
                "score": 5,
                "evidence": [
                    "short_weak_phonetic_sequence"
                ]
            }
            
        score = 0
        evidence = []

        if count >= 2:
            score += 2
            evidence.append("multiple_phonemes")

        if unique >= 2:
            score += 2
            evidence.append("phoneme_variation")

        if transitions >= 1:
            score += 2
            evidence.append("phonetic_transitions")

        if diversity >= EVIDENCE_MIN_DIVERSITY:
            score += 1
            evidence.append("high_diversity")

        if active_non_blank >= EVIDENCE_MIN_ACTIVE_NON_BLANK:
            score += 1
            evidence.append("active_ctc_structure")

        if confidence >= EVIDENCE_MIN_PHONETIC_CONFIDENCE:
            score += 1
            evidence.append("phonetic_confidence")

        if count >= 4:
            score += 1
            evidence.append("rich_sequence")

        if unique >= 3:
            score += 1
            evidence.append("multiple_unique_phonemes")

        if score >= EVIDENCE_ACCEPT_SCORE:
            decision = SpeechDecision.ACCEPT

        elif score <= EVIDENCE_REJECT_SCORE:
            decision = SpeechDecision.REJECT

        else:
            decision = SpeechDecision.UNCERTAIN
        
        if (
            decision == SpeechDecision.ACCEPT
            and confidence < CTC_ACCEPT_MIN_CONFIDENCE
        ):
            decision = SpeechDecision.UNCERTAIN
            evidence.append(
                "low_accept_confidence"
            )

        return {
            "decision": decision,
            "score": score,
            "evidence": evidence
        }