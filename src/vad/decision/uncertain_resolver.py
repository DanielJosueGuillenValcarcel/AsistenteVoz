from ..config import (
    AST_MIN_GROUP_SCORE,
    AST_MIN_MARGIN,
)

from .speech_evidence_classifier import SpeechDecision


class UncertainResolver:
    def resolve(self, ast_result):
        speech = ast_result["speech_score"]
        non_speech = ast_result["non_speech_score"]

        if (
            speech >= AST_MIN_GROUP_SCORE
            and speech - non_speech >= AST_MIN_MARGIN
        ):
            return SpeechDecision.ACCEPT

        return SpeechDecision.REJECT