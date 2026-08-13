import json
import wave

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import SAMPLE_RATE


class CalibrationRecorder:
    def __init__(self, root="calibration_data_v2"):
        self.root = Path(root)

        self.windows_root = (
            self.root / "audio" / "windows"
        )

        self.candidates_root = (
            self.root / "audio" / "candidates"
        )

        self.records_path = (
            self.root / "records.jsonl"
        )

        self.windows_root.mkdir(
            parents=True,
            exist_ok=True
        )

        self.candidates_root.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(
        self,
        label,
        window_audio,
        ten_probabilities,
        ten_flags,
        candidate_audio=None,
        phonetic=None,
        classification=None,
        articulation=None,
        ast_result=None
    ):
        sample_id = (
            f"{label}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid4().hex[:8]}"
        )

        window_directory = (
            self.windows_root / label
        )

        window_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        window_path = (
            window_directory
            / f"{sample_id}.wav"
        )

        self._write_wav(
            window_path,
            window_audio
        )

        ten_detected = any(
            flag == 1
            for flag in ten_flags
        )

        record = {
            "id": sample_id,
            "label": label,
            "window_audio_path": str(
                window_path.as_posix()
            ),
            "window_duration": (
                len(window_audio)
                / SAMPLE_RATE
            ),
            "ten_detected": ten_detected,
            "ten_positive_frames": sum(
                flag == 1
                for flag in ten_flags
            ),
            "ten_total_frames": len(
                ten_flags
            ),
            "ten_max_probability": (
                max(ten_probabilities)
                if ten_probabilities
                else 0.0
            ),
            "ten_mean_probability": (
                sum(ten_probabilities)
                / len(ten_probabilities)
                if ten_probabilities
                else 0.0
            ),
            "candidate_audio_path": None,
            "phonetic": phonetic,
            "classification": None,
            "articulation": articulation,
            "ast": ast_result,
        }

        if candidate_audio is not None:
            candidate_directory = (
                self.candidates_root / label
            )

            candidate_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            candidate_path = (
                candidate_directory
                / f"{sample_id}.wav"
            )

            self._write_wav(
                candidate_path,
                candidate_audio
            )

            record["candidate_audio_path"] = str(
                candidate_path.as_posix()
            )

        if classification is not None:
            record["classification"] = {
                "decision": classification[
                    "decision"
                ].value,
                "score": classification[
                    "score"
                ],
                "evidence": classification[
                    "evidence"
                ],
            }

        record = self._json_safe(
            record
        )

        with self.records_path.open(
            "a",
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

        return window_path

    def _write_wav(
        self,
        path,
        audio
    ):
        pcm = (
            audio
            .reshape(-1)
            .astype(np.int16)
        )

        with wave.open(
            str(path),
            "wb"
        ) as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(
                SAMPLE_RATE
            )

            wav.writeframes(
                pcm.tobytes()
            )

    def _json_safe(
        self,
        value
    ):
        if isinstance(
            value,
            dict
        ):
            return {
                key: self._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple)
        ):
            return [
                self._json_safe(item)
                for item in value
            ]

        if isinstance(
            value,
            np.generic
        ):
            return value.item()

        return value