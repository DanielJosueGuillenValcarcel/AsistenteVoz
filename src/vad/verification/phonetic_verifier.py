import numpy as np
import torch

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from ..config import SAMPLE_RATE, PHONETIC_MODEL


class PhoneticVerifier:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = Wav2Vec2Processor.from_pretrained(
            PHONETIC_MODEL
        )

        self.model = Wav2Vec2ForCTC.from_pretrained(
            PHONETIC_MODEL
        )

        self.model.to(self.device)
        self.model.eval()

        self.blank_id = self.model.config.pad_token_id

        if self.blank_id is None:
            self.blank_id = self.processor.tokenizer.pad_token_id

        self.special_ids = set(
            self.processor.tokenizer.all_special_ids
        )

    def analyze(self, audio):
        waveform = audio.reshape(-1).astype(np.float32)
        waveform /= 32768.0

        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        frame_confidence, predicted_ids = torch.max(
            probabilities,
            dim=-1
        )

        valid_mask = predicted_ids != self.blank_id

        for special_id in self.special_ids:
            if special_id != self.blank_id:
                valid_mask &= predicted_ids != special_id

        valid_frames = int(
            valid_mask.sum().item()
        )

        total_frames = len(predicted_ids)

        non_blank_ratio = (
            valid_frames / total_frames
            if total_frames > 0
            else 0.0
        )

        if valid_mask.any():
            phoneme_confidence = (
                frame_confidence[valid_mask]
                .mean()
                .item()
            )

            indexes = torch.where(valid_mask)[0]

            first_index = indexes[0].item()
            last_index = indexes[-1].item()

            active_span_frames = (
                last_index - first_index + 1
            )
        else:
            phoneme_confidence = 0.0
            active_span_frames = 0

        collapsed_ids = []
        previous_id = None

        for token_id in predicted_ids.tolist():
            if (
                token_id != previous_id
                and token_id != self.blank_id
                and token_id not in self.special_ids
            ):
                collapsed_ids.append(token_id)

            previous_id = token_id

        phoneme_count = len(collapsed_ids)

        unique_phoneme_count = len(
            set(collapsed_ids)
        )

        transition_count = sum(
            current != previous
            for previous, current in zip(
                collapsed_ids,
                collapsed_ids[1:]
            )
        )

        phoneme_diversity = (
            unique_phoneme_count / phoneme_count
            if phoneme_count > 0
            else 0.0
        )

        duration = len(waveform) / SAMPLE_RATE

        seconds_per_frame = (
            duration / total_frames
            if total_frames > 0
            else 0.0
        )

        active_span_duration = (
            active_span_frames
            * seconds_per_frame
        )

        active_non_blank_ratio = (
            valid_frames / active_span_frames
            if active_span_frames > 0
            else 0.0
        )

        active_phoneme_rate = (
            phoneme_count / active_span_duration
            if active_span_duration > 0
            else 0.0
        )

        transcription = self.processor.batch_decode(
            predicted_ids.unsqueeze(0)
        )[0]

        return {
            "duration": duration,
            "phonemes": transcription,
            "phoneme_count": phoneme_count,
            "unique_phoneme_count": unique_phoneme_count,
            "transition_count": transition_count,
            "phoneme_diversity": phoneme_diversity,
            "non_blank_ratio": non_blank_ratio,
            "active_span_duration": active_span_duration,
            "active_non_blank_ratio": active_non_blank_ratio,
            "active_phoneme_rate": active_phoneme_rate,
            "phoneme_confidence": phoneme_confidence,
        }