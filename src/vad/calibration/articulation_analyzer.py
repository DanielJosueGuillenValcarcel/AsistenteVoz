import numpy as np

from ..config import (
    SAMPLE_RATE,
    ARTICULATION_FRAME_MS,
    ARTICULATION_HOP_MS,
    ARTICULATION_MIN_FREQ,
    ARTICULATION_MAX_FREQ,
    ARTICULATION_ACTIVE_RATIO,
)


class ArticulationAnalyzer:
    def __init__(self):
        self.frame_size = int(
            SAMPLE_RATE * ARTICULATION_FRAME_MS / 1000
        )

        self.hop_size = int(
            SAMPLE_RATE * ARTICULATION_HOP_MS / 1000
        )

        self.window = np.hanning(
            self.frame_size
        ).astype(np.float32)

        self.frequencies = np.fft.rfftfreq(
            self.frame_size,
            d=1.0 / SAMPLE_RATE
        )

        self.frequency_mask = (
            (self.frequencies >= ARTICULATION_MIN_FREQ)
            & (self.frequencies <= ARTICULATION_MAX_FREQ)
        )

        self.band_edges = [
            80,
            200,
            400,
            800,
            1200,
            2000,
            3000,
            4500,
            7600,
        ]

    def analyze(self, audio):
        waveform = audio.reshape(-1).astype(np.float32)
        waveform /= 32768.0

        frames = self._create_frames(waveform)

        rms = np.sqrt(
            np.mean(frames * frames, axis=1) + 1e-12
        )

        reference_rms = np.percentile(
            rms,
            90
        )

        if reference_rms <= 1e-8:
            return self._empty_result()

        active_threshold = (
            reference_rms
            * ARTICULATION_ACTIVE_RATIO
        )

        active_indexes = np.where(
            rms >= active_threshold
        )[0]

        if len(active_indexes) == 0:
            return self._empty_result()

        first = active_indexes[0]
        last = active_indexes[-1]

        active_frames = frames[
            first:last + 1
        ]

        active_rms = rms[
            first:last + 1
        ]

        spectrum = np.abs(
            np.fft.rfft(
                active_frames * self.window,
                axis=1
            )
        ) ** 2

        spectrum = spectrum[
            :,
            self.frequency_mask
        ]

        frequencies = self.frequencies[
            self.frequency_mask
        ]

        spectrum += 1e-12

        normalized = spectrum / np.sum(
            spectrum,
            axis=1,
            keepdims=True
        )

        if len(normalized) > 1:
            spectral_difference = (
                normalized[1:]
                - normalized[:-1]
            )

            flux_values = np.sum(
                np.maximum(
                    spectral_difference,
                    0.0
                ),
                axis=1
            )

            spectral_flux_mean = float(
                np.mean(flux_values)
            )

            spectral_flux_p90 = float(
                np.percentile(
                    flux_values,
                    90
                )
            )
        else:
            spectral_flux_mean = 0.0
            spectral_flux_p90 = 0.0

        band_energies = []

        full_frequencies = self.frequencies

        for low, high in zip(
            self.band_edges[:-1],
            self.band_edges[1:]
        ):
            mask = (
                (full_frequencies >= low)
                & (full_frequencies < high)
            )

            energy = np.sum(
                np.abs(
                    np.fft.rfft(
                        active_frames
                        * self.window,
                        axis=1
                    )
                )[:, mask] ** 2,
                axis=1
            )

            band_energies.append(
                np.log(energy + 1e-12)
            )

        band_energies = np.stack(
            band_energies,
            axis=1
        )

        band_energies -= np.mean(
            band_energies,
            axis=1,
            keepdims=True
        )

        if len(band_energies) > 1:
            band_change_values = np.mean(
                np.abs(
                    band_energies[1:]
                    - band_energies[:-1]
                ),
                axis=1
            )

            band_change_mean = float(
                np.mean(
                    band_change_values
                )
            )

            band_change_p90 = float(
                np.percentile(
                    band_change_values,
                    90
                )
            )
        else:
            band_change_mean = 0.0
            band_change_p90 = 0.0

        centroid = np.sum(
            normalized * frequencies,
            axis=1
        )

        centroid_std = float(
            np.std(centroid)
        )

        if len(centroid) > 1:
            centroid_change = float(
                np.mean(
                    np.abs(
                        np.diff(centroid)
                    )
                )
            )
        else:
            centroid_change = 0.0

        entropy = -np.sum(
            normalized
            * np.log(
                normalized + 1e-12
            ),
            axis=1
        )

        entropy /= np.log(
            normalized.shape[1]
        )

        entropy_mean = float(
            np.mean(entropy)
        )

        entropy_std = float(
            np.std(entropy)
        )

        log_rms = np.log(
            active_rms + 1e-8
        )

        rms_variation = float(
            np.std(log_rms)
        )

        signs = np.signbit(
            active_frames
        )

        zero_crossing = np.mean(
            signs[:, 1:]
            != signs[:, :-1],
            axis=1
        )

        zcr_mean = float(
            np.mean(zero_crossing)
        )

        zcr_std = float(
            np.std(zero_crossing)
        )

        active_span_duration = (
            (
                (last - first)
                * self.hop_size
            )
            + self.frame_size
        ) / SAMPLE_RATE

        active_frame_ratio = (
            len(active_indexes)
            / len(frames)
        )

        return {
            "active_span_duration": active_span_duration,
            "active_frame_ratio": active_frame_ratio,
            "spectral_flux_mean": spectral_flux_mean,
            "spectral_flux_p90": spectral_flux_p90,
            "band_change_mean": band_change_mean,
            "band_change_p90": band_change_p90,
            "centroid_std": centroid_std,
            "centroid_change": centroid_change,
            "entropy_mean": entropy_mean,
            "entropy_std": entropy_std,
            "rms_variation": rms_variation,
            "zcr_mean": zcr_mean,
            "zcr_std": zcr_std,
        }

    def _create_frames(self, waveform):
        if len(waveform) < self.frame_size:
            waveform = np.pad(
                waveform,
                (
                    0,
                    self.frame_size
                    - len(waveform)
                )
            )

        frame_count = (
            1
            + (
                len(waveform)
                - self.frame_size
            )
            // self.hop_size
        )

        frames = []

        for index in range(frame_count):
            start = (
                index
                * self.hop_size
            )

            end = (
                start
                + self.frame_size
            )

            frames.append(
                waveform[start:end]
            )

        return np.stack(
            frames,
            axis=0
        )

    def _empty_result(self):
        return {
            "active_span_duration": 0.0,
            "active_frame_ratio": 0.0,
            "spectral_flux_mean": 0.0,
            "spectral_flux_p90": 0.0,
            "band_change_mean": 0.0,
            "band_change_p90": 0.0,
            "centroid_std": 0.0,
            "centroid_change": 0.0,
            "entropy_mean": 0.0,
            "entropy_std": 0.0,
            "rms_variation": 0.0,
            "zcr_mean": 0.0,
            "zcr_std": 0.0,
        }