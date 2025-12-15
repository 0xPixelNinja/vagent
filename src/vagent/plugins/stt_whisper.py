"""
Faster-Whisper STT plugin for LiveKit Agents.
Uses the local large-v3-turbo model for speech-to-text.
"""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, shortuuid

from vagent.utils.latency import LatencyTracker

logger = logging.getLogger("stt-whisper")


@dataclass
class STTOptions:
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    without_timestamps: bool = True


class FasterWhisperSTT(stt.STT):
    """Faster-Whisper based STT for LiveKit Agents."""

    def __init__(
        self,
        model_path: str = "models",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        without_timestamps: bool = True,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._model_path = model_path
        self._device = device
        self._compute_type = compute_type
        self._opts = STTOptions(
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            without_timestamps=without_timestamps,
        )
        self._model: WhisperModel | None = None

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == 16000:
            return audio.astype(np.float32, copy=False)

        # Polyphase resampling is generally faster than FFT resampling.
        from scipy import signal
        import math

        g = math.gcd(src_sr, 16000)
        up = 16000 // g
        down = src_sr // g
        return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            logger.info(f"Loading Whisper model from {self._model_path}")
            self._model = WhisperModel(
                self._model_path,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("Whisper model loaded successfully")
        return self._model

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech from audio buffer."""

        LatencyTracker.get().stt_started()

        model = self._ensure_model()

        # Handle both single frame and list of frames
        if isinstance(buffer, list):
            frames = buffer
        else:
            frames = [buffer]

        # Combine all frames into a single audio array
        all_data = b"".join(frame.data for frame in frames)
        sample_rate = frames[0].sample_rate
        num_channels = frames[0].num_channels

        # Convert bytes to numpy array (assuming 16-bit signed int)
        audio_data = np.frombuffer(all_data, dtype=np.int16).astype(np.float32)

        # Convert to mono if stereo
        if num_channels > 1:
            audio_data = audio_data.reshape(-1, num_channels).mean(axis=1)

        # Normalize to [-1, 1]
        audio_data = audio_data / 32768.0

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio_data = self._resample_to_16k(audio_data, sample_rate)

        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        lang = language if language is not NOT_GIVEN else self._opts.language
        
        segments, info = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                audio_data,
                language=lang,
                beam_size=self._opts.beam_size,
                vad_filter=self._opts.vad_filter,
                without_timestamps=self._opts.without_timestamps,
            ),
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts)

        logger.debug(f"Transcribed: {full_text}")

        LatencyTracker.get().stt_finished(full_text)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=shortuuid(),
            alternatives=[
                stt.SpeechData(
                    text=full_text,
                    language=info.language if info.language else "en",
                    confidence=1.0,
                )
            ],
        )
