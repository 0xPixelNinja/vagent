"""
Faster-Whisper STT plugin for LiveKit Agents (Client).
Connects to a remote STT server.
"""

import logging
from dataclasses import dataclass

import httpx
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, shortuuid

from vagent.utils.latency import LatencyTracker

logger = logging.getLogger("stt-whisper-client")


@dataclass
class STTOptions:
    language: str | None = None
    beam_size: int = 1
    vad_filter: bool = True
    without_timestamps: bool = True
    url: str = "http://localhost:8083/transcribe"


class FasterWhisperSTT(stt.STT):
    """Faster-Whisper based STT client for LiveKit Agents."""

    def __init__(
        self,
        url: str = "http://localhost:8083/transcribe",
        language: str | None = None,
        beam_size: int = 1,
        vad_filter: bool = True,
        without_timestamps: bool = True,
        # Keep these for compatibility with existing agent.py calls, but ignore them
        model_path: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._opts = STTOptions(
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            without_timestamps=without_timestamps,
            url=url,
        )
        self._client = httpx.AsyncClient(timeout=30.0)

    def load(self):
        """No-op for client, or check connection."""
        pass

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech from audio buffer."""

        LatencyTracker.get().stt_started()

        try:
            # Handle both single frame and list of frames
            if isinstance(buffer, list):
                frames = buffer
            else:
                frames = [buffer]

            if not frames:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=shortuuid(),
                    alternatives=[],
                )

            # Combine all frames into a single audio buffer
            all_data = b"".join(frame.data for frame in frames)
            sample_rate = frames[0].sample_rate
            num_channels = frames[0].num_channels

            lang = language if language is not NOT_GIVEN else self._opts.language

            # Prepare form data
            data = {
                "beam_size": str(self._opts.beam_size),
                "vad_filter": str(self._opts.vad_filter).lower(),
                "without_timestamps": str(self._opts.without_timestamps).lower(),
                "sample_rate": str(sample_rate),
                "channels": str(num_channels),
            }
            if lang:
                data["language"] = lang

            files = {"file": ("audio.raw", all_data, "application/octet-stream")}

            response = await self._client.post(self._opts.url, data=data, files=files)
            response.raise_for_status()
            result = response.json()

            if "error" in result and result["error"]:
                raise Exception(result["error"])

            full_text = result.get("text", "")
            detected_lang = result.get("language", "en")

            logger.debug(f"Transcribed: {full_text}")

            LatencyTracker.get().stt_finished(full_text)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=shortuuid(),
                alternatives=[
                    stt.SpeechData(
                        text=full_text,
                        language=detected_lang,
                        confidence=1.0,
                    )
                ],
            )
        except Exception as e:
            logger.error(f"STT recognition failed: {e}")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=shortuuid(),
                alternatives=[],
            )
