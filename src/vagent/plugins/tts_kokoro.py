"""
Kokoro TTS plugin for LiveKit Agents.
Uses Kokoro-FastAPI (OpenAI-compatible endpoint) for text-to-speech.
"""

import asyncio
import logging
from dataclasses import dataclass

import httpx
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

from vagent.utils.latency import LatencyTracker

logger = logging.getLogger("tts-kokoro")

# Audio format constants
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class TTSOptions:
    voice: str = "af_heart"
    speed: float = 1.0


class KokoroTTS(tts.TTS):
    """Kokoro-FastAPI TTS for LiveKit Agents via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8880/v1",
        voice: str = "af_heart",
        speed: float = 1.0,
    ):
        super().__init__(
            # Kokoro endpoint is OpenAI-compatible TTS over HTTP (chunked response),
            # but it does not support incremental text input via TTS.stream().
            # Advertise streaming=False so LiveKit uses StreamAdapter on top of synthesize().
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._base_url = base_url.rstrip("/")
        self._opts = TTSOptions(voice=voice, speed=speed)
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=20,
                    keepalive_expiry=120,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "KokoroChunkedStream":
        return KokoroChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            base_url=self._base_url,
        )


class KokoroChunkedStream(tts.ChunkedStream):
    """Chunked TTS stream via Kokoro-FastAPI."""

    def __init__(
        self,
        *,
        tts: KokoroTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: TTSOptions,
        base_url: str,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._kokoro_tts = tts
        self._opts = opts
        self._base_url = base_url

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Stream audio chunks from Kokoro-FastAPI."""
        request_id = shortuuid()

        tracker = LatencyTracker.get()
        tracker.tts_started(self._input_text)
        first_audio_marked = False

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        payload = {
            "model": "kokoro",
            "input": self._input_text,
            "voice": self._opts.voice,
            "speed": self._opts.speed,
            "response_format": "pcm",
        }

        client = self._kokoro_tts._ensure_client()

        try:
            async with client.stream(
                "POST",
                f"{self._base_url}/audio/speech",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes(chunk_size=4800):
                    if chunk:
                        if not first_audio_marked:
                            first_audio_marked = True
                            tracker.tts_first_audio_chunk()
                        output_emitter.push(chunk)
        finally:
            # Always close out timing for this turn, even if TTS fails.
            tracker.tts_finished()
