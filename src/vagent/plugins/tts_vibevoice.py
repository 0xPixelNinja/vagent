"""
VibeVoice TTS plugin for LiveKit Agents.
Uses Microsoft VibeVoice-Realtime via WebSocket for streaming text-to-speech.
https://github.com/microsoft/VibeVoice
"""

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

from vagent.utils.latency import LatencyTracker

logger = logging.getLogger("tts-vibevoice")

# VibeVoice outputs 24kHz PCM16 audio
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class VibeVoiceOptions:
    voice: str = "en-Davis_man"
    cfg_scale: float = 1.5
    inference_steps: int = 5


class VibeVoiceTTS(tts.TTS):
    """
    VibeVoice-Realtime TTS for LiveKit Agents via WebSocket streaming.
    
    VibeVoice-Realtime is a 0.5B parameter model that produces initial 
    audible speech in ~300ms and supports streaming text input.
    """

    def __init__(
        self,
        base_url: str = "ws://localhost:3000",
        voice: str = "en-Davis_man",
        cfg_scale: float = 1.5,
        inference_steps: int = 5,
    ):
        super().__init__(
            # VibeVoice supports streaming audio output but not incremental text input
            # via LiveKit's TTS.stream() interface. Use StreamAdapter on top of synthesize().
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        # Normalize URL: ensure ws:// or wss:// scheme
        if base_url.startswith("http://"):
            base_url = "ws://" + base_url[7:]
        elif base_url.startswith("https://"):
            base_url = "wss://" + base_url[8:]
        self._base_url = base_url.rstrip("/")
        self._opts = VibeVoiceOptions(
            voice=voice,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
        )

    async def aclose(self) -> None:
        pass

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "VibeVoiceChunkedStream":
        return VibeVoiceChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            base_url=self._base_url,
        )


class VibeVoiceChunkedStream(tts.ChunkedStream):
    """Chunked TTS stream via VibeVoice WebSocket."""

    def __init__(
        self,
        *,
        tts: VibeVoiceTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: VibeVoiceOptions,
        base_url: str,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._vibevoice_tts = tts
        self._opts = opts
        self._base_url = base_url

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Stream audio chunks from VibeVoice WebSocket."""
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

        try:
            import websockets
        except ImportError:
            logger.error("websockets package required for VibeVoice TTS. Install with: pip install websockets")
            tracker.tts_finished()
            return

        # Build WebSocket URL with query parameters
        from urllib.parse import urlencode, quote
        params = {
            "text": self._input_text,
            "cfg": f"{self._opts.cfg_scale:.3f}",
            "steps": str(self._opts.inference_steps),
        }
        if self._opts.voice:
            params["voice"] = self._opts.voice
        
        ws_url = f"{self._base_url}/stream?{urlencode(params, quote_via=quote)}"

        try:
            async with websockets.connect(
                ws_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                async for message in ws:
                    # VibeVoice sends binary PCM16 audio or JSON log messages
                    if isinstance(message, bytes):
                        if not first_audio_marked:
                            first_audio_marked = True
                            tracker.tts_first_audio_chunk()
                        
                        # Convert PCM16 bytes to PCM bytes for LiveKit
                        # VibeVoice outputs 16-bit signed PCM at 24kHz
                        output_emitter.push(message)
                    else:
                        # JSON log message from server - ignore or log for debugging
                        pass

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"VibeVoice WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"VibeVoice TTS synthesis failed: {e}")
        finally:
            tracker.tts_finished()
