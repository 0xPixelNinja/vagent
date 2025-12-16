"""Cartesia TTS plugin for LiveKit Agents.

Implements true streaming (incremental text in, audio chunks out) using Cartesia's
TTS WebSocket endpoint with contexts/continuations.

Env/config is expected to provide an API key and a voice id.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

from vagent.utils.latency import LatencyTracker

logger = logging.getLogger("tts-cartesia")


@dataclass(frozen=True)
class CartesiaTTSOptions:
    api_key: str
    cartesia_version: str = "2025-04-16"
    ws_url: str = "wss://api.cartesia.ai/tts/websocket"

    model_id: str = "sonic-3"
    voice_id: str = ""
    language: str | None = "en"

    sample_rate: int = 24000
    encoding: str = "pcm_s16le"
    container: str = "raw"

    # Cartesia-side buffering for token-by-token input
    max_buffer_delay_ms: int = 3000


class CartesiaTTS(tts.TTS):
    """Cartesia Sonic TTS over WebSocket with input streaming support."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model_id: str = "sonic-3",
        language: str | None = "en",
        cartesia_version: str = "2025-04-16",
        ws_url: str = "wss://api.cartesia.ai/tts/websocket",
        sample_rate: int = 24000,
        max_buffer_delay_ms: int = 3000,
    ):
        if not api_key:
            raise ValueError("Cartesia API key is required")
        if not voice_id:
            raise ValueError(
                "Cartesia voice_id is required (set VAGENT_CARTESIA_VOICE_ID or pass voice_id=...)"
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = CartesiaTTSOptions(
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            language=language,
            cartesia_version=cartesia_version,
            ws_url=ws_url.rstrip("/"),
            sample_rate=sample_rate,
            max_buffer_delay_ms=max_buffer_delay_ms,
        )

    async def aclose(self) -> None:
        pass

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "CartesiaSynthesizeStream":
        return CartesiaSynthesizeStream(tts=self, conn_options=conn_options, opts=self._opts)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "CartesiaChunkedStream":
        return CartesiaChunkedStream(tts=self, input_text=text, conn_options=conn_options, opts=self._opts)


class CartesiaChunkedStream(tts.ChunkedStream):
    """Non-incremental TTS call implemented on top of the WebSocket API."""

    def __init__(
        self,
        *,
        tts: CartesiaTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: CartesiaTTSOptions,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        tracker = LatencyTracker.get()
        first_audio_marked = False

        request_id = shortuuid()
        tracker.tts_started(self._input_text, request_id=request_id)
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        try:
            import websockets
        except ImportError as e:
            tracker.tts_finished()
            raise RuntimeError("websockets is required for Cartesia TTS") from e

        context_id = shortuuid()
        ws_url = self._opts.ws_url

        generation_req = _build_generation_request(
            opts=self._opts,
            context_id=context_id,
            transcript=self._input_text,
            may_continue=False,
        )

        try:
            async with websockets.connect(
                ws_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
                additional_headers={
                    "X-API-Key": self._opts.api_key,
                    "Cartesia-Version": self._opts.cartesia_version,
                },
            ) as ws:
                await ws.send(json.dumps(generation_req))

                async for message in ws:
                    if not isinstance(message, str):
                        continue

                    payload = json.loads(message)
                    msg_type = payload.get("type")

                    if msg_type in {"error"}:
                        raise RuntimeError(payload.get("message") or payload)

                    audio_bytes = _extract_audio_bytes(payload)
                    if audio_bytes:
                        if not first_audio_marked:
                            first_audio_marked = True
                            tracker.tts_first_audio_chunk(request_id=request_id)
                        output_emitter.push(audio_bytes)

                    if msg_type in {"done"} or payload.get("done") is True:
                        break
        finally:
            tracker.tts_finished(request_id=request_id)


class CartesiaSynthesizeStream(tts.SynthesizeStream):
    """Incremental text-in/audio-out stream using Cartesia contexts."""

    def __init__(self, *, tts: CartesiaTTS, conn_options: APIConnectOptions, opts: CartesiaTTSOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            import websockets
        except ImportError as e:
            raise RuntimeError("websockets is required for Cartesia TTS") from e

        tracker = LatencyTracker.get()
        state: dict[str, bool] = {
            "tts_started": False,
            "first_audio": False,
        }

        request_id = shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        context_id = shortuuid()
        ws_url = self._opts.ws_url

        input_closed_event = asyncio.Event()

        async with websockets.connect(
            ws_url,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
            additional_headers={
                "X-API-Key": self._opts.api_key,
                "Cartesia-Version": self._opts.cartesia_version,
            },
        ) as ws:
            send_task = asyncio.create_task(
                _tts_sender_task(
                    ws=ws,
                    input_ch=self._input_ch,
                    opts=self._opts,
                    context_id=context_id,
                    tracker=tracker,
                    request_id=request_id,
                    state=state,
                    input_closed_event=input_closed_event,
                ),
                name="cartesia-tts-sender",
            )

            recv_task = asyncio.create_task(
                _tts_receiver_task(
                    ws=ws,
                    output_emitter=output_emitter,
                    tracker=tracker,
                    request_id=request_id,
                    state=state,
                    input_closed_event=input_closed_event,
                    default_segment_id=context_id,
                ),
                name="cartesia-tts-receiver",
            )

            try:
                await asyncio.gather(send_task, recv_task)
            finally:
                tracker.tts_finished(request_id=request_id)


async def _tts_sender_task(
    *,
    ws: Any,
    input_ch: Any,
    opts: CartesiaTTSOptions,
    context_id: str,
    tracker: LatencyTracker,
    request_id: str,
    state: dict[str, bool],
    input_closed_event: asyncio.Event,
) -> None:
    """Reads tokens/flushes from LiveKit and forwards them as Cartesia inputs."""

    try:
        async for item in input_ch:
            if isinstance(item, str):
                if item and not state["tts_started"]:
                    tracker.tts_started("", request_id=request_id)
                    state["tts_started"] = True

                req = _build_generation_request(
                    opts=opts,
                    context_id=context_id,
                    transcript=item,
                    may_continue=True,
                )
                await ws.send(json.dumps(req))
                continue

            # Flush sentinel from LiveKit -> ask Cartesia to flush buffered text/audio
            req = _build_generation_request(
                opts=opts,
                context_id=context_id,
                transcript="",
                may_continue=True,
                flush=True,
            )
            await ws.send(json.dumps(req))

        # End of input: finalize context
        req = _build_generation_request(
            opts=opts,
            context_id=context_id,
            transcript="",
            may_continue=False,
        )
        await ws.send(json.dumps(req))
    finally:
        input_closed_event.set()


async def _tts_receiver_task(
    *,
    ws: Any,
    output_emitter: tts.AudioEmitter,
    tracker: LatencyTracker,
    request_id: str,
    state: dict[str, bool],
    input_closed_event: asyncio.Event,
    default_segment_id: str,
) -> None:
    current_segment_id: str | None = None
    async for message in ws:
        if not isinstance(message, str):
            continue

        payload = json.loads(message)
        msg_type = payload.get("type")

        if msg_type == "error":
            raise RuntimeError(payload.get("message") or payload)

        audio_bytes = _extract_audio_bytes(payload)
        if audio_bytes:
            segment_id = payload.get("context_id") or default_segment_id
            if current_segment_id is None:
                current_segment_id = str(segment_id)
                output_emitter.start_segment(segment_id=current_segment_id)
            if not state["first_audio"]:
                tracker.tts_first_audio_chunk(request_id=request_id)
                state["first_audio"] = True
            output_emitter.push(audio_bytes)

        if msg_type in {"flush_done"}:
            # Cartesia flushed its internal buffer. LiveKit will handle its own segmentation;
            # we just keep streaming audio under the current segment.
            pass

        if msg_type in {"done"} or payload.get("done") is True:
            if input_closed_event.is_set():
                output_emitter.end_input()
            break


def _build_generation_request(
    *,
    opts: CartesiaTTSOptions,
    context_id: str,
    transcript: str,
    may_continue: bool,
    flush: bool = False,
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "model_id": opts.model_id,
        "transcript": transcript,
        "voice": {"mode": "id", "id": opts.voice_id},
        "output_format": {
            "container": opts.container,
            "encoding": opts.encoding,
            "sample_rate": opts.sample_rate,
        },
        "context_id": context_id,
        "continue": bool(may_continue),
        "max_buffer_delay_ms": int(opts.max_buffer_delay_ms),
    }

    if opts.language:
        req["language"] = opts.language

    if flush:
        req["flush"] = True

    return req


def _extract_audio_bytes(payload: dict[str, Any]) -> bytes | None:
    """Best-effort extraction of audio bytes from Cartesia WebSocket payloads."""

    # Cartesia's schema is JSON, and the audio chunk uses base64.
    audio_b64 = (
        payload.get("audio")
        or payload.get("data")
        or payload.get("chunk")
        or payload.get("audio_data")
    )

    if not audio_b64 or not isinstance(audio_b64, str):
        return None

    try:
        return base64.b64decode(audio_b64)
    except Exception:
        return None
