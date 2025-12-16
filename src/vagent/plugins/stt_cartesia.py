"""Cartesia STT plugin for LiveKit Agents.

Supports streaming speech-to-text via Cartesia Ink WebSocket (`/stt/websocket`).
Also provides a non-streaming recognize implementation using the batch `/stt`
endpoint for fallback/compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
import livekit.rtc as rtc
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, shortuuid

from vagent.utils.latency import LatencyTracker

logger = logging.getLogger("stt-cartesia")


@dataclass(frozen=True)
class CartesiaSTTOptions:
    api_key: str
    cartesia_version: str = "2025-04-16"

    # Streaming
    ws_url: str = "wss://api.cartesia.ai/stt/websocket"
    model: str = "ink-whisper"
    language: str = "en"
    encoding: str = "pcm_s16le"
    sample_rate: int = 16000
    min_volume: float = 0.0
    max_silence_duration_secs: float = 0.8

    # Batch fallback
    http_url: str = "https://api.cartesia.ai/stt"


class CartesiaSTT(stt.STT):
    """Cartesia Ink STT (streaming + batch fallback)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "ink-whisper",
        language: str = "en",
        cartesia_version: str = "2025-04-16",
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        min_volume: float = 0.0,
        max_silence_duration_secs: float = 0.8,
        ws_url: str = "wss://api.cartesia.ai/stt/websocket",
        http_url: str = "https://api.cartesia.ai/stt",
    ):
        if not api_key:
            raise ValueError("Cartesia API key is required")

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        self._opts = CartesiaSTTOptions(
            api_key=api_key,
            model=model,
            language=language,
            cartesia_version=cartesia_version,
            encoding=encoding,
            sample_rate=sample_rate,
            min_volume=min_volume,
            max_silence_duration_secs=max_silence_duration_secs,
            ws_url=ws_url.rstrip("/"),
            http_url=http_url.rstrip("/"),
        )

        self._client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "CartesiaRecognizeStream":
        if language is NOT_GIVEN:
            lang: str = self._opts.language
        else:
            assert isinstance(language, str)
            lang = language
        return CartesiaRecognizeStream(
            stt=self,
            conn_options=conn_options,
            opts=self._opts,
            language=lang,
            sample_rate=self._opts.sample_rate,
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        request_id = shortuuid()
        LatencyTracker.get().stt_started(request_id=request_id)

        try:
            if isinstance(buffer, list):
                raw_frames = buffer
            else:
                raw_frames = [buffer]

            frames = [frame for frame in raw_frames if isinstance(frame, rtc.AudioFrame)]

            if not frames:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[],
                )

            all_data = b"".join(bytes(frame.data) for frame in frames)
            sample_rate = frames[0].sample_rate
            if language is NOT_GIVEN:
                lang = self._opts.language
            else:
                lang = language

            params = {
                "encoding": self._opts.encoding,
                "sample_rate": str(sample_rate),
            }

            data = {
                "model": self._opts.model,
                "language": lang,
            }

            files = {"file": ("audio.raw", all_data, "application/octet-stream")}

            resp = await self._client.post(
                self._opts.http_url,
                params=params,
                data=data,
                files=files,
                headers={
                    "X-API-Key": self._opts.api_key,
                    "Cartesia-Version": self._opts.cartesia_version,
                },
            )
            resp.raise_for_status()
            payload = resp.json()

            text = payload.get("text", "")
            detected_lang = payload.get("language") or (lang or "en")

            LatencyTracker.get().stt_finished(text, request_id=request_id)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        language=detected_lang,
                        confidence=1.0,
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Cartesia STT recognition failed: {e}")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[],
            )


class CartesiaRecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: CartesiaSTT,
        conn_options: APIConnectOptions,
        opts: CartesiaSTTOptions,
        language: str,
        sample_rate: int,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._opts = opts
        self._language = language

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError as e:
            raise RuntimeError("websockets is required for Cartesia STT") from e

        request_id = shortuuid()
        tracker = LatencyTracker.get()
        state: dict[str, Any] = {
            "started": False,
            "latest_text": "",
            "stt_timing_started": False,
        }

        ws_url = (
            f"{self._opts.ws_url}?model={self._opts.model}"
            f"&language={self._language}"
            f"&encoding={self._opts.encoding}"
            f"&sample_rate={self._opts.sample_rate}"
            f"&min_volume={self._opts.min_volume}"
            f"&max_silence_duration_secs={self._opts.max_silence_duration_secs}"
        )

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
            sender = asyncio.create_task(
                self._sender(ws, request_id=request_id, tracker=tracker, state=state),
                name="cartesia-stt-sender",
            )
            receiver = asyncio.create_task(
                self._receiver(ws, request_id=request_id, tracker=tracker, state=state),
                name="cartesia-stt-receiver",
            )

            try:
                await asyncio.gather(sender, receiver)
            finally:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()

    async def _sender(self, ws: Any, *, request_id: str, tracker: LatencyTracker, state: dict[str, Any]) -> None:
        async for item in self._input_ch:
            # `RecognizeStream` yields either audio frames or a flush sentinel.
            # Use getattr to avoid type-checker complaints about the sentinel.
            data = getattr(item, "data", None)
            if data is not None:
                if not state.get("stt_timing_started"):
                    state["stt_timing_started"] = True
                    tracker.stt_started(request_id=request_id)
                await ws.send(bytes(data))
                continue

            # Flush sentinel
            await ws.send("finalize")

        await ws.send("done")

    async def _receiver(
        self,
        ws: Any,
        *,
        request_id: str,
        tracker: LatencyTracker,
        state: dict[str, Any],
    ) -> None:
        async for message in ws:
            if not isinstance(message, str):
                continue

            payload = json.loads(message)
            msg_type = payload.get("type")

            if msg_type == "error":
                raise RuntimeError(payload.get("message") or payload)

            if msg_type == "transcript":
                text = payload.get("text") or ""
                is_final = bool(payload.get("is_final") or payload.get("final"))

                if text:
                    state["latest_text"] = text

                if text and not state["started"]:
                    state["started"] = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.START_OF_SPEECH,
                            request_id=request_id,
                            alternatives=[],
                        )
                    )

                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT if is_final else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=request_id,
                        alternatives=[
                            stt.SpeechData(
                                text=text,
                                language=self._language,
                                confidence=1.0,
                            )
                        ],
                    )
                )

            if msg_type == "flush_done":
                if state.get("latest_text"):
                    tracker.stt_finished(str(state.get("latest_text")), request_id=request_id)
                state["latest_text"] = ""
                state["started"] = False
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_SPEECH,
                        request_id=request_id,
                        alternatives=[],
                    )
                )

            if msg_type == "done":
                break
