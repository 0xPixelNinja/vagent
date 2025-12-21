"""
WhisperLiveKit STT plugin for LiveKit Agents.

Streams audio to a WhisperLiveKit WebSocket server for real-time transcription.
The server must be started with --pcm-input flag to accept raw PCM audio.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, shortuuid

from vagent.utils.latency import LatencyTracker

logger = logging.getLogger("stt-whisperlivekit")


@dataclass(frozen=True)
class WhisperLiveKitOptions:
    """Configuration for WhisperLiveKit STT."""
    ws_url: str = "ws://localhost:8000/asr"
    language: str = "auto"
    sample_rate: int = 16000


class WhisperLiveKitSTT(stt.STT):
    """
    WhisperLiveKit streaming STT client for LiveKit Agents.
    
    Connects to a WhisperLiveKit WebSocket server (must use --pcm-input mode).
    """

    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8000/asr",
        language: str = "auto",
        sample_rate: int = 16000,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        self._opts = WhisperLiveKitOptions(
            ws_url=ws_url.rstrip("/"),
            language=language,
            sample_rate=sample_rate,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "WhisperLiveKitRecognizeStream":
        lang = self._opts.language if language is NOT_GIVEN else language
        return WhisperLiveKitRecognizeStream(
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
        """Non-streaming recognition (batch mode)."""
        request_id = shortuuid()
        LatencyTracker.get().stt_started(request_id=request_id)

        try:
            import websockets
        except ImportError as e:
            raise RuntimeError("websockets is required for WhisperLiveKit STT") from e

        try:
            frames = buffer if isinstance(buffer, list) else [buffer]
            if not frames:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[],
                )

            all_data = b"".join(bytes(frame.data) for frame in frames)
            lang = self._opts.language if language is NOT_GIVEN else language
            final_text = ""

            async with websockets.connect(
                self._opts.ws_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                # Wait for config
                await asyncio.wait_for(ws.recv(), timeout=5.0)
                
                # Send audio and signal end
                await ws.send(all_data)
                await ws.send(b"")

                # Collect results
                async for message in ws:
                    if isinstance(message, str):
                        payload = json.loads(message)
                        if payload.get("type") == "ready_to_stop":
                            break
                        if payload.get("status") == "active_transcription":
                            lines = payload.get("lines", [])
                            buffer_text = payload.get("buffer_transcription", "")
                            line_texts = [
                                (l.get("text", "") if isinstance(l, dict) else l)
                                for l in lines
                            ]
                            final_text = " ".join(filter(None, line_texts)).strip()
                            if buffer_text:
                                final_text = f"{final_text} {buffer_text}".strip()

            LatencyTracker.get().stt_finished(final_text, request_id=request_id)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[
                    stt.SpeechData(
                        text=final_text,
                        language=lang or "en",
                        confidence=1.0,
                    )
                ],
            )
        except Exception as e:
            logger.error(f"WhisperLiveKit recognition failed: {e}")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[],
            )


class WhisperLiveKitRecognizeStream(stt.RecognizeStream):
    """Streaming recognition for WhisperLiveKit."""

    def __init__(
        self,
        *,
        stt: WhisperLiveKitSTT,
        conn_options: APIConnectOptions,
        opts: WhisperLiveKitOptions,
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
            raise RuntimeError("websockets is required for WhisperLiveKit STT") from e

        request_id = shortuuid()
        tracker = LatencyTracker.get()
        state: dict[str, Any] = {
            "started": False,
            "latest_text": "",
            "stt_timing_started": False,
            "committed_text": "",  # Track cumulative committed text
        }

        try:
            async with websockets.connect(
                self._opts.ws_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                # Wait for config message from server
                try:
                    config_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    config = json.loads(config_msg)
                    logger.debug(f"WhisperLiveKit config: {config}")
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for WhisperLiveKit config")

                sender = asyncio.create_task(
                    self._sender(ws, request_id=request_id, tracker=tracker, state=state),
                    name="whisperlivekit-stt-sender",
                )
                receiver = asyncio.create_task(
                    self._receiver(ws, request_id=request_id, tracker=tracker, state=state),
                    name="whisperlivekit-stt-receiver",
                )

                try:
                    await asyncio.gather(sender, receiver)
                finally:
                    for task in (sender, receiver):
                        if not task.done():
                            task.cancel()

        except Exception as e:
            logger.error(f"WhisperLiveKit WebSocket error: {e}")
            raise

    async def _sender(
        self,
        ws: Any,
        *,
        request_id: str,
        tracker: LatencyTracker,
        state: dict[str, Any],
    ) -> None:
        """Send audio frames to WhisperLiveKit server."""
        async for item in self._input_ch:
            data = getattr(item, "data", None)
            if data is not None:
                if not state.get("stt_timing_started"):
                    state["stt_timing_started"] = True
                    tracker.stt_started(request_id=request_id)
                await ws.send(bytes(data))
                continue

            # Flush sentinel - signal end of utterance
            await ws.send(b"")

        # Final done signal
        await ws.send(b"")

    async def _receiver(
        self,
        ws: Any,
        *,
        request_id: str,
        tracker: LatencyTracker,
        state: dict[str, Any],
    ) -> None:
        """Receive transcription results from WhisperLiveKit server."""
        async for message in ws:
            if not isinstance(message, str):
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from WhisperLiveKit: {message}")
                continue

            msg_type = payload.get("type")

            # Handle done signal
            if msg_type == "ready_to_stop":
                if state["started"]:
                    if state.get("latest_text"):
                        tracker.stt_finished(str(state.get("latest_text")), request_id=request_id)
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.END_OF_SPEECH,
                            request_id=request_id,
                            alternatives=[],
                        )
                    )
                break

            # Handle errors
            if payload.get("status") == "error":
                error_msg = payload.get("error", "Unknown error")
                logger.error(f"WhisperLiveKit error: {error_msg}")
                continue

            # Handle transcription
            status = payload.get("status")
            if status in ("active_transcription", "no_audio_detected"):
                lines = payload.get("lines", [])
                buffer_text = payload.get("buffer_transcription", "")

                # Extract committed text from lines (these are cumulative from server)
                line_texts = []
                for line in lines:
                    if isinstance(line, dict):
                        text = line.get("text", "")
                        if text:
                            line_texts.append(text)
                    elif isinstance(line, str) and line:
                        line_texts.append(line)

                committed_text = "\n".join(line_texts).strip()
                prev_committed = state["committed_text"]
                
                # Detect new committed (final) text
                new_final_text = ""
                if committed_text and committed_text != prev_committed:
                    # Extract only the NEW portion
                    if prev_committed and committed_text.startswith(prev_committed):
                        new_final_text = committed_text[len(prev_committed):].strip()
                    else:
                        new_final_text = committed_text
                    state["committed_text"] = committed_text

                # Emit START_OF_SPEECH on first text
                if (new_final_text or buffer_text) and not state["started"]:
                    state["started"] = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.START_OF_SPEECH,
                            request_id=request_id,
                            alternatives=[],
                        )
                    )

                # Emit FINAL for new committed text only
                if new_final_text:
                    state["latest_text"] = new_final_text
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=request_id,
                            alternatives=[
                                stt.SpeechData(
                                    text=new_final_text,
                                    language=self._language if self._language != "auto" else "en",
                                    confidence=1.0,
                                )
                            ],
                        )
                    )
                    tracker.stt_finished(new_final_text, request_id=request_id)

                # Emit INTERIM for buffer text (in-progress, not yet committed)
                if buffer_text:
                    state["latest_text"] = buffer_text
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=request_id,
                            alternatives=[
                                stt.SpeechData(
                                    text=buffer_text,
                                    language=self._language if self._language != "auto" else "en",
                                    confidence=1.0,
                                )
                            ],
                        )
                    )

