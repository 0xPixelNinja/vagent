"""Latency wrappers for LiveKit-provided STT/TTS plugins.

Why this exists:
- Some providers (e.g., LiveKit's Cartesia plugin) live outside this repo.
- We still want consistent latency metrics without modifying external packages.

Approach:
- Monkey-patch the returned stream objects (`_run`) for TTS to detect request_id,
  time-to-first-audio, and total stream time.
- Monkey-patch STT recognize/stream objects to infer start/finish from emitted
  `SpeechEvent`s.

Enable metrics via `VAGENT_LATENCY=1`.
Optional formatting:
- `VAGENT_LATENCY_PRETTY=1`
- `VAGENT_LATENCY_COLOR=1`
"""

from __future__ import annotations

import types
from typing import Any, Awaitable, Callable, cast

from livekit.agents import stt, tts

from vagent.utils.latency import LatencyTracker


def instrument_livekit_tts(tts_obj: Any) -> Any:
    """Wrap a LiveKit TTS provider instance to emit vagent latency metrics.

    This function returns the same object instance with patched `synthesize()` and
    `stream()` methods.
    """

    if getattr(tts_obj, "_vagent_latency_instrumented", False):
        return tts_obj

    tracker = LatencyTracker.get()
    if not tracker.enabled:
        return tts_obj

    def patch_stream_run(stream_obj: Any, *, input_text: str | None) -> Any:
        if stream_obj is None:
            return stream_obj
        if getattr(stream_obj, "_vagent_latency_instrumented", False):
            return stream_obj

        orig_run = getattr(stream_obj, "_run", None)
        if orig_run is None:
            return stream_obj

        async def _run(self_stream: Any, output_emitter: Any) -> None:
            state: dict[str, Any] = {
                "request_id": None,
                "started": False,
                "first_audio": False,
            }

            class _EmitterProxy:
                def initialize(self, *args: Any, **kwargs: Any) -> Any:
                    request_id = kwargs.get("request_id")
                    if request_id is None and args:
                        request_id = args[0]
                    state["request_id"] = request_id

                    if not state["started"]:
                        tracker.tts_started(input_text or "", request_id=request_id)
                        state["started"] = True

                    return output_emitter.initialize(*args, **kwargs)

                def push(self, data: bytes) -> Any:
                    if data and not state["first_audio"]:
                        tracker.tts_first_audio_chunk(request_id=state.get("request_id"))
                        state["first_audio"] = True
                    return output_emitter.push(data)

                def __getattr__(self, name: str) -> Any:
                    return getattr(output_emitter, name)

            emitter = _EmitterProxy()

            try:
                await orig_run(emitter)
            finally:
                # If initialize() never fired, still close the turn.
                if not state["started"]:
                    tracker.tts_started(input_text or "", request_id=state.get("request_id"))
                tracker.tts_finished(request_id=state.get("request_id"))

        # Always bind with the expected (self, output_emitter) signature.
        stream_obj._run = types.MethodType(_run, stream_obj)
        stream_obj._vagent_latency_instrumented = True
        return stream_obj

    orig_synthesize = getattr(tts_obj, "synthesize", None)
    if callable(orig_synthesize):

        def synthesize(text: str, *args: Any, **kwargs: Any) -> Any:
            stream_obj = orig_synthesize(text, *args, **kwargs)
            return patch_stream_run(stream_obj, input_text=text)

        tts_obj.synthesize = synthesize  # type: ignore[assignment]

    orig_stream = getattr(tts_obj, "stream", None)
    if callable(orig_stream):

        def stream(*args: Any, **kwargs: Any) -> Any:
            stream_obj = orig_stream(*args, **kwargs)
            # Streaming input: we may not know full text here.
            return patch_stream_run(stream_obj, input_text="")

        tts_obj.stream = stream  # type: ignore[assignment]

    tts_obj._vagent_latency_instrumented = True
    return tts_obj


def instrument_livekit_stt(stt_obj: Any) -> Any:
    """Wrap a LiveKit STT provider instance to emit vagent latency metrics.

    Returns the same object instance with patched `stream()` and (best-effort)
    `_recognize_impl()`.
    """

    if getattr(stt_obj, "_vagent_latency_instrumented", False):
        return stt_obj

    tracker = LatencyTracker.get()
    if not tracker.enabled:
        return stt_obj

    def patch_recognize_stream(rs: Any) -> Any:
        if rs is None:
            return rs
        if getattr(rs, "_vagent_latency_instrumented", False):
            return rs

        # Best-effort: infer timing from events emitted by the stream.
        state_by_req: dict[str, dict[str, Any]] = {}

        event_ch = getattr(rs, "_event_ch", None)
        if event_ch is not None and hasattr(event_ch, "send_nowait"):
            orig_send = event_ch.send_nowait

            def send_nowait(evt: stt.SpeechEvent) -> Any:
                try:
                    req_id = getattr(evt, "request_id", None)
                    evt_type = getattr(evt, "type", None)
                    if req_id:
                        s = state_by_req.setdefault(req_id, {"started": False, "latest_text": ""})

                        # Capture text whenever present
                        alts = getattr(evt, "alternatives", None) or []
                        if alts:
                            text = getattr(alts[0], "text", "") or ""
                            if text:
                                s["latest_text"] = text

                        if not s["started"] and evt_type in {
                            stt.SpeechEventType.START_OF_SPEECH,
                            stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            stt.SpeechEventType.FINAL_TRANSCRIPT,
                        }:
                            s["started"] = True
                            tracker.stt_started(request_id=req_id)

                        if evt_type in {stt.SpeechEventType.FINAL_TRANSCRIPT, stt.SpeechEventType.END_OF_SPEECH}:
                            tracker.stt_finished(s.get("latest_text", ""), request_id=req_id)
                except Exception:
                    # Never break the audio pipeline for telemetry.
                    pass

                return orig_send(evt)

            event_ch.send_nowait = send_nowait  # type: ignore[assignment]

        rs._vagent_latency_instrumented = True
        return rs

    # Patch `stream()` (streaming STT)
    orig_stream = getattr(stt_obj, "stream", None)
    if callable(orig_stream):

        def stream(*args: Any, **kwargs: Any) -> Any:
            rs = orig_stream(*args, **kwargs)
            return patch_recognize_stream(rs)

        stt_obj.stream = stream  # type: ignore[assignment]

    # Patch batch recognize (best-effort)
    orig_recognize_impl = getattr(stt_obj, "_recognize_impl", None)
    if callable(orig_recognize_impl):

        recognize_impl = cast(Callable[..., Awaitable[stt.SpeechEvent]], orig_recognize_impl)

        async def _recognize_impl(*args: Any, **kwargs: Any) -> stt.SpeechEvent:
            tracker.stt_started(request_id=None)
            evt: stt.SpeechEvent = await recognize_impl(*args, **kwargs)

            text = ""
            try:
                alts = getattr(evt, "alternatives", None) or []
                if alts:
                    text = getattr(alts[0], "text", "") or ""
            except Exception:
                pass

            tracker.stt_finished(text, request_id=None)
            return evt

        stt_obj._recognize_impl = _recognize_impl  # type: ignore[assignment]

    stt_obj._vagent_latency_instrumented = True
    return stt_obj


def _is_bound_method(fn: Any) -> bool:
    try:
        return isinstance(fn, types.MethodType)
    except Exception:
        return False
