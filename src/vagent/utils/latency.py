"""Latency instrumentation for vagent.

This module provides lightweight end-to-end timing across the voice pipeline.
It intentionally measures *agent-side* latencies (STT runtime, time until TTS starts,
TTS time-to-first-audio-chunk, total TTS streaming time).

Enable with environment variable: `VAGENT_LATENCY=1`.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("vagent-latency")


def _enabled() -> bool:
    return os.getenv("VAGENT_LATENCY", "").strip().lower() in {"1", "true", "yes", "on"}


def _pretty_enabled() -> bool:
    return os.getenv("VAGENT_LATENCY_PRETTY", "").strip().lower() in {"1", "true", "yes", "on"}


def _color_enabled() -> bool:
    v = os.getenv("VAGENT_LATENCY_COLOR", "").strip().lower()
    if v in {"0", "false", "no", "off"}:
        return False
    if v in {"1", "true", "yes", "on"}:
        return True
    if os.getenv("NO_COLOR") is not None:
        return False
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def _ms(dt_s: float | None) -> float | None:
    if dt_s is None:
        return None
    return dt_s * 1000.0


@dataclass
class _Turn:
    turn_id: int
    stt_request_id: str | None = None
    tts_request_id: str | None = None
    stt_start: float | None = None
    stt_end: float | None = None
    transcript_chars: int | None = None

    tts_start: float | None = None
    tts_first_audio: float | None = None
    tts_end: float | None = None
    tts_input_chars: int | None = None


class LatencyTracker:
    """Tracks a rolling window of turn latencies.

    Notes:
    - Correlation is best-effort: the next TTS call is assumed to correspond to the
      most recent STT completion that hasn't been paired yet.
    - Measures exclude network playback latency on the client.
    """

    _instance: "LatencyTracker | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.enabled = _enabled()
        self._pretty = _pretty_enabled()
        self._color = _color_enabled()
        self._lock = threading.Lock()
        self._turn_seq = 0
        # Turns waiting for TTS to begin (paired by arrival order as a fallback).
        self._awaiting_tts: deque[_Turn] = deque(maxlen=50)
        # Active turns keyed by request id (best-effort correlation).
        self._stt_by_request: dict[str, _Turn] = {}
        self._tts_by_request: dict[str, _Turn] = {}
        # Most recent turn touched (used only as last-resort fallback).
        self._latest: _Turn | None = None
        self._history: deque[_Turn] = deque(maxlen=50)

    @classmethod
    def get(cls) -> "LatencyTracker":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _new_turn_locked(self) -> _Turn:
        self._turn_seq += 1
        turn = _Turn(turn_id=self._turn_seq)
        self._latest = turn
        return turn

    def stt_started(self, request_id: str | None = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn = self._new_turn_locked()
            turn.stt_request_id = request_id
            turn.stt_start = now
            if request_id:
                self._stt_by_request[request_id] = turn

    def stt_finished(self, transcript: str, request_id: str | None = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn: _Turn | None = None
            if request_id:
                turn = self._stt_by_request.get(request_id)
            if turn is None:
                # Fall back to the latest turn with an STT start and no STT end.
                latest = self._latest
                if latest is not None and latest.stt_start is not None and latest.stt_end is None:
                    turn = latest
            if turn is None:
                turn = self._new_turn_locked()
            turn.stt_end = now
            turn.transcript_chars = len(transcript or "")
            self._latest = turn

            # Queue for pairing with the next TTS start.
            if turn not in self._awaiting_tts:
                self._awaiting_tts.append(turn)

    def tts_started(self, text: str, request_id: str | None = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn: _Turn | None = None

            if request_id:
                turn = self._tts_by_request.get(request_id)

            if turn is None and self._awaiting_tts:
                # Pair with the oldest completed STT not yet paired.
                turn = self._awaiting_tts.popleft()

            if turn is None:
                turn = self._new_turn_locked()

            turn.tts_request_id = request_id
            turn.tts_start = now
            if text:
                turn.tts_input_chars = len(text)
            else:
                # Don't overwrite with 0 if input is incremental and starts with empty.
                turn.tts_input_chars = turn.tts_input_chars

            self._latest = turn
            if request_id:
                self._tts_by_request[request_id] = turn

    def tts_first_audio_chunk(self, request_id: str | None = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn: _Turn | None = None
            if request_id:
                turn = self._tts_by_request.get(request_id)
            if turn is None:
                turn = self._latest
            if turn is None or turn.tts_start is None:
                return
            if turn.tts_first_audio is None:
                turn.tts_first_audio = now
                self._latest = turn

    def tts_finished(self, request_id: str | None = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn: _Turn | None = None
            if request_id:
                turn = self._tts_by_request.get(request_id)
            if turn is None:
                turn = self._latest
            if turn is None:
                return
            turn.tts_end = now
            self._history.append(turn)

            # Cleanup maps for this turn.
            if turn.stt_request_id:
                self._stt_by_request.pop(turn.stt_request_id, None)
            if turn.tts_request_id:
                self._tts_by_request.pop(turn.tts_request_id, None)

            self._latest = None

        self._log_turn(turn)

    def _log_turn(self, turn: _Turn) -> None:
        stt_ms = _ms((turn.stt_end - turn.stt_start) if turn.stt_end and turn.stt_start else None)
        llm_ms = _ms((turn.tts_start - turn.stt_end) if turn.tts_start and turn.stt_end else None)
        tts_ttft_ms = _ms(
            (turn.tts_first_audio - turn.tts_start)
            if turn.tts_first_audio and turn.tts_start
            else None
        )
        tts_total_ms = _ms((turn.tts_end - turn.tts_start) if turn.tts_end and turn.tts_start else None)
        e2e_first_ms = _ms(
            (turn.tts_first_audio - turn.stt_start)
            if turn.tts_first_audio and turn.stt_start
            else None
        )
        e2e_end_ms = _ms((turn.tts_end - turn.stt_start) if turn.tts_end and turn.stt_start else None)

        logger.info(
            "turn=%s stt_ms=%s llm_ms=%s tts_ttft_ms=%s tts_total_ms=%s e2e_first_audio_ms=%s e2e_end_ms=%s transcript_chars=%s tts_input_chars=%s",
            turn.turn_id,
            _fmt_ms(stt_ms),
            _fmt_ms(llm_ms),
            _fmt_ms(tts_ttft_ms),
            _fmt_ms(tts_total_ms),
            _fmt_ms(e2e_first_ms),
            _fmt_ms(e2e_end_ms),
            turn.transcript_chars,
            turn.tts_input_chars,
        )

        if self._pretty:
            logger.info(
                "%s",
                _format_pretty_turn(
                    turn_id=turn.turn_id,
                    stt_ms=stt_ms,
                    llm_ms=llm_ms,
                    tts_ttft_ms=tts_ttft_ms,
                    tts_total_ms=tts_total_ms,
                    e2e_first_ms=e2e_first_ms,
                    e2e_end_ms=e2e_end_ms,
                    color=self._color,
                ),
            )

        avg = self.summary()
        if avg:
            logger.info(
                "avg(last_%s) stt_ms=%s llm_ms=%s tts_ttft_ms=%s tts_total_ms=%s e2e_first_audio_ms=%s e2e_end_ms=%s",
                avg["n"],
                _fmt_ms(avg["stt_ms"]),
                _fmt_ms(avg["llm_ms"]),
                _fmt_ms(avg["tts_ttft_ms"]),
                _fmt_ms(avg["tts_total_ms"]),
                _fmt_ms(avg["e2e_first_audio_ms"]),
                _fmt_ms(avg["e2e_end_ms"]),
            )

    def summary(self) -> dict[str, float] | None:
        if not self.enabled:
            return None

        with self._lock:
            turns = list(self._history)

        if not turns:
            return None

        def avg_ms(values: list[float | None]) -> float:
            present = [v for v in values if v is not None]
            if not present:
                return float("nan")
            return sum(present) / len(present)

        stt_list: list[float | None] = []
        llm_list: list[float | None] = []
        tts_ttft_list: list[float | None] = []
        tts_total_list: list[float | None] = []
        e2e_first_list: list[float | None] = []
        e2e_end_list: list[float | None] = []

        for t in turns:
            stt_list.append(_ms((t.stt_end - t.stt_start) if t.stt_end and t.stt_start else None))
            llm_list.append(_ms((t.tts_start - t.stt_end) if t.tts_start and t.stt_end else None))
            tts_ttft_list.append(_ms((t.tts_first_audio - t.tts_start) if t.tts_first_audio and t.tts_start else None))
            tts_total_list.append(_ms((t.tts_end - t.tts_start) if t.tts_end and t.tts_start else None))
            e2e_first_list.append(_ms((t.tts_first_audio - t.stt_start) if t.tts_first_audio and t.stt_start else None))
            e2e_end_list.append(_ms((t.tts_end - t.stt_start) if t.tts_end and t.stt_start else None))

        return {
            "n": float(len(turns)),
            "stt_ms": avg_ms(stt_list),
            "llm_ms": avg_ms(llm_list),
            "tts_ttft_ms": avg_ms(tts_ttft_list),
            "tts_total_ms": avg_ms(tts_total_list),
            "e2e_first_audio_ms": avg_ms(e2e_first_list),
            "e2e_end_ms": avg_ms(e2e_end_list),
        }


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "na"
    if value != value:  # NaN
        return "nan"
    return f"{value:.0f}"


def _format_pretty_turn(
    *,
    turn_id: int,
    stt_ms: float | None,
    llm_ms: float | None,
    tts_ttft_ms: float | None,
    tts_total_ms: float | None,
    e2e_first_ms: float | None,
    e2e_end_ms: float | None,
    color: bool,
) -> str:
    def c(code: str) -> str:
        if not color:
            return ""
        return f"\x1b[{code}m"

    reset = c("0")
    dim = c("2")
    bold = c("1")
    cyan = c("36")
    green = c("32")
    yellow = c("33")
    magenta = c("35")

    def ms(v: float | None) -> str:
        return _fmt_ms(v)

    # One-line, fixed order, easy to scan.
    return (
        f"{dim}latency{reset} "
        f"{bold}turn{reset}={cyan}{turn_id}{reset} "
        f"stt={green}{ms(stt_ms)}{reset}ms "
        f"llm={yellow}{ms(llm_ms)}{reset}ms "
        f"tts_ttft={magenta}{ms(tts_ttft_ms)}{reset}ms "
        f"tts_total={magenta}{ms(tts_total_ms)}{reset}ms "
        f"e2e_first={cyan}{ms(e2e_first_ms)}{reset}ms "
        f"e2e_end={cyan}{ms(e2e_end_ms)}{reset}ms"
    )
