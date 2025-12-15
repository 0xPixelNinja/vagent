"""Latency instrumentation for vagent.

This module provides lightweight end-to-end timing across the voice pipeline.
It intentionally measures *agent-side* latencies (STT runtime, time until TTS starts,
TTS time-to-first-audio-chunk, total TTS streaming time).

Enable with environment variable: `VAGENT_LATENCY=1`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("vagent-latency")


def _enabled() -> bool:
    return os.getenv("VAGENT_LATENCY", "").strip().lower() in {"1", "true", "yes", "on"}


def _ms(dt_s: float | None) -> float | None:
    if dt_s is None:
        return None
    return dt_s * 1000.0


@dataclass
class _Turn:
    turn_id: int
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
        self._lock = threading.Lock()
        self._turn_seq = 0
        self._pending: _Turn | None = None
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
        self._pending = turn
        return turn

    def stt_started(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn = self._pending
            if turn is None or turn.stt_start is not None and turn.tts_end is None:
                turn = self._new_turn_locked()
            turn.stt_start = now

    def stt_finished(self, transcript: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn = self._pending
            if turn is None:
                turn = self._new_turn_locked()
            turn.stt_end = now
            turn.transcript_chars = len(transcript or "")

    def tts_started(self, text: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn = self._pending
            if turn is None or turn.tts_start is not None:
                turn = self._new_turn_locked()
            turn.tts_start = now
            turn.tts_input_chars = len(text or "")

    def tts_first_audio_chunk(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn = self._pending
            if turn is None:
                return
            if turn.tts_first_audio is None:
                turn.tts_first_audio = now

    def tts_finished(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            turn = self._pending
            if turn is None:
                return
            turn.tts_end = now
            self._history.append(turn)
            self._pending = None

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
