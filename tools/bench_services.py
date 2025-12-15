"""Service latency benchmark for vagent.

Runs a direct benchmark of:
- STT (faster-whisper local model)
- LLM (Ollama OpenAI-compatible endpoint)
- TTS (Kokoro-FastAPI OpenAI-compatible endpoint)

This is *not* a LiveKit call-flow test; it benchmarks the underlying services
that the agent uses.

Examples:
  python tools/bench_services.py --audio audio/sample.wav
  python tools/bench_services.py --stt-from-tts

Notes:
- For STT, only WAV input is supported by default.
- For LLM, measures time-to-first-token (TTFT) and total time.
- For TTS, measures time-to-first-audio-chunk (TTFA) and total stream time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import openai
from faster_whisper import WhisperModel

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class Timing:
    name: str
    ms: float


def _ms(dt_s: float) -> float:
    return dt_s * 1000.0


def _now() -> float:
    return time.perf_counter()


def _load_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file to mono float32 in range [-1, 1]."""
    try:
        from scipy.io import wavfile

        sr, data = wavfile.read(str(path))

        if data.ndim == 2:
            data = data.mean(axis=1)

        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32 or data.dtype == np.float64:
            audio = data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported WAV dtype: {data.dtype}")

        return audio, int(sr)
    except Exception as e:
        print(f"Error loading WAV file {path}: {e}")
        sys.exit(1)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)

    num_samples = int(len(audio) * dst_sr / src_sr)
    if num_samples <= 0:
        return np.zeros((0,), dtype=np.float32)

    # Prefer polyphase resampling for speed + quality.
    # scipy.signal.resample is FFT-based and tends to be slower for long streams.
    from scipy import signal

    try:
        import math

        g = math.gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)
    except Exception:
        return signal.resample(audio, num_samples).astype(np.float32)


def bench_stt(
    *,
    model_path: Path,
    audio_16k: np.ndarray,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    without_timestamps: bool,
    device: str,
    compute_type: str,
) -> tuple[str, list[Timing]]:
    timings: list[Timing] = []

    try:
        t0 = _now()
        model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute_type,
        )
        timings.append(Timing("stt_model_load", _ms(_now() - t0)))

        t1 = _now()
        segments, info = model.transcribe(
            audio_16k,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            without_timestamps=without_timestamps,
        )
        text_parts: list[str] = []
        for seg in segments:
            text_parts.append(seg.text.strip())
        transcript = " ".join([p for p in text_parts if p])
        timings.append(Timing("stt_transcribe", _ms(_now() - t1)))

        if info.language:
            transcript = transcript

        return transcript, timings
    except Exception as e:
        print(f"STT benchmark failed: {e}")
        return "", timings


async def bench_llm(*, client: openai.AsyncClient, model: str, prompt: str) -> tuple[str, list[Timing]]:
    timings: list[Timing] = []

    t0 = _now()
    first_token_at: float | None = None
    chunks: list[str] = []

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        async for event in stream:
            delta = None
            try:
                delta = event.choices[0].delta.content
            except Exception:
                delta = None

            if delta:
                if first_token_at is None:
                    first_token_at = _now()
                chunks.append(delta)
    except Exception as e:
        print(f"LLM benchmark failed: {e}")

    t_end = _now()

    if first_token_at is not None:
        timings.append(Timing("llm_ttft", _ms(first_token_at - t0)))
    else:
        timings.append(Timing("llm_ttft", float("nan")))

    timings.append(Timing("llm_total", _ms(t_end - t0)))

    return "".join(chunks).strip(), timings


async def bench_tts(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
    voice: str,
    speed: float,
    chunk_size: int,
) -> tuple[bytes, list[Timing]]:
    timings: list[Timing] = []

    url = f"{base_url.rstrip('/')}/audio/speech"
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": voice,
        "speed": speed,
        "response_format": "pcm",
    }

    t0 = _now()
    first_audio_at: float | None = None
    out = bytearray()

    try:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=chunk_size):
                if not chunk:
                    continue
                if first_audio_at is None:
                    first_audio_at = _now()
                out.extend(chunk)
    except Exception as e:
        print(f"TTS benchmark failed: {e}")

    t_end = _now()

    if first_audio_at is not None:
        timings.append(Timing("tts_ttfa", _ms(first_audio_at - t0)))
    else:
        timings.append(Timing("tts_ttfa", float("nan")))

    timings.append(Timing("tts_total", _ms(t_end - t0)))
    timings.append(Timing("tts_bytes", float(len(out))))

    return bytes(out), timings


def _pcm16le_to_float32_mono(pcm: bytes) -> np.ndarray:
    audio_i16 = np.frombuffer(pcm, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0).copy()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default=None, help="Path to WAV file for STT benchmark")
    parser.add_argument(
        "--stt-from-tts",
        action="store_true",
        help="Generate audio via TTS and feed it into STT (no external audio file needed)",
    )

    parser.add_argument("--whisper-model", type=str, default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="float16")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--vad-filter",
        dest="vad_filter",
        action="store_true",
        help="Enable VAD filtering inside Whisper (can reduce hallucinations; may add latency)",
    )
    parser.add_argument(
        "--no-vad-filter",
        dest="vad_filter",
        action="store_false",
        help="Disable VAD filtering inside Whisper (often faster for already-trimmed audio)",
    )
    parser.set_defaults(vad_filter=True)
    parser.add_argument(
        "--without-timestamps",
        dest="without_timestamps",
        action="store_true",
        help="Disable timestamps in Whisper decoding (often faster)",
    )
    parser.add_argument(
        "--with-timestamps",
        dest="without_timestamps",
        action="store_false",
        help="Enable timestamps in Whisper decoding",
    )
    parser.set_defaults(without_timestamps=True)

    parser.add_argument("--ollama-base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--ollama-model", type=str, default="gemma3:4b")

    parser.add_argument("--kokoro-base-url", type=str, default="http://localhost:8880/v1")
    parser.add_argument("--kokoro-voice", type=str, default="af_heart")
    parser.add_argument("--kokoro-speed", type=float, default=1.0)

    parser.add_argument("--prompt", type=str, default="Hey, what's up?")
    parser.add_argument("--tts-text", type=str, default="Hello! This is a latency benchmark.")
    parser.add_argument("--chunk-size", type=int, default=4800)

    args = parser.parse_args()

    # Reuse clients so we don't pay connection setup costs each call.
    tts_http = httpx.AsyncClient(
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=10),
    )
    llm_http = httpx.AsyncClient(
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=10),
    )
    llm_client = openai.AsyncClient(
        api_key="ollama",
        base_url=args.ollama_base_url,
        max_retries=1,
        http_client=llm_http,
    )

    try:
        model_path = Path(args.whisper_model)
        timings: list[Timing] = []

        # --- Build STT input ---
        if args.audio:
            audio_path = Path(args.audio)
            if not audio_path.exists():
                raise SystemExit(f"Audio file not found: {audio_path}")
            audio, sr = _load_wav_mono_float32(audio_path)
            audio_16k = _resample(audio, sr, 16000)
        elif args.stt_from_tts:
            pcm, tts_timings = await bench_tts(
                client=tts_http,
                base_url=args.kokoro_base_url,
                text=args.tts_text,
                voice=args.kokoro_voice,
                speed=args.kokoro_speed,
                chunk_size=args.chunk_size,
            )
            timings.extend(tts_timings)
            audio_24k = _pcm16le_to_float32_mono(pcm)
            audio_16k = _resample(audio_24k, 24000, 16000)
        else:
            audio_16k = None

        # --- STT ---
        transcript: str | None = None
        if audio_16k is not None:
            transcript, stt_timings = bench_stt(
                model_path=model_path,
                audio_16k=audio_16k,
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=args.vad_filter,
                without_timestamps=args.without_timestamps,
                device=args.device,
                compute_type=args.compute_type,
            )
            timings.extend(stt_timings)

        # --- LLM ---
        prompt = transcript if transcript else args.prompt
        llm_text, llm_timings = await bench_llm(
            client=llm_client,
            model=args.ollama_model,
            prompt=prompt,
        )
        timings.extend(llm_timings)

        # --- TTS ---
        pcm2, tts2_timings = await bench_tts(
            client=tts_http,
            base_url=args.kokoro_base_url,
            text=llm_text or args.tts_text,
            voice=args.kokoro_voice,
            speed=args.kokoro_speed,
            chunk_size=args.chunk_size,
        )
        timings.extend(tts2_timings)

        # --- Summary ---
        out = {
            "transcript": transcript,
            "llm_text": (llm_text[:200] + "...") if llm_text and len(llm_text) > 200 else llm_text,
            "timings": [{"name": t.name, "ms": t.ms} for t in timings],
        }

        # Compute a convenience e2e from STT transcribe start to TTS first-audio of the last TTS.
        # (This is a service benchmark, not a client mic->speaker measurement.)
        def get(name: str) -> float | None:
            for t in timings:
                if t.name == name:
                    return t.ms
            return None

        # best-effort: if we did STT, include it; else only LLM+TTS.
        stt_transcribe = get("stt_transcribe")
        llm_ttft = get("llm_ttft")
        tts_ttfa_last = None
        for t in reversed(timings):
            if t.name == "tts_ttfa":
                tts_ttfa_last = t.ms
                break

        if stt_transcribe is not None and llm_ttft is not None and tts_ttfa_last is not None:
            out["approx_e2e_to_first_audio_ms"] = stt_transcribe + llm_ttft + tts_ttfa_last
        elif llm_ttft is not None and tts_ttfa_last is not None:
            out["approx_e2e_to_first_audio_ms"] = llm_ttft + tts_ttfa_last

        print(json.dumps(out, indent=2))
        return 0
    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        return 1
    finally:
        await tts_http.aclose()
        await llm_http.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
