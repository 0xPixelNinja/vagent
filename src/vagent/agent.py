"""
Voice Agent using LiveKit Agents SDK v1.3
STT: faster-whisper (large-v3-turbo)
LLM: Ollama gemma3:4b
TTS: Kokoro-FastAPI or VibeVoice-Realtime (configurable)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
import httpx
import openai
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    cli,
    llm,
)
from livekit.plugins import silero, openai as lk_openai

from vagent.plugins import FasterWhisperSTT, KokoroTTS, VibeVoiceTTS
from vagent.utils.latency import LatencyTracker
from vagent.utils.lk_latency_wrap import instrument_livekit_stt, instrument_livekit_tts

try:
    # LiveKit plugin modules register themselves and must be imported on the main thread.
    from livekit.plugins import cartesia as lk_cartesia  # type: ignore

    _LK_CARTESIA_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover
    lk_cartesia = None  # type: ignore
    _LK_CARTESIA_IMPORT_ERROR = e

load_dotenv()

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

logging.getLogger("vagent-latency").setLevel(logging.INFO)

# Resolve models directory relative to project root
# agent.py is at src/vagent/agent.py, so 3 parents up = project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm models to reduce first-response latency."""
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")

    stt_engine = os.getenv("VAGENT_STT_ENGINE", "whisper").strip().lower()

    if stt_engine == "cartesia":
        try:
            cartesia_stt_model = os.getenv("VAGENT_CARTESIA_STT_MODEL", "ink-whisper")
            cartesia_stt_language = os.getenv("VAGENT_CARTESIA_STT_LANGUAGE", "en")
            cartesia_api_key = os.getenv("VAGENT_CARTESIA_API_KEY") or os.getenv("CARTESIA_API_KEY")
            if cartesia_api_key:
                os.environ.setdefault("CARTESIA_API_KEY", cartesia_api_key)

            try:
                if lk_cartesia is None:
                    raise RuntimeError(_LK_CARTESIA_IMPORT_ERROR or "Cartesia plugin not installed")

                proc.userdata["stt"] = lk_cartesia.STT(
                    model=cartesia_stt_model,
                    language=cartesia_stt_language,
                )
                # Add latency metrics even for LiveKit-provided plugins.
                if LatencyTracker.get().enabled:
                    proc.userdata["stt"] = instrument_livekit_stt(proc.userdata["stt"])
                logger.info("Using Cartesia STT (LiveKit plugin)")
            except Exception as e:
                logger.warning(f"LiveKit Cartesia STT plugin unavailable, falling back to custom: {e}")

                from vagent.plugins import CartesiaSTT

                cartesia_version = os.getenv("VAGENT_CARTESIA_VERSION", "2025-04-16")
                cartesia_stt_sr = int(os.getenv("VAGENT_CARTESIA_STT_SAMPLE_RATE", "16000"))
                cartesia_min_volume = float(os.getenv("VAGENT_CARTESIA_STT_MIN_VOLUME", "0.0"))
                cartesia_max_silence = float(os.getenv("VAGENT_CARTESIA_STT_MAX_SILENCE_SECS", "0.8"))

                proc.userdata["stt"] = CartesiaSTT(
                    api_key=cartesia_api_key or "",
                    model=cartesia_stt_model,
                    language=cartesia_stt_language,
                    cartesia_version=cartesia_version,
                    sample_rate=cartesia_stt_sr,
                    min_volume=cartesia_min_volume,
                    max_silence_duration_secs=cartesia_max_silence,
                )
                logger.info("Using Cartesia STT (custom plugin)")
        except Exception as e:
            logger.error(f"Failed to initialize Cartesia STT: {e}")
    else:
        # Latency/quality knobs
        # Default beam_size to 1 for speed
        beam_size = int(os.getenv("VAGENT_STT_BEAM_SIZE", "1"))
        vad_filter = os.getenv("VAGENT_STT_VAD_FILTER", "1").strip().lower() in {"1", "true", "yes", "on"}
        without_timestamps = os.getenv("VAGENT_STT_WITHOUT_TIMESTAMPS", "1").strip().lower() in {"1", "true", "yes", "on"}

        try:
            stt_instance = FasterWhisperSTT(
                model_path=str(MODELS_DIR),
                device="cuda",
                compute_type="float16",
                beam_size=beam_size,
                vad_filter=vad_filter,
                without_timestamps=without_timestamps,
            )
            stt_instance.load()  # Load model into VRAM immediately
            proc.userdata["stt"] = stt_instance
        except Exception as e:
            logger.error(f"Failed to initialize STT: {e}")

    # TTS engine selection: "kokoro" (default), "vibevoice", or "cartesia"
    tts_engine = os.getenv("VAGENT_TTS_ENGINE", "kokoro").strip().lower()
    
    try:
        if tts_engine == "cartesia":
            cartesia_model = os.getenv("VAGENT_CARTESIA_TTS_MODEL", "sonic-3")
            cartesia_voice_id = os.getenv("VAGENT_CARTESIA_VOICE_ID", "")
            cartesia_language = os.getenv("VAGENT_CARTESIA_TTS_LANGUAGE", "en")
            cartesia_api_key = os.getenv("VAGENT_CARTESIA_API_KEY") or os.getenv("CARTESIA_API_KEY")
            if cartesia_api_key:
                os.environ.setdefault("CARTESIA_API_KEY", cartesia_api_key)

            try:
                if lk_cartesia is None:
                    raise RuntimeError(_LK_CARTESIA_IMPORT_ERROR or "Cartesia plugin not installed")

                proc.userdata["tts"] = lk_cartesia.TTS(
                    model=cartesia_model,
                    voice=cartesia_voice_id,
                    language=cartesia_language,
                )
                # Add latency metrics even for LiveKit-provided plugins.
                if LatencyTracker.get().enabled:
                    proc.userdata["tts"] = instrument_livekit_tts(proc.userdata["tts"])
                logger.info("Using Cartesia TTS (LiveKit plugin)")
            except Exception as e:
                logger.warning(f"LiveKit Cartesia TTS plugin unavailable, falling back to custom: {e}")

                from vagent.plugins import CartesiaTTS

                cartesia_version = os.getenv("VAGENT_CARTESIA_VERSION", "2025-04-16")
                cartesia_sr = int(os.getenv("VAGENT_CARTESIA_TTS_SAMPLE_RATE", "24000"))
                cartesia_max_buffer = int(os.getenv("VAGENT_CARTESIA_TTS_MAX_BUFFER_DELAY_MS", "3000"))

                proc.userdata["tts"] = CartesiaTTS(
                    api_key=cartesia_api_key or "",
                    voice_id=cartesia_voice_id,
                    model_id=cartesia_model,
                    language=cartesia_language,
                    cartesia_version=cartesia_version,
                    sample_rate=cartesia_sr,
                    max_buffer_delay_ms=cartesia_max_buffer,
                )
                logger.info("Using Cartesia TTS (custom plugin)")
        elif tts_engine == "vibevoice":
            # VibeVoice-Realtime TTS via WebSocket
            vibevoice_url = os.getenv("VAGENT_VIBEVOICE_URL", "ws://localhost:3000")
            vibevoice_voice = os.getenv("VAGENT_VIBEVOICE_VOICE", "en-WHTest_man")
            vibevoice_cfg = float(os.getenv("VAGENT_VIBEVOICE_CFG", "1.5"))
            vibevoice_steps = int(os.getenv("VAGENT_VIBEVOICE_STEPS", "5"))
            
            proc.userdata["tts"] = VibeVoiceTTS(
                base_url=vibevoice_url,
                voice=vibevoice_voice,
                cfg_scale=vibevoice_cfg,
                inference_steps=vibevoice_steps,
            )
            logger.info(f"Using VibeVoice TTS at {vibevoice_url}")
        else:
            # Default: Kokoro TTS via HTTP
            kokoro_url = os.getenv("VAGENT_KOKORO_URL", "http://localhost:8880/v1")
            kokoro_voice = os.getenv("VAGENT_KOKORO_VOICE", "af_heart")
            
            proc.userdata["tts"] = KokoroTTS(
                base_url=kokoro_url,
                voice=kokoro_voice,
            )
            logger.info(f"Using Kokoro TTS at {kokoro_url}")
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")


server.setup_fnc = prewarm


class VoiceAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. Keep responses concise and natural "
                "for spoken conversation. Aim for 1-2 sentences unless more detail is needed."
            ),
        )

    async def on_enter(self):
        """Called when the agent is added to the session."""
        self.session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice agent."""

    try:
        # Ollama's OpenAI-compatible streaming
        ollama_http = httpx.AsyncClient(
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        )
        ollama_client = openai.AsyncClient(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            max_retries=2,
            http_client=ollama_http,
        )
        ollama_llm = lk_openai.LLM(model="gemma3:4b", client=ollama_client)
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return

    try:
        # Ensure models are available, falling back to lazy loading if prewarm failed
        vad = ctx.proc.userdata.get("vad")
        if not vad:
            vad = silero.VAD.load()
            
        stt = ctx.proc.userdata.get("stt")
        if not stt:
            # Fallback STT initialization if prewarm failed
            stt_engine = os.getenv("VAGENT_STT_ENGINE", "whisper").strip().lower()

            if stt_engine == "cartesia":
                cartesia_stt_model = os.getenv("VAGENT_CARTESIA_STT_MODEL", "ink-whisper")
                cartesia_stt_language = os.getenv("VAGENT_CARTESIA_STT_LANGUAGE", "en")
                cartesia_api_key = os.getenv("VAGENT_CARTESIA_API_KEY") or os.getenv("CARTESIA_API_KEY")
                if cartesia_api_key:
                    os.environ.setdefault("CARTESIA_API_KEY", cartesia_api_key)

                try:
                    if lk_cartesia is None:
                        raise RuntimeError(_LK_CARTESIA_IMPORT_ERROR or "Cartesia plugin not installed")

                    stt = lk_cartesia.STT(
                        model=cartesia_stt_model,
                        language=cartesia_stt_language,
                    )
                    if LatencyTracker.get().enabled:
                        stt = instrument_livekit_stt(stt)
                except Exception as e:
                    logger.warning(f"LiveKit Cartesia STT plugin unavailable, falling back to custom: {e}")

                    from vagent.plugins import CartesiaSTT

                    cartesia_version = os.getenv("VAGENT_CARTESIA_VERSION", "2025-04-16")
                    cartesia_stt_sr = int(os.getenv("VAGENT_CARTESIA_STT_SAMPLE_RATE", "16000"))
                    cartesia_min_volume = float(os.getenv("VAGENT_CARTESIA_STT_MIN_VOLUME", "0.0"))
                    cartesia_max_silence = float(os.getenv("VAGENT_CARTESIA_STT_MAX_SILENCE_SECS", "0.8"))

                    stt = CartesiaSTT(
                        api_key=cartesia_api_key or "",
                        model=cartesia_stt_model,
                        language=cartesia_stt_language,
                        cartesia_version=cartesia_version,
                        sample_rate=cartesia_stt_sr,
                        min_volume=cartesia_min_volume,
                        max_silence_duration_secs=cartesia_max_silence,
                    )
            else:
                beam_size = int(os.getenv("VAGENT_STT_BEAM_SIZE", "1"))
                vad_filter = os.getenv("VAGENT_STT_VAD_FILTER", "1").strip().lower() in {"1", "true", "yes", "on"}
                without_timestamps = os.getenv("VAGENT_STT_WITHOUT_TIMESTAMPS", "1").strip().lower() in {"1", "true", "yes", "on"}
                stt = FasterWhisperSTT(
                    model_path=str(MODELS_DIR),
                    device="cuda",
                    compute_type="float16",
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    without_timestamps=without_timestamps,
                )
                stt.load()

        tts = ctx.proc.userdata.get("tts")
        if not tts:
            # Fallback TTS initialization if prewarm failed
            tts_engine = os.getenv("VAGENT_TTS_ENGINE", "kokoro").strip().lower()
            
            if tts_engine == "cartesia":
                cartesia_model = os.getenv("VAGENT_CARTESIA_TTS_MODEL", "sonic-3")
                cartesia_voice_id = os.getenv("VAGENT_CARTESIA_VOICE_ID", "")
                cartesia_language = os.getenv("VAGENT_CARTESIA_TTS_LANGUAGE", "en")
                cartesia_api_key = os.getenv("VAGENT_CARTESIA_API_KEY") or os.getenv("CARTESIA_API_KEY")
                if cartesia_api_key:
                    os.environ.setdefault("CARTESIA_API_KEY", cartesia_api_key)

                try:
                    if lk_cartesia is None:
                        raise RuntimeError(_LK_CARTESIA_IMPORT_ERROR or "Cartesia plugin not installed")

                    tts = lk_cartesia.TTS(
                        model=cartesia_model,
                        voice=cartesia_voice_id,
                        language=cartesia_language,
                    )
                    if LatencyTracker.get().enabled:
                        tts = instrument_livekit_tts(tts)
                except Exception as e:
                    logger.warning(f"LiveKit Cartesia TTS plugin unavailable, falling back to custom: {e}")

                    from vagent.plugins import CartesiaTTS

                    cartesia_version = os.getenv("VAGENT_CARTESIA_VERSION", "2025-04-16")
                    cartesia_sr = int(os.getenv("VAGENT_CARTESIA_TTS_SAMPLE_RATE", "24000"))
                    cartesia_max_buffer = int(os.getenv("VAGENT_CARTESIA_TTS_MAX_BUFFER_DELAY_MS", "3000"))

                    tts = CartesiaTTS(
                        api_key=cartesia_api_key or "",
                        voice_id=cartesia_voice_id,
                        model_id=cartesia_model,
                        language=cartesia_language,
                        cartesia_version=cartesia_version,
                        sample_rate=cartesia_sr,
                        max_buffer_delay_ms=cartesia_max_buffer,
                    )
            elif tts_engine == "vibevoice":
                vibevoice_url = os.getenv("VAGENT_VIBEVOICE_URL", "ws://localhost:3000")
                vibevoice_voice = os.getenv("VAGENT_VIBEVOICE_VOICE", "en-WHTest_man")
                vibevoice_cfg = float(os.getenv("VAGENT_VIBEVOICE_CFG", "1.5"))
                vibevoice_steps = int(os.getenv("VAGENT_VIBEVOICE_STEPS", "5"))
                
                tts = VibeVoiceTTS(
                    base_url=vibevoice_url,
                    voice=vibevoice_voice,
                    cfg_scale=vibevoice_cfg,
                    inference_steps=vibevoice_steps,
                )
            else:
                kokoro_url = os.getenv("VAGENT_KOKORO_URL", "http://localhost:8880/v1")
                kokoro_voice = os.getenv("VAGENT_KOKORO_VOICE", "af_heart")
                
                tts = KokoroTTS(
                    base_url=kokoro_url,
                    voice=kokoro_voice,
                )

        session = AgentSession(
            vad=vad,
            stt=stt,
            llm=ollama_llm,
            tts=tts,
            # Aggressive endpointing for speed
            min_endpointing_delay=float(os.getenv("VAGENT_MIN_ENDPOINTING_DELAY", "0.1")),
        )
    except Exception as e:
        logger.error(f"Failed to create AgentSession: {e}")
        return

    try:
        await session.start(
            agent=VoiceAssistant(),
            room=ctx.room,
            room_input_options=RoomInputOptions(close_on_disconnect=False),
        )
    except Exception as e:
        logger.error(f"Failed to start session: {e}")


def main():
    try:
        cli.run_app(server)
    except Exception as e:
        logger.critical(f"Application failed to run: {e}")


if __name__ == "__main__":
    main()
