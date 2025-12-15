"""
Voice Agent using LiveKit Agents SDK v1.3
STT: faster-whisper (large-v3-turbo)
LLM: Ollama gemma3:4b
TTS: Kokoro-FastAPI
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
)
from livekit.plugins import silero, openai as lk_openai

from vagent.plugins import FasterWhisperSTT, KokoroTTS

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
    proc.userdata["vad"] = silero.VAD.load()

    # Latency/quality knobs (defaults preserve current behavior)
    beam_size = int(os.getenv("VAGENT_STT_BEAM_SIZE", "5"))
    vad_filter = os.getenv("VAGENT_STT_VAD_FILTER", "1").strip().lower() in {"1", "true", "yes", "on"}
    without_timestamps = os.getenv("VAGENT_STT_WITHOUT_TIMESTAMPS", "1").strip().lower() in {"1", "true", "yes", "on"}

    proc.userdata["stt"] = FasterWhisperSTT(
        model_path=str(MODELS_DIR),
        device="cuda",
        compute_type="float16",
        beam_size=beam_size,
        vad_filter=vad_filter,
        without_timestamps=without_timestamps,
    )
    proc.userdata["tts"] = KokoroTTS(
        base_url="http://localhost:8880/v1",
        voice="af_heart",
    )


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

    # Ollama's OpenAI-compatible streaming can legitimately pause (model load, long decode).
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

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=ctx.proc.userdata["stt"],
        llm=ollama_llm,
        tts=ctx.proc.userdata["tts"],
        min_endpointing_delay=float(os.getenv("VAGENT_MIN_ENDPOINTING_DELAY", "0.5")),
    )

    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(close_on_disconnect=False),
    )


def main():
    cli.run_app(server)


if __name__ == "__main__":
    main()
