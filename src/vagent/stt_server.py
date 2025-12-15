
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("stt-server")

# Global model instance
model: WhisperModel | None = None

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global model
    try:
        logger.info(f"Loading Whisper model from {MODELS_DIR}")
        model = WhisperModel(
            str(MODELS_DIR),
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise
    yield
    # Cleanup if needed
    model = None


app = FastAPI(lifespan=lifespan)


def resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 16000:
        return audio.astype(np.float32, copy=False)

    from scipy import signal
    import math

    g = math.gcd(src_sr, 16000)
    up = 16000 // g
    down = src_sr // g
    return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)


@app.post("/transcribe")
async def transcribe(
    file: Annotated[UploadFile, File()],
    language: Annotated[str | None, Form()] = None,
    beam_size: Annotated[int, Form()] = 1,
    vad_filter: Annotated[bool, Form()] = True,
    without_timestamps: Annotated[bool, Form()] = True,
    sample_rate: Annotated[int, Form()] = 48000,
    channels: Annotated[int, Form()] = 1,
):
    if model is None:
        return {"error": "Model not loaded", "text": ""}

    local_model = model

    try:
        # Read audio bytes
        content = await file.read()
        
        # Convert bytes to numpy array (assuming 16-bit signed int from LiveKit)
        audio_data = np.frombuffer(content, dtype=np.int16).astype(np.float32)

        # Convert to mono if stereo
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels).mean(axis=1)

        # Normalize to [-1, 1]
        audio_data = audio_data / 32768.0

        # Resample to 16kHz
        if sample_rate != 16000:
            audio_data = resample_to_16k(audio_data, sample_rate)

        # Run transcription
        # We run this in a thread pool automatically by FastAPI since it's not an async def 
        # (Wait, WhisperModel.transcribe is blocking, so we should run it in executor if we want async concurrency,
        # but FastAPI handles standard defs in threadpool. However, we made this route async def to await file.read().
        # So we must wrap the blocking call.)
        
        loop = asyncio.get_running_loop()
        
        segments, info = await loop.run_in_executor(
            None,
            lambda: local_model.transcribe(
                audio_data,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                without_timestamps=without_timestamps,
            )
        )

        text_parts = [segment.text.strip() for segment in segments]
        full_text = " ".join(text_parts)
        
        return {
            "text": full_text,
            "language": info.language,
            "language_probability": info.language_probability
        }

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"error": str(e), "text": ""}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
