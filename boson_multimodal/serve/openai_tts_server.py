"""
OpenAI-compatible TTS server built on Higgs Audio.

Exposes the following endpoints:
- GET  /v1/audio/voices              -> list available voice prompts
- GET  /v1/audio/voices/{voice_id}/sample -> stream the reference audio
- POST /v1/audio/speech              -> synthesize speech using Higgs Audio
- GET  /health                       -> health check

The server scans the ``voices`` directory for .wav + .txt pairs
and uses them as voice prompts for voice cloning.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ..data_types import AudioContent, ChatMLSample, Message
from .text_preprocessor import preprocess_text

# Audio prompts accepted for voice conditioning
SUPPORTED_PROMPT_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Audio formats supported for output
RESPONSE_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}

MODEL_ID = "higgs-audio"
DEFAULT_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
MODEL_IDLE_TIMEOUT_SECONDS = 300
MODEL_IDLE_CHECK_INTERVAL_SECONDS = 30

# Default generation parameters
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_MAX_NEW_TOKENS = 2048

logger = logging.getLogger(__name__)


@dataclass
class VoiceDescriptor:
    """Describes a voice with its audio reference and transcript."""

    voice_id: str
    audio_path: Path
    transcript: str

    @property
    def extension(self) -> str:
        return self.audio_path.suffix.lower().lstrip(".")

    @property
    def display_name(self) -> str:
        return self.voice_id.replace("_", " ").title()


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech request model."""

    model: str = Field(
        default=MODEL_ID,
        description="Model identifier. Use 'higgs-audio'.",
    )
    input: str = Field(..., description="Text that should be synthesized.")
    voice: str = Field(..., description="Voice identifier returned by /v1/audio/voices.")
    response_format: str = Field(
        default="wav",
        description="Requested audio format. Supported: wav, mp3.",
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed multiplier. Only 1.0 is currently supported.",
    )
    temperature: Optional[float] = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation.",
    )
    top_p: Optional[float] = Field(
        default=DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter.",
    )
    top_k: Optional[int] = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        description="Top-k sampling parameter.",
    )
    max_new_tokens: Optional[int] = Field(
        default=DEFAULT_MAX_NEW_TOKENS,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate.",
    )


def _detect_device() -> str:
    """Detect the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _voices_directory() -> Path:
    """Get the voices directory from environment or default."""
    env_override = os.getenv("HIGGS_VOICES_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    # Default: ./voices relative to project root
    return Path.cwd() / "voices"


def _discover_voices(directory: Path) -> Dict[str, VoiceDescriptor]:
    """
    Discover voice prompts in the given directory.

    Looks for .wav/.mp3/etc files with corresponding .txt transcript files.
    """
    if not directory.exists():
        logger.warning(f"Voices directory not found: {directory}")
        return {}

    voices: Dict[str, VoiceDescriptor] = {}

    for audio_path in sorted(directory.iterdir()):
        if audio_path.suffix.lower() not in SUPPORTED_PROMPT_EXTENSIONS:
            continue

        voice_id = audio_path.stem
        txt_path = audio_path.with_suffix(".txt")

        if not txt_path.exists():
            logger.warning(f"No transcript file for voice '{voice_id}': expected {txt_path}")
            continue

        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Failed to read transcript for voice '{voice_id}': {e}")
            continue

        if not transcript:
            logger.warning(f"Empty transcript for voice '{voice_id}'")
            continue

        if voice_id in voices:
            logger.warning(f"Duplicate voice identifier: {voice_id}")
            continue

        voices[voice_id] = VoiceDescriptor(
            voice_id=voice_id,
            audio_path=audio_path,
            transcript=transcript,
        )

    if voices:
        logger.info(f"Discovered {len(voices)} voice(s) in {directory}")
    else:
        logger.warning(f"No valid voices found in {directory}")

    return voices


def _encode_audio_to_base64(audio_path: Path) -> str:
    """Encode audio file to base64 string."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_voice_clone_sample(
    reference_transcript: str,
    reference_audio_b64: str,
    text_to_synthesize: str,
) -> ChatMLSample:
    """
    Build a ChatMLSample for voice cloning.

    Creates a conversation with:
    - User message: reference transcript
    - Assistant message: reference audio
    - User message: text to synthesize
    """
    messages = [
        Message(
            role="user",
            content=reference_transcript,
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio_b64, audio_url="placeholder"),
        ),
        Message(
            role="user",
            content=text_to_synthesize,
        ),
    ]
    return ChatMLSample(messages=messages)


def _render_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    response_format: str,
) -> bytes:
    """
    Serialize numpy waveform to the desired audio format.

    Uses pydub for format conversion.
    """
    import io
    from pydub import AudioSegment

    # Ensure waveform is 1D
    if waveform.ndim > 1:
        waveform = waveform.flatten()

    # Convert float32 [-1, 1] to int16 PCM
    waveform_int16 = (waveform * 32767).astype(np.int16)

    # Create AudioSegment from raw PCM data
    audio_segment = AudioSegment(
        waveform_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit = 2 bytes
        channels=1,
    )

    # Export to requested format
    buffer = io.BytesIO()
    audio_segment.export(buffer, format=response_format)
    buffer.seek(0)

    return buffer.read()


def _load_tts_engine():
    """Load the HiggsAudioServeEngine."""
    from .serve_engine import HiggsAudioServeEngine

    device = _detect_device()
    model_path = os.getenv("HIGGS_MODEL_PATH", DEFAULT_MODEL_PATH)

    # Audio tokenizer is a separate model from the main LLM
    # Check for local subdirectory first, then env var, then HuggingFace
    local_tokenizer_path = os.path.join(model_path, "higgs-audio-v2-tokenizer")
    if os.path.exists(local_tokenizer_path):
        audio_tokenizer_path = local_tokenizer_path
    else:
        audio_tokenizer_path = os.getenv("HIGGS_AUDIO_TOKENIZER_PATH", "bosonai/higgs-audio-v2-tokenizer")

    logger.info(f"Loading Higgs Audio model from {model_path} on device: {device}")
    logger.info(f"Loading audio tokenizer from {audio_tokenizer_path}")
    print(f"[Higgs Audio] Loading model from {model_path} on device: {device}")

    engine = HiggsAudioServeEngine(
        model_name_or_path=model_path,
        audio_tokenizer_name_or_path=audio_tokenizer_path,
        device=device,
    )

    return engine


async def _ensure_tts_engine():
    """Ensure the TTS engine is loaded, loading it if necessary."""
    async with app.state.model_lock:
        engine = app.state.tts_engine
        if engine is None:
            engine = await asyncio.to_thread(_load_tts_engine)
            app.state.tts_engine = engine
            app.state.last_usage = time.monotonic()
        return engine


async def _idle_model_reaper():
    """Background task to unload idle model."""
    timeout = int(os.getenv("HIGGS_IDLE_TIMEOUT", str(MODEL_IDLE_TIMEOUT_SECONDS)))
    logger.info(f"Idle model reaper started with timeout={timeout}s, check_interval={MODEL_IDLE_CHECK_INTERVAL_SECONDS}s")

    while True:
        try:
            await asyncio.sleep(MODEL_IDLE_CHECK_INTERVAL_SECONDS)

            engine_to_release = None
            async with app.state.model_lock:
                engine = app.state.tts_engine
                last_used = app.state.last_usage

                if engine is None or last_used is None:
                    continue

                if app.state.inference_lock.locked():
                    logger.debug("Skipping unload check - inference in progress")
                    continue

                elapsed = time.monotonic() - last_used
                if elapsed < timeout:
                    continue

                logger.info(
                    f"Unloading Higgs Audio model after {elapsed:.0f} seconds of inactivity"
                )
                print(f"[Higgs Audio] Unloading model after {elapsed:.0f}s idle")
                engine_to_release = engine
                app.state.tts_engine = None
                app.state.last_usage = None

            if engine_to_release is not None:
                # Explicitly delete model components to release GPU memory
                if hasattr(engine_to_release, 'model'):
                    del engine_to_release.model
                if hasattr(engine_to_release, 'audio_tokenizer'):
                    del engine_to_release.audio_tokenizer
                del engine_to_release
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Model unloaded and GPU memory cleared")

        except asyncio.CancelledError:
            logger.info("Idle model reaper cancelled")
            break
        except Exception as e:
            logger.error(f"Error in idle model reaper: {e}")
            # Continue running despite errors


app = FastAPI(
    title="Higgs Audio OpenAI-compatible TTS API",
    version="1.0.0",
    description="OpenAI-compatible TTS server using Higgs Audio for voice cloning.",
)


@app.on_event("startup")
async def _startup_event():
    """Initialize app state on startup."""
    voices_dir = _voices_directory()
    app.state.voices = _discover_voices(voices_dir)
    app.state.model_lock = asyncio.Lock()
    app.state.inference_lock = asyncio.Lock()
    app.state.tts_engine = None
    app.state.last_usage = None
    app.state.unload_task = asyncio.create_task(_idle_model_reaper())


@app.on_event("shutdown")
async def _shutdown_event():
    """Cleanup on shutdown."""
    task = getattr(app.state, "unload_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _voice_listing(base_url: str, voices: Iterable[VoiceDescriptor]):
    """Build voice listing response."""
    data = []
    for voice in voices:
        sample_url = f"{base_url}/v1/audio/voices/{voice.voice_id}/sample"
        data.append({
            "object": "voice",
            "voice_id": voice.voice_id,
            "name": voice.display_name,
            "prompt_format": voice.extension,
            "sample_url": sample_url,
        })
    return {
        "object": "list",
        "data": data,
    }


@app.get("/v1/audio/voices")
async def list_voices(request: Request):
    """List all available voices."""
    base_url = str(request.base_url).rstrip("/")
    return _voice_listing(base_url, app.state.voices.values())


@app.get("/v1/audio/voices/{voice_id}")
async def get_voice(voice_id: str, request: Request):
    """Get details about a specific voice."""
    voices: Dict[str, VoiceDescriptor] = app.state.voices
    voice = voices.get(voice_id)
    if voice is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{voice_id}' not found.",
        )
    base_url = str(request.base_url).rstrip("/")
    payload = _voice_listing(base_url, [voice])["data"][0]
    return payload


@app.get("/v1/audio/voices/{voice_id}/sample")
async def get_voice_sample(voice_id: str):
    """Stream the reference audio for a voice."""
    voices: Dict[str, VoiceDescriptor] = app.state.voices
    voice = voices.get(voice_id)
    if voice is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{voice_id}' not found.",
        )
    return FileResponse(voice.audio_path, filename=voice.audio_path.name)


async def _synthesize(
    text: str,
    voice: VoiceDescriptor,
    engine,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> tuple[np.ndarray, int]:
    """
    Synthesize speech using voice cloning.

    Returns:
        Tuple of (audio waveform as numpy array, sample rate)
    """

    def _generate() -> tuple[np.ndarray, int]:
        # Preprocess the input text
        processed_text = preprocess_text(text)

        # Encode reference audio to base64
        reference_b64 = _encode_audio_to_base64(voice.audio_path)

        # Build the voice cloning sample
        sample = _build_voice_clone_sample(
            reference_transcript=voice.transcript,
            reference_audio_b64=reference_b64,
            text_to_synthesize=processed_text,
        )

        # Generate audio
        response = engine.generate(
            sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            force_audio_gen=True,
        )

        if response.audio is None:
            raise RuntimeError("Model did not generate audio output")

        return response.audio, response.sampling_rate

    # Ensure sequential inference
    async with app.state.inference_lock:
        app.state.last_usage = time.monotonic()
        result = await asyncio.to_thread(_generate)
        app.state.last_usage = time.monotonic()
        return result


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Generate speech from text using voice cloning."""
    # Validate model
    if request.model != MODEL_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model '{request.model}'. Use '{MODEL_ID}'.",
        )

    # Validate response format
    response_format = request.response_format.lower()
    if response_format not in RESPONSE_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported response_format '{request.response_format}'. "
            f"Supported formats: {', '.join(RESPONSE_MIME_TYPES)}.",
        )

    # Validate speed
    if request.speed not in (None, 1.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Playback speed values other than 1.0 are not supported.",
        )

    # Validate voice
    voices: Dict[str, VoiceDescriptor] = app.state.voices
    voice = voices.get(request.voice)
    if voice is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{request.voice}' not found.",
        )

    # Validate input text
    if not request.input or not request.input.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty.",
        )

    # Ensure engine is loaded
    engine = await _ensure_tts_engine()

    # Generate audio
    try:
        waveform, sample_rate = await _synthesize(
            request.input,
            voice,
            engine,
            temperature=request.temperature or DEFAULT_TEMPERATURE,
            top_p=request.top_p or DEFAULT_TOP_P,
            top_k=request.top_k or DEFAULT_TOP_K,
            max_new_tokens=request.max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
        )
    except Exception as err:
        logger.error(f"Speech synthesis failed: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {str(err)}",
        ) from err

    # Render to requested format
    try:
        payload = _render_waveform(waveform, sample_rate, response_format)
    except RuntimeError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(err),
        ) from err

    mime_type = RESPONSE_MIME_TYPES[response_format]
    filename = f"speech.{response_format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=payload, media_type=mime_type, headers=headers)


@app.get("/health")
async def healthcheck():
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_ID}


def main():
    """Run the server."""
    import uvicorn

    port = int(os.getenv("HIGGS_PORT", "8005"))

    uvicorn.run(
        "boson_multimodal.serve.openai_tts_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
