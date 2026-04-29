"""
Chatterbox Multilingual TTS API Server
Runs on GPU pods (RunPod/Vast.ai), serves voice generation endpoints.

Endpoints:
  GET  /health                          — Server health + model status
  POST /api/v1/voice/chatterbox_basic   — Text → speech (no cloning)
  POST /api/v1/voice/chatterbox_clone   — Text + reference audio → cloned speech
  GET  /api/v1/download/{job_id}        — Download generated audio
"""

import gc
import os
import sys
import uuid
import subprocess
from pathlib import Path
from typing import Optional

# ── Monkey-patch perth watermarker BEFORE any chatterbox import ──────────
# The native perth library crashes on most RunPod pods (missing libnvrtc).
import types
_fake_perth = types.ModuleType("perth")
class _DummyWatermarker:
    def embed(self, a, s): return a
    def detect(self, a, s): return False
    def apply_watermark(self, wav, sample_rate=None): return wav
_fake_perth.PerthImplicitWatermarker = _DummyWatermarker
_fake_perth.DummyWatermarker = _DummyWatermarker
_fake_perth.WatermarkerBase = _DummyWatermarker
sys.modules["perth"] = _fake_perth

import torch
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import FileResponse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/tmp/audio-outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_API_KEY = os.getenv("AUDIO_API_KEY", "").strip()

# ── Global model ─────────────────────────────────────────────────────────────
MODEL = None
MODEL_SR = 24000  # Chatterbox output sample rate

app = FastAPI(title="Chatterbox Multilingual TTS API")


def verify_api_key(x_api_key: Optional[str]):
    if AUDIO_API_KEY and x_api_key != AUDIO_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_model():
    global MODEL, MODEL_SR
    if MODEL is not None:
        return MODEL
    logger.info("Loading Chatterbox Multilingual model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    MODEL_SR = MODEL.sr
    logger.info(f"Model loaded! SR={MODEL_SR}, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return MODEL


def wav_to_mp3(wav_path: Path, mp3_path: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "192k", str(mp3_path)],
        capture_output=True, timeout=30,
    )


def ensure_wav(audio_bytes: bytes) -> str:
    """Convert any audio format to WAV 24kHz mono for Chatterbox reference."""
    import tempfile
    in_path = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    in_path.write(audio_bytes)
    in_path.close()
    out_path = in_path.name + ".wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", in_path.name, "-ar", "24000", "-ac", "1", "-f", "wav", out_path],
        capture_output=True, timeout=30,
    )
    os.unlink(in_path.name)
    return out_path


# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Starting Chatterbox TTS API server...")
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    gpu_name = None
    gpu_mem = None
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
    except Exception:
        pass

    return {
        "status": "healthy" if MODEL is not None else "loading",
        "model": "chatterbox-multilingual",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "gpu_memory_used": gpu_mem,
        "supported_languages": ["ar","da","de","el","en","es","fi","fr","he","hi","it","ja","ko","ms","nl","no","pl","pt","ru","sv","sw","tr","zh"],
    }


# ── Voice Basic ──────────────────────────────────────────────────────────────
@app.post("/api/v1/voice/chatterbox_basic")
async def voice_basic(
    text: str = Form(...),
    language_id: str = Form("hi"),
    cfg_weight: float = Form(0.0),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    top_p: float = Form(0.9),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    job_id = str(uuid.uuid4())
    wav_path = OUTPUT_DIR / f"{job_id}.wav"
    mp3_path = OUTPUT_DIR / f"{job_id}.mp3"

    try:
        model = load_model()
        logger.info(f"chatterbox_basic: len={len(clean_text)} lang={language_id} cfg={cfg_weight} exag={exaggeration}")

        wav = model.generate(
            clean_text,
            language_id=language_id,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature,
            top_p=top_p,
        )

        import torchaudio
        torchaudio.save(str(wav_path), wav.cpu(), MODEL_SR)
        wav_to_mp3(wav_path, mp3_path)

        return {
            "job_id": job_id,
            "engine": "chatterbox-multilingual",
            "download_wav": f"/api/v1/download/{job_id}?format=wav",
            "download_mp3": f"/api/v1/download/{job_id}?format=mp3",
        }
    except Exception as e:
        logger.exception("chatterbox_basic failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Voice Clone ──────────────────────────────────────────────────────────────
@app.post("/api/v1/voice/chatterbox_clone")
async def voice_clone(
    text: str = Form(...),
    speaker_wav: UploadFile = File(...),
    language_id: str = Form("hi"),
    reference_text: str = Form(""),
    cfg_weight: float = Form(0.0),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    top_p: float = Form(0.9),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    job_id = str(uuid.uuid4())
    wav_path = OUTPUT_DIR / f"{job_id}.wav"
    mp3_path = OUTPUT_DIR / f"{job_id}.mp3"
    ref_path = None

    try:
        model = load_model()

        # Save and convert reference audio to WAV
        ref_bytes = await speaker_wav.read()
        ref_path = ensure_wav(ref_bytes)

        logger.info(
            f"chatterbox_clone: len={len(clean_text)} lang={language_id} "
            f"ref_audio={len(ref_bytes)} bytes cfg={cfg_weight} exag={exaggeration}"
        )

        wav = model.generate(
            clean_text,
            language_id=language_id,
            audio_prompt_path=ref_path,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature,
            top_p=top_p,
        )

        import torchaudio
        torchaudio.save(str(wav_path), wav.cpu(), MODEL_SR)
        wav_to_mp3(wav_path, mp3_path)

        return {
            "job_id": job_id,
            "engine": "chatterbox-multilingual-clone",
            "download_wav": f"/api/v1/download/{job_id}?format=wav",
            "download_mp3": f"/api/v1/download/{job_id}?format=mp3",
        }
    except Exception as e:
        logger.exception("chatterbox_clone failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


# ── Music/SFX — Not Supported ───────────────────────────────────────────────
@app.post("/api/v1/music")
async def music_not_supported():
    raise HTTPException(
        status_code=400,
        detail="Chatterbox is voice-only. Music generation is not supported. "
               "Use an XTTS or Fish Audio pod for music/SFX generation.",
    )

@app.post("/api/v1/sfx")
async def sfx_not_supported():
    raise HTTPException(
        status_code=400,
        detail="Chatterbox is voice-only. SFX generation is not supported. "
               "Use an XTTS or Fish Audio pod for music/SFX generation.",
    )


# ── Download ─────────────────────────────────────────────────────────────────
@app.get("/api/v1/download/{job_id}")
async def download(job_id: str, format: str = "mp3"):
    if format not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="Invalid format")
    path = OUTPUT_DIR / f"{job_id}.{format}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path,
        media_type="audio/mpeg" if format == "mp3" else "audio/wav",
        filename=f"{job_id}.{format}",
    )


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
