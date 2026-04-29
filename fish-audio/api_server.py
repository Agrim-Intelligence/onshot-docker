from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
import os
import gc
import sys
import time
import asyncio
import logging
import subprocess
from datetime import datetime, timezone

import torch
import numpy as np
import soundfile as sf
import httpx

try:
    from audiocraft.models import MusicGen, AudioGen
    from audiocraft.data.audio import audio_write
    _HAS_AUDIOCRAFT = True
except Exception as _audiocraft_err:
    # xformers or other audiocraft dep may be missing; music/sfx endpoints will return 503
    print(f"[WARN] audiocraft import failed: {_audiocraft_err}")
    MusicGen = None    # type: ignore[assignment,misc]
    AudioGen = None    # type: ignore[assignment,misc]
    audio_write = None # type: ignore[assignment]
    _HAS_AUDIOCRAFT = False

# Provider-aware imports — only load coqui-tts/transformers in xtts mode
AUDIO_TTS_PROVIDER = os.getenv("AUDIO_TTS_PROVIDER", "xtts").strip().lower()

if AUDIO_TTS_PROVIDER != "fish":
    from transformers import pipeline as _hf_pipeline
    from TTS.api import TTS as _TTS
else:
    _hf_pipeline = None  # type: ignore[assignment]
    _TTS = None          # type: ignore[assignment]

try:
    import ormsgpack as _msgpack
    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

# faster-whisper for auto-transcribing reference audio (voice cloning)
try:
    from faster_whisper import WhisperModel as _WhisperModel
    _HAS_WHISPER = True
except ImportError:
    _WhisperModel = None  # type: ignore[assignment,misc]
    _HAS_WHISPER = False

# ---------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------
os.environ["COQUI_TOS_AGREED"] = "1"

# ---------------------------------------------------------------------
# LOGGING  — always on local container disk (not network volume)
# ---------------------------------------------------------------------
LOG_DIR = Path("/root/audio-logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "server.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("audio-server")

# ---------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------
app = FastAPI(title="Audio Generation API", version="1.3.0")

API_KEY = os.getenv("AUDIO_API_KEY", "sk-audio-2024-change-me")
OUTPUT_DIR = Path("/tmp/audio-outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fish-speech API server runs on port 8080 inside the pod (started by deploy.sh).
# The main API server proxies /api/v1/voice/fish_* requests to it.
FISH_SPEECH_BASE = os.getenv("FISH_SPEECH_BASE_URL", "http://127.0.0.1:8080")

logger.info(f"Starting audio server: AUDIO_TTS_PROVIDER={AUDIO_TTS_PROVIDER}")


# ---------------------------------------------------------------------
# MODEL MANAGER — fix ref: step_6_fix_plan.json cluster C13
# (added 2026-04-20 after BGM/SFX generation failures on Varanasi-Potter run)
# ---------------------------------------------------------------------
# Problem this solves:
#   The fish-audio pod hosts THREE heavy GPU workloads — fish-speech (TTS),
#   MusicGen (BGM), AudioGen (SFX). On a 20-24GB GPU they cannot coexist
#   in VRAM. Previously fish-speech was preloaded at boot, hogging ~6-8GB;
#   MusicGen requests would then attempt a swap but the logic was scattered
#   across multiple helpers with race-prone globals (_fish_stopped_for_vram).
#   First request after boot would often hang or OOM silently.
#
# This manager:
#   1. Tracks the active_kind (single model in VRAM at a time).
#   2. Serializes every load / unload / generate through a single asyncio.Lock.
#   3. Before loading a different kind: unload current + gc + cuda empty_cache
#      + (for fish-speech) kill the subprocess, THEN load target.
#   4. Reports VRAM before/after every swap into a rolling history.
#   5. Exposes live state via GET /health, GET /api/v1/status, and
#      POST /api/v1/prepare (pre-warm without generating).
#   6. Handles fish-speech (subprocess) as just another kind.
#   7. Idempotent: if the requested kind is already active, reuse.
#
# Kinds (order matters only for documentation):
#   "music"        : MusicGen medium/large
#   "sfx"          : AudioGen medium
#   "fish_speech"  : fish-speech S2-Pro subprocess on :8080
#   "xtts"         : XTTS-v2 (xtts provider only)
#   "mms"          : facebook/mms-tts-hin (xtts provider only)

_VALID_KINDS = {"music", "sfx", "fish_speech", "xtts", "mms"}


def _vram_stats_gb() -> Dict[str, float]:
    """Return current CUDA VRAM usage in GB. Safe to call when no CUDA."""
    try:
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
        props = torch.cuda.get_device_properties(0)
        return {
            "allocated": round(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024, 2),
            "reserved":  round(torch.cuda.memory_reserved(0)  / 1024 / 1024 / 1024, 2),
            "total":     round(props.total_memory             / 1024 / 1024 / 1024, 2),
        }
    except Exception:
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}


class ModelManager:
    """Single-active-kind model manager with explicit swap + VRAM audit.

    Thread-safe via asyncio.Lock. All generation endpoints MUST acquire
    the lock before dispatching — they use ``async with manager.lock``
    indirectly by calling ``manager.ensure(kind)`` inside their handler.
    """

    def __init__(self):
        # Current active kind (None = VRAM clean)
        self.active_kind: Optional[str] = None
        # The actual model object for audiocraft kinds. Fish-speech is None
        # (it's a subprocess managed via helpers, not a torch module).
        self._instance: Optional[Any] = None
        # For MusicGen: remember which HF variant is loaded so a size change
        # forces a reload.
        self._music_variant: str = ""
        # Swap history (last 20) for debugging / observability
        self._swap_history: list = []
        # Lock — single mutex around every swap + every generate.
        self.lock: asyncio.Lock = asyncio.Lock()
        # Fish-speech process handle when we started it ourselves
        self._fish_proc: Optional[subprocess.Popen] = None

    # ---- public API --------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Non-blocking snapshot. Safe to call from /health."""
        return {
            "active_kind": self.active_kind,
            "music_variant": self._music_variant if self.active_kind == "music" else None,
            "vram_gb": _vram_stats_gb(),
            "recent_swaps": self._swap_history[-10:],
        }

    async def ensure(self, kind: str, *, music_variant: str = "") -> Any:
        """Ensure ``kind`` is the active model. Swap if needed.

        Idempotent: returns the live instance if already active.
        Raises HTTPException(503) if model load fails (VRAM stays clean).
        """
        if kind not in _VALID_KINDS:
            raise ValueError(f"Unknown kind: {kind!r}. Valid: {sorted(_VALID_KINDS)}")

        async with self.lock:
            # Music variant change triggers reload even when kind matches
            wants_reload = (
                kind == "music"
                and self.active_kind == "music"
                and music_variant
                and music_variant != self._music_variant
            )
            if self.active_kind == kind and not wants_reload:
                logger.info(f"[VRAM] ensure({kind}): already active — reuse")
                return self._instance

            t0 = time.monotonic()
            before_kind = self.active_kind
            before_vram = _vram_stats_gb()

            # 1. Unload whatever is currently active.
            if self.active_kind is not None:
                self._unload_current()

            # 2. Load the target kind.
            try:
                self._instance = self._load(kind, music_variant=music_variant)
                self.active_kind = kind
                if kind == "music":
                    self._music_variant = music_variant or _default_music_model()
            except Exception as load_err:
                # Leave VRAM clean on failure — do not pin a broken ref.
                self._instance = None
                self.active_kind = None
                self._force_vram_clean()
                logger.exception(f"[VRAM] ensure({kind}) failed")
                raise HTTPException(status_code=503, detail=f"Failed to load {kind}: {load_err}") from load_err

            elapsed = round(time.monotonic() - t0, 2)
            after_vram = _vram_stats_gb()
            entry = {
                "at": datetime.now(timezone.utc).isoformat(),
                "from": before_kind,
                "to": kind,
                "elapsed_s": elapsed,
                "vram_before_gb": before_vram,
                "vram_after_gb": after_vram,
            }
            self._swap_history.append(entry)
            if len(self._swap_history) > 20:
                self._swap_history = self._swap_history[-20:]
            logger.info(
                f"[VRAM] swap {before_kind or 'empty'} -> {kind} in {elapsed}s "
                f"(allocated {before_vram['allocated']}GB -> {after_vram['allocated']}GB / "
                f"{after_vram['total']}GB total)"
            )
            return self._instance

    async def unload_all(self) -> Dict[str, Any]:
        """Explicitly release all VRAM. Used before shutdown or by /api/v1/unload."""
        async with self.lock:
            before = self.active_kind
            before_vram = _vram_stats_gb()
            if self.active_kind is not None:
                self._unload_current()
            self._force_vram_clean()
            after_vram = _vram_stats_gb()
            logger.info(
                f"[VRAM] unload_all: was={before} "
                f"freed {before_vram['allocated'] - after_vram['allocated']:.2f}GB"
            )
            return {"released_from": before, "vram_before_gb": before_vram, "vram_after_gb": after_vram}

    # ---- private helpers --------------------------------------------

    def _unload_current(self) -> None:
        """Unload whatever's currently in VRAM, no lock (caller holds it)."""
        kind = self.active_kind
        if kind is None:
            return
        logger.info(f"[VRAM] unload current: {kind}")
        if kind == "fish_speech":
            _stop_fish_speech_subprocess(self._fish_proc)
            self._fish_proc = None
        else:
            # Drop strong reference; GC + empty_cache will reclaim VRAM.
            self._instance = None
        self.active_kind = None
        self._music_variant = ""
        self._force_vram_clean()

    def _force_vram_clean(self) -> None:
        """GC + torch.cuda.empty_cache(). Idempotent."""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                # torch.cuda.ipc_collect() also releases IPC-shared memory
                torch.cuda.ipc_collect()
                # Give the driver a moment to actually release segments
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"[VRAM] empty_cache failed (non-fatal): {e}")

    def _load(self, kind: str, *, music_variant: str = "") -> Any:
        """Actually load the target kind into VRAM. Runs with lock held."""
        if kind == "music":
            if not _HAS_AUDIOCRAFT:
                raise RuntimeError("audiocraft not available (import failed).")
            model_id = music_variant or _default_music_model()
            logger.info(f"[VRAM] loading MusicGen: {model_id}")
            m = MusicGen.get_pretrained(model_id)
            m.set_generation_params(duration=30)
            return m
        elif kind == "sfx":
            if not _HAS_AUDIOCRAFT:
                raise RuntimeError("audiocraft not available (import failed).")
            logger.info("[VRAM] loading AudioGen medium")
            return AudioGen.get_pretrained("facebook/audiogen-medium")
        elif kind == "fish_speech":
            if AUDIO_TTS_PROVIDER != "fish":
                raise RuntimeError("fish_speech only available when AUDIO_TTS_PROVIDER=fish")
            logger.info("[VRAM] starting fish-speech subprocess (port 8080)")
            self._fish_proc = _start_fish_speech_subprocess()
            # Block until healthy
            if not _wait_fish_speech_healthy(timeout_s=180):
                raise RuntimeError("fish-speech subprocess did not become healthy within 180s")
            return None  # subprocess managed separately
        elif kind == "xtts":
            if AUDIO_TTS_PROVIDER == "fish":
                raise RuntimeError("XTTS not available in fish provider mode")
            logger.info("[VRAM] loading XTTS-v2")
            return _TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=torch.cuda.is_available(),
            )
        elif kind == "mms":
            if AUDIO_TTS_PROVIDER == "fish":
                raise RuntimeError("MMS not available in fish provider mode")
            logger.info("[VRAM] loading MMS Hindi TTS")
            return _hf_pipeline(
                "text-to-speech",
                model="facebook/mms-tts-hin",
                device=0 if torch.cuda.is_available() else -1,
            )
        else:
            raise RuntimeError(f"Unknown kind: {kind!r}")


# Single global manager instance
manager = ModelManager()


# Back-compat: some legacy endpoints / helpers still reference the old
# ``models`` dict. Keep it as an always-empty reflection (so any stale
# `models[kind] is None` checks still work) but new code should use the
# manager exclusively.
models: dict = {
    "music":       None,
    "sfx":         None,
    "fish_speech": None,
}
if AUDIO_TTS_PROVIDER != "fish":
    models["mms"]  = None
    models["xtts"] = None

# Optional startup behavior: by default, do NOT auto-load fish-speech.
# It costs 2-3 min cold start; first /voice/fish_* request triggers it.
# Set FISH_SPEECH_AUTOSTART=true to pre-load.
@app.on_event("startup")
async def _maybe_prewarm_fish_speech():
    autostart = os.getenv("FISH_SPEECH_AUTOSTART", "false").strip().lower() in ("true", "1", "yes")
    if not autostart or AUDIO_TTS_PROVIDER != "fish":
        logger.info(
            f"[startup] fish-speech prewarm skipped (autostart={autostart}, provider={AUDIO_TTS_PROVIDER}). "
            f"Will lazy-load on first /voice/fish_* request."
        )
        return
    # Check if fish-speech is already running (survives across server restarts)
    try:
        r = httpx.get(f"{FISH_SPEECH_BASE}/v1/health", timeout=2)
        if r.status_code == 200:
            logger.info("[startup] fish-speech already running externally — will adopt lazily")
            return
    except Exception:
        pass
    logger.info("[startup] FISH_SPEECH_AUTOSTART=true — pre-loading fish-speech now...")
    try:
        await manager.ensure("fish_speech")
    except Exception as e:
        logger.warning(f"[startup] fish-speech prewarm failed (non-fatal, will retry on first request): {e}")


@app.on_event("shutdown")
async def _release_vram_on_shutdown():
    try:
        await manager.unload_all()
    except Exception as e:
        logger.warning(f"[shutdown] unload_all failed: {e}")


# Track which MusicGen variant is currently loaded so we can reload when size changes.
# Kept for back-compat with code that reads it directly; authoritative source is manager._music_variant.
_loaded_music_model: str = ""

# Whisper model for auto-transcription of reference audio (lazy-loaded)
_whisper_model = None

def _get_whisper_model():
    """Lazy-load faster-whisper medium model. Uses ~1.5GB VRAM on GPU.

    'medium' is required (not 'base') because the base model confuses
    Hindi/Urdu scripts — it outputs Arabic script for Hindi audio.
    The medium model handles Indic languages correctly.
    """
    global _whisper_model
    if _whisper_model is None and _HAS_WHISPER:
        logger.info("Loading faster-whisper 'medium' model for auto-transcription...")
        _whisper_model = _WhisperModel("medium", device="cuda", compute_type="float16")
        logger.info("faster-whisper 'medium' model loaded.")
    return _whisper_model


def _auto_transcribe(audio_bytes: bytes, language_hint: str = "") -> str:
    """Transcribe audio bytes using faster-whisper. Returns transcript or empty string on failure.

    This is used to auto-fill reference_text for fish-speech voice cloning
    when the user doesn't provide a transcript. Having the transcript
    dramatically improves voice cloning quality (accent, cadence, pronunciation).

    Args:
        audio_bytes: Raw audio bytes (WAV/MP3/etc).
        language_hint: ISO 639-1 language code (e.g. 'hi', 'en', 'es').
            When provided, Whisper skips language detection and transcribes
            in the specified language, avoiding Hindi/Urdu script confusion.
    """
    if not _HAS_WHISPER:
        logger.warning("faster-whisper not installed — cannot auto-transcribe reference audio")
        return ""
    import tempfile as _tf
    tmp_path = None
    try:
        model = _get_whisper_model()
        if model is None:
            return ""
        # Write audio bytes to temp file (faster-whisper needs a file path)
        with _tf.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            tmp_path = f.name
        # Pass language hint to avoid script confusion (e.g. Hindi→Urdu)
        transcribe_kwargs = {"beam_size": 5}
        if language_hint:
            transcribe_kwargs["language"] = language_hint
            logger.info(f"_auto_transcribe: using language hint '{language_hint}'")
        segments, info = model.transcribe(tmp_path, **transcribe_kwargs)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"_auto_transcribe: language={info.language} prob={info.language_probability:.2f} "
                     f"transcript='{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
        return transcript
    except Exception as e:
        logger.warning(f"_auto_transcribe failed: {e}")
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

def _default_music_model() -> str:
    """Resolve MUSICGEN_MODEL env var to a full HuggingFace model ID."""
    val = os.getenv("MUSICGEN_MODEL", "medium").strip().lower()
    if val in ("large", "facebook/musicgen-large"):
        return "facebook/musicgen-large"
    return "facebook/musicgen-medium"  # default: medium

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _stop_fish_speech_subprocess(proc: Optional[subprocess.Popen] = None) -> bool:
    """Low-level helper — stop fish-speech on port 8080 + wait for VRAM release.

    Used by ModelManager._unload_current when fish_speech is the active kind.
    Returns True if something was running and we stopped it.
    """
    if AUDIO_TTS_PROVIDER != "fish":
        return False
    was_running = False
    # 1. Check via HTTP if it's responding
    try:
        resp = httpx.get(f"{FISH_SPEECH_BASE}/v1/health", timeout=2)
        was_running = resp.status_code == 200
    except Exception:
        pass
    # 2. Also kill by process handle if we own one (handles the case where
    # the HTTP check fails because it's mid-load but the process is alive).
    if proc is not None and proc.poll() is None:
        was_running = True
    # 3. If nothing running we own or can see, short-circuit.
    if not was_running:
        return False

    logger.info("[VRAM] stopping fish-speech subprocess on port 8080...")
    try:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Belt-and-braces: kill anything bound to :8080
        subprocess.run(
            ["bash", "-c", "fuser -k 8080/tcp 2>/dev/null; lsof -ti :8080 | xargs -r kill -9 2>/dev/null; true"],
            capture_output=True, timeout=10,
        )
        # Wait for process tables to clear + VRAM to actually release.
        import time as _time
        for _ in range(10):
            _time.sleep(0.5)
            try:
                resp = httpx.get(f"{FISH_SPEECH_BASE}/v1/health", timeout=1)
                if resp.status_code != 200:
                    break
            except Exception:
                break
        logger.info("[VRAM] fish-speech stopped")
        return True
    except Exception as e:
        logger.warning(f"[VRAM] failed to stop fish-speech: {e}")
        return False


def _start_fish_speech_subprocess() -> Optional[subprocess.Popen]:
    """Low-level helper — spawn fish-speech subprocess on :8080.

    Returns the Popen handle so the manager can track + kill it cleanly.
    Does NOT wait for health (caller uses _wait_fish_speech_healthy).
    """
    if AUDIO_TTS_PROVIDER != "fish":
        return None
    # Prefer the deploy-generated start script
    start_script = Path("/workspace/start_fish.sh")
    if start_script.exists():
        return subprocess.Popen(
            ["bash", str(start_script)],
            stdout=open(LOG_DIR / "fish_speech.log", "a"),
            stderr=subprocess.STDOUT,
        )
    # Fallback: start directly with the same flags deploy.sh step 7.5 uses
    fish_dir = Path("/workspace/fish-speech")
    venv_python = "/workspace/fish-venv/bin/python"
    if not Path(venv_python).exists():
        venv_python = "/workspace/audio-venv/bin/python"  # xtts-shared venv fallback
    s2pro_path = Path("/workspace/models/fish-speech-s2pro")
    v15_path = Path("/workspace/models/fish-speech-1.5")
    is_s2pro = s2pro_path.exists()
    ckpt = str(s2pro_path) if is_s2pro else str(v15_path)
    if is_s2pro:
        decoder_path = f"{ckpt}/codec.pth"
        decoder_config = "modded_dac_vq"
    else:
        decoder_path = f"{ckpt}/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        decoder_config = "firefly_gan_vq"
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return subprocess.Popen(
        [venv_python, "tools/api_server.py",
         "--listen", "127.0.0.1:8080",
         "--llama-checkpoint-path", ckpt,
         "--decoder-checkpoint-path", decoder_path,
         "--decoder-config-name", decoder_config,
         "--half"],
        cwd=str(fish_dir),
        env=env,
        stdout=open(LOG_DIR / "fish_speech.log", "a"),
        stderr=subprocess.STDOUT,
    )


def _wait_fish_speech_healthy(timeout_s: int = 180) -> bool:
    """Poll fish-speech /v1/health until ready or timeout."""
    import time as _time
    deadline = _time.monotonic() + timeout_s
    attempts = 0
    while _time.monotonic() < deadline:
        attempts += 1
        try:
            r = httpx.get(f"{FISH_SPEECH_BASE}/v1/health", timeout=2)
            if r.status_code == 200:
                logger.info(f"[VRAM] fish-speech healthy after {attempts} attempts")
                return True
        except Exception:
            pass
        _time.sleep(1)
    logger.warning(f"[VRAM] fish-speech did not become healthy within {timeout_s}s")
    return False


# Back-compat wrappers — preserve old names used elsewhere in the file but
# re-route them through the manager. Any remaining direct callers (e.g.
# _proxy_fish_speech) continue to work without changes.
def _stop_fish_speech() -> bool:
    # Synchronous caller path; use the subprocess stopper directly.
    # (ModelManager.unload_all is async; for legacy code paths that are not
    # async we fall through to the low-level helper.)
    return _stop_fish_speech_subprocess(manager._fish_proc if manager.active_kind == "fish_speech" else None)


def _start_fish_speech() -> None:
    """Legacy sync entry — kept for the _proxy_fish_speech fallback.

    New code paths should call ``await manager.ensure("fish_speech")`` instead.
    """
    if AUDIO_TTS_PROVIDER != "fish":
        return
    # Best-effort sync path: spawn + wait inline.
    proc = _start_fish_speech_subprocess()
    _wait_fish_speech_healthy(timeout_s=180)
    # Remember the handle so unload can find it
    manager._fish_proc = proc
    manager.active_kind = "fish_speech"


def wav_to_mp3(wav: Path, mp3: Path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(wav),
            "-codec:a", "libmp3lame",
            "-b:a", "192k",
            str(mp3),
        ],
        check=True,
    )

# ---------------------------------------------------------------------
# BASIC HINDI TTS (MMS) — xtts mode only
# ---------------------------------------------------------------------
@app.post("/api/v1/voice/basic")
async def voice_basic(
    text: str = Form(...),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)

    if AUDIO_TTS_PROVIDER == "fish":
        raise HTTPException(
            status_code=400,
            detail="voice/basic (MMS Hindi) is not available in fish provider mode. "
                   "Use voice/fish_basic instead.",
        )

    # Guard: MMS-TTS needs non-empty Hindi text (at least a few chars)
    clean_text = text.strip()
    if not clean_text or len(clean_text) < 2:
        raise HTTPException(
            status_code=400,
            detail="Text is too short for TTS. Please provide at least a short Hindi sentence.",
        )

    job_id = str(uuid.uuid4())
    wav = OUTPUT_DIR / f"{job_id}.wav"
    mp3 = OUTPUT_DIR / f"{job_id}.mp3"

    try:
        tts = await manager.ensure("mms")
        result = tts(clean_text)

        audio = result["audio"]
        sr = int(result["sampling_rate"])

        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        elif audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.T

        # Guard: model produced no audio (can happen with unsupported characters)
        if audio.size == 0 or audio.shape[0] < 100:
            raise HTTPException(
                status_code=422,
                detail="MMS model produced no audio. The text may contain characters "
                       "unsupported by the Hindi MMS model. Try using Devanagari script.",
            )

        sf.write(wav, audio, sr, format="WAV", subtype="PCM_16")
        wav_to_mp3(wav, mp3)

        return {
            "job_id": job_id,
            "engine": "mms-hindi",
            "download_wav": f"/api/v1/download/{job_id}?format=wav",
            "download_mp3": f"/api/v1/download/{job_id}?format=mp3",
        }

    except HTTPException:
        raise  # re-raise our own validation errors
    except Exception as e:
        logger.exception("MMS TTS failed")
        # Check for the common "input size 0" error from MMS tokenizer
        err_str = str(e)
        if "negative output size" in err_str or "input size 0" in err_str:
            raise HTTPException(
                status_code=422,
                detail="MMS model could not process this text — the tokenizer produced "
                       "empty input. This usually happens with English text or unsupported "
                       "characters. Use Hindi (Devanagari) text for voice/basic, or use "
                       "voice/clone for other languages.",
            )
        raise HTTPException(status_code=500, detail=err_str)

# ---------------------------------------------------------------------
# VOICE CLONING (XTTS) — xtts mode only
# ---------------------------------------------------------------------
@app.post("/api/v1/voice/clone")
async def voice_clone(
    text: str = Form(...),
    language: str = Form("hi"),
    speaker_wav: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)

    if AUDIO_TTS_PROVIDER == "fish":
        raise HTTPException(
            status_code=400,
            detail="voice/clone (XTTS) is not available in fish provider mode. "
                   "Use voice/fish_clone instead.",
        )

    job_id = str(uuid.uuid4())
    wav = OUTPUT_DIR / f"{job_id}.wav"
    mp3 = OUTPUT_DIR / f"{job_id}.mp3"
    ref = OUTPUT_DIR / f"{job_id}_ref.wav"

    try:
        raw_bytes = await speaker_wav.read()
        wav_bytes = _ensure_wav(raw_bytes)
        with open(ref, "wb") as f:
            f.write(wav_bytes)

        tts = await manager.ensure("xtts")
        tts.tts_to_file(
            text=text,
            speaker_wav=str(ref),
            language=language,
            file_path=str(wav),
        )

        wav_to_mp3(wav, mp3)

        return {
            "job_id": job_id,
            "engine": "xtts-clone",
            "download_wav": f"/api/v1/download/{job_id}?format=wav",
            "download_mp3": f"/api/v1/download/{job_id}?format=mp3",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("XTTS clone failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if ref.exists():
            ref.unlink()

# ---------------------------------------------------------------------
# FISH-SPEECH TTS (self-hosted, proxied to port 8080)
# ---------------------------------------------------------------------

def _ensure_wav(audio_bytes: bytes) -> bytes:
    """Convert ALL audio to 44.1kHz mono WAV for fish-speech S2-Pro.

    Always convert — even MP3 and existing WAV files — to ensure consistent
    44.1kHz/mono/16-bit PCM input. Fish-speech S2-Pro's DAC codec extracts
    speaker embeddings from raw PCM; MP3 lossy compression and low sample
    rates degrade voice cloning quality significantly (accent mismatch,
    default voice fallback).
    """
    if not audio_bytes:
        return audio_bytes

    # Always convert to ensure 44.1kHz mono WAV
    import tempfile as _tf
    in_path = out_path = None
    try:
        with _tf.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(audio_bytes)
            in_path = f.name
        out_path = in_path + ".wav"
        # Use 44100Hz — Fish-speech S2-Pro extracts voice characteristics better
        # from high-quality audio. 16kHz destroys vocal detail and causes accent
        # fallback (e.g. South Indian default for Hindi).
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ar", "44100", "-ac", "1", "-f", "wav", out_path],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as wf:
                wav_bytes = wf.read()
            logger.info(f"_ensure_wav: converted {len(audio_bytes)} bytes -> {len(wav_bytes)} bytes WAV")
            return wav_bytes
        else:
            logger.warning(f"_ensure_wav: ffmpeg failed (rc={result.returncode}), using original bytes. "
                           f"stderr={result.stderr[:200]}")
            return audio_bytes
    except Exception as e:
        logger.warning(f"_ensure_wav: conversion failed ({e}), using original bytes")
        return audio_bytes
    finally:
        for p in (in_path, out_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


async def _register_fish_reference(
    ref_wav_bytes: bytes,
    ref_text: str,
) -> str | None:
    """Pre-register a reference voice with fish-speech via /v1/references/add.

    Returns the reference_id on success, None on failure.
    Pre-registered references produce significantly better cloning quality
    than inline references because the voice embedding is computed once at
    high quality and cached server-side.
    """
    import hashlib
    ref_id = "ref-" + hashlib.sha256(ref_wav_bytes).hexdigest()[:16]

    try:
        # Check if already registered
        async with httpx.AsyncClient(timeout=10.0) as client:
            list_resp = await client.get(f"{FISH_SPEECH_BASE}/v1/references/list")
            if list_resp.status_code == 200:
                existing = list_resp.json().get("reference_ids", [])
                if ref_id in existing:
                    logger.info(f"[fish-ref] Reference {ref_id} already registered")
                    return ref_id

        # Register new reference via msgpack (same format as /v1/tts)
        # The /v1/references/add endpoint expects: id (str), audio (bytes), text (str)
        if not _HAS_MSGPACK:
            logger.warning("[fish-ref] ormsgpack not available — cannot register")
            return None

        try:
            add_payload = _msgpack.packb({
                "id": ref_id,
                "audio": ref_wav_bytes,
                "text": ref_text or "reference audio",
            })
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{FISH_SPEECH_BASE}/v1/references/add",
                    content=add_payload,
                    headers={"Content-Type": "application/msgpack"},
                )
            if resp.status_code == 200:
                logger.info(f"[fish-ref] Registered reference {ref_id} ({len(ref_wav_bytes)} bytes, text={ref_text[:60]!r})")
                return ref_id
            elif resp.status_code == 409:
                # Already exists (race condition) — still usable
                logger.info(f"[fish-ref] Reference {ref_id} already exists (409)")
                return ref_id
            else:
                resp_text = resp.text[:200] if resp.headers.get("content-type", "").startswith("text") else f"(binary, {len(resp.content)} bytes)"
                logger.warning(f"[fish-ref] Failed to register: {resp.status_code} {resp_text}")
        except Exception as reg_err:
            logger.warning(f"[fish-ref] Registration request failed: {reg_err}")
    except Exception as e:
        logger.warning(f"[fish-ref] Registration failed: {e}")

    return None


async def _proxy_fish_speech(
    text: str,
    speaker_wav_bytes: bytes = b"",
    reference_text: str = "",
) -> dict:
    """Send a TTS or voice-clone request to the fish-speech API server on port 8080.

    fish-speech uses the same msgpack API format as fish.audio cloud.

    Voice cloning strategy (in order of preference):
    1. Pre-register the reference via /v1/references/add, then use reference_id
       in the TTS request. This gives the best quality because the voice embedding
       is computed once at high quality.
    2. Fallback: send inline reference audio bytes (lower quality, but works
       when registration fails).

    Args:
        text: The text to synthesize.
        speaker_wav_bytes: Raw bytes of the reference audio for voice cloning.
        reference_text: Transcript of the reference audio.
    """
    fmt = "wav"

    if not _HAS_MSGPACK:
        raise HTTPException(
            status_code=500,
            detail="ormsgpack is required for fish-speech integration. "
                   "Run: pip install ormsgpack>=1.6.0 in the audio venv.",
        )

    payload: dict = {
        "text": text,
        "format": fmt,
        "normalize": True,
        "latency": "normal",
        "chunk_length": 200,
    }
    if speaker_wav_bytes:
        ref_audio = _ensure_wav(speaker_wav_bytes)

        # Estimate duration for logging
        est_duration = max(0, (len(ref_audio) - 44)) / (44100 * 1 * 2)
        quality = "good" if est_duration >= 10 else "moderate" if est_duration >= 5 else "poor"

        # Try pre-registered reference first (better accent matching).
        # Falls back to inline if registration fails.
        ref_id = await _register_fish_reference(ref_audio, reference_text)
        if ref_id:
            payload["reference_id"] = ref_id
            logger.info(f"[fish-tts] Pre-registered ref: {ref_id}, ~{est_duration:.1f}s, quality={quality}")
        else:
            payload["references"] = [{"audio": ref_audio, "text": reference_text}]
            logger.info(f"[fish-tts] Inline ref: {len(ref_audio)} bytes, ~{est_duration:.1f}s, quality={quality}")

    body = _msgpack.packb(payload, option=_msgpack.OPT_NON_STR_KEYS)

    # Fix ref: C13 — always route through the manager. This handles the case
    # where music/sfx swapped fish-speech out (as well as the first-request-
    # after-boot lazy-load case). The manager serializes the swap so concurrent
    # callers don't race each other into OOM.
    try:
        await manager.ensure("fish_speech")
    except HTTPException as swap_err:
        raise HTTPException(
            status_code=503,
            detail=(
                f"fish-speech failed to load: {swap_err.detail}. "
                f"Check /root/audio-logs/fish_speech.log for details."
            ),
        )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{FISH_SPEECH_BASE}/v1/tts",
                content=body,
                headers={"Content-Type": "application/msgpack"},
            )
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"fish-speech server returned HTTP {resp.status_code}: {resp.text[:300]}",
            )
        audio_bytes = resp.content
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                "fish-speech became unresponsive mid-request. "
                "Retry in 30s — the manager will re-ensure it on the next call. "
                "Check /root/audio-logs/fish_speech.log for details."
            ),
        )

    job_id = str(uuid.uuid4())
    wav = OUTPUT_DIR / f"{job_id}.wav"
    mp3 = OUTPUT_DIR / f"{job_id}.mp3"
    wav.write_bytes(audio_bytes)
    wav_to_mp3(wav, mp3)

    return {
        "job_id": job_id,
        "engine": "fish-speech",
        "download_wav": f"/api/v1/download/{job_id}?format=wav",
        "download_mp3": f"/api/v1/download/{job_id}?format=mp3",
    }


@app.post("/api/v1/voice/fish_basic")
async def voice_fish_basic(
    text: str = Form(...),
    voice_id: str = Form(""),
    x_api_key: str = Header(None),
):
    """fish-speech TTS without voice cloning (uses default or pre-set voice model)."""
    verify_api_key(x_api_key)
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    logger.info(f"fish_basic: len={len(clean_text)}")
    return await _proxy_fish_speech(clean_text)


@app.post("/api/v1/voice/fish_clone")
async def voice_fish_clone(
    text: str = Form(...),
    speaker_wav: UploadFile = File(...),
    reference_text: str = Form(""),
    language: str = Form(""),
    x_api_key: str = Header(None),
):
    """fish-speech voice cloning — provide reference audio to clone the voice.

    Args:
        text: The text to synthesize in the cloned voice.
        speaker_wav: Reference audio file containing the target voice.
        reference_text: Transcript of the reference audio. Fish-speech S2-Pro
            uses this alongside the audio for significantly better voice
            extraction. When empty, auto-transcribed using Whisper.
        language: ISO 639-1 code (e.g. 'hi', 'en'). Passed to Whisper for
            accurate transcription — avoids Hindi/Urdu script confusion.
        x_api_key: API key for authentication.
    """
    verify_api_key(x_api_key)
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    ref_bytes = await speaker_wav.read()

    # Ensure reference audio is in a format whisper/fish-speech can handle
    ref_wav = _ensure_wav(ref_bytes)

    ref_text = reference_text.strip()
    lang_hint = language.strip()
    # Auto-transcribe reference audio when no transcript provided.
    # Fish-speech S2-Pro quality improves dramatically with reference_text —
    # without it, the model can't align phonemes to the speaker's voice,
    # resulting in default accent (often South Indian for Hindi).
    if not ref_text and ref_wav:
        logger.info(f"fish_clone: reference_text empty — auto-transcribing (lang_hint={lang_hint!r})...")
        ref_text = _auto_transcribe(ref_wav, language_hint=lang_hint)
        if ref_text:
            logger.info(f"fish_clone: auto-transcribed: '{ref_text[:120]}'")
        else:
            logger.warning("fish_clone: auto-transcription returned empty — cloning quality may be reduced")

    logger.info(f"fish_clone: text_len={len(clean_text)} ref_audio={len(ref_wav)} bytes "
                f"(orig={len(ref_bytes)}) ref_text_len={len(ref_text)} ref_text='{ref_text[:80]}'")
    return await _proxy_fish_speech(clean_text, speaker_wav_bytes=ref_wav, reference_text=ref_text)


# ---------------------------------------------------------------------
# MUSIC (MusicGen)
# ---------------------------------------------------------------------
@app.post("/api/v1/music")
async def generate_music(
    prompt: str = Form(...),
    duration: int = Form(30),
    music_model: Optional[str] = Form(None),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)

    job_id = str(uuid.uuid4())
    wav = OUTPUT_DIR / f"{job_id}.wav"
    mp3 = OUTPUT_DIR / f"{job_id}.mp3"

    # Resolve short names to full HF IDs
    resolved_model = ""
    if music_model:
        v = music_model.strip().lower()
        if v in ("large", "facebook/musicgen-large"):
            resolved_model = "facebook/musicgen-large"
        elif v in ("medium", "facebook/musicgen-medium"):
            resolved_model = "facebook/musicgen-medium"
        else:
            resolved_model = music_model  # pass-through for custom IDs

    t_start = time.monotonic()
    try:
        # ModelManager ensures MusicGen is the active kind — swaps out
        # fish-speech / AudioGen / XTTS if needed. Idempotent when already loaded.
        music = await manager.ensure("music", music_variant=resolved_model)
        music.set_generation_params(duration=duration)

        audio = music.generate([prompt])[0].cpu()
        gen_elapsed = round(time.monotonic() - t_start, 2)

        audio_write(
            wav.with_suffix(""),
            audio,
            music.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )

        wav_to_mp3(wav, mp3)

        # NOTE: intentionally NOT unloading MusicGen here. The manager keeps
        # it resident; subsequent BGM requests reuse without reload. Next
        # /voice/fish_* or /sfx request will trigger the swap via manager.ensure.
        # This removes the 180s blocking tail we used to pay on every BGM call.

        return {
            "job_id": job_id,
            "engine": "musicgen",
            "music_model": manager._music_variant,
            "duration_requested_s": duration,
            "elapsed_s": gen_elapsed,
            "download_wav": f"/api/v1/download/{job_id}?format=wav",
            "download_mp3": f"/api/v1/download/{job_id}?format=mp3",
        }

    except HTTPException:
        raise  # already-classified error from manager.ensure
    except Exception as e:
        logger.exception("MusicGen generation failed")
        raise HTTPException(status_code=500, detail=f"MusicGen failed: {e}")

# ---------------------------------------------------------------------
# SOUND EFFECTS (AudioGen)
# ---------------------------------------------------------------------
@app.post("/api/v1/sfx")
async def generate_sfx(
    prompt: str = Form(...),
    duration: int = Form(8),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)

    job_id = str(uuid.uuid4())
    wav = OUTPUT_DIR / f"{job_id}.wav"
    mp3 = OUTPUT_DIR / f"{job_id}.mp3"

    t_start = time.monotonic()
    try:
        # ModelManager ensures AudioGen is the active kind — swaps out
        # fish-speech / MusicGen / XTTS if needed. Idempotent when already loaded.
        sfx = await manager.ensure("sfx")
        sfx.set_generation_params(duration=duration)

        audio = sfx.generate([prompt])[0].cpu()
        gen_elapsed = round(time.monotonic() - t_start, 2)

        audio_write(
            wav.with_suffix(""),
            audio,
            sfx.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
        )

        wav_to_mp3(wav, mp3)

        # NOTE: keep AudioGen resident — subsequent SFX requests reuse.
        # Next /voice/fish_* or /music call triggers the swap via manager.ensure.

        return {
            "job_id": job_id,
            "engine": "audiogen",
            "duration_requested_s": duration,
            "elapsed_s": gen_elapsed,
            "download_wav": f"/api/v1/download/{job_id}?format=wav",
            "download_mp3": f"/api/v1/download/{job_id}?format=mp3",
        }

    except HTTPException:
        raise  # already-classified error from manager.ensure
    except Exception as e:
        logger.exception("AudioGen generation failed")
        raise HTTPException(status_code=500, detail=f"AudioGen failed: {e}")

# ---------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------
@app.get("/api/v1/download/{job_id}")
async def download_audio(
    job_id: str,
    format: str = Query("wav"),
    x_api_key: str = Header(None),
):
    verify_api_key(x_api_key)

    if format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="Invalid format")

    path = OUTPUT_DIR / f"{job_id}.{format}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path,
        media_type="audio/mpeg" if format == "mp3" else "audio/wav",
        filename=f"{job_id}.{format}",
    )

# ---------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------
@app.get("/health")
async def health():
    gpu_name = None
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as e:
        gpu_name = f"GPU error: {e}"

    # Check fish-speech server health (only in fish mode)
    fish_status = "not_applicable"
    if AUDIO_TTS_PROVIDER == "fish":
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{FISH_SPEECH_BASE}/v1/health")
                fish_status = "healthy" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception:
            fish_status = "not_running"

    mgr = manager.status()
    return {
        "status": "healthy",
        "tts_provider": AUDIO_TTS_PROVIDER,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "active_model": mgr["active_kind"],       # single source of truth
        "music_variant": mgr["music_variant"],     # when active_model == "music"
        "vram_gb": mgr["vram_gb"],                 # {allocated, reserved, total}
        "recent_swaps": mgr["recent_swaps"],       # last 10 swaps with timings
        "musicgen_available": _HAS_AUDIOCRAFT,
        "audiogen_available": _HAS_AUDIOCRAFT,
        # Back-compat fields
        "loaded_model": mgr["active_kind"],
        "fish_speech": fish_status,
    }


# ---------------------------------------------------------------------
# MODEL MANAGER ENDPOINTS (C13)
# ---------------------------------------------------------------------

@app.get("/api/v1/status")
async def model_status(x_api_key: str = Header(None)):
    """Non-blocking snapshot of ModelManager state.

    Auth-protected (unlike /health) because it exposes swap history that
    could leak usage patterns. Same payload as /health's manager section.
    """
    verify_api_key(x_api_key)
    return manager.status()


@app.post("/api/v1/prepare")
async def prepare_model(
    kind: str = Form(...),
    music_variant: Optional[str] = Form(None),
    x_api_key: str = Header(None),
):
    """Pre-warm a specific model. Useful before bulk generation.

    Blocks until the model is loaded (or fails). Idempotent — if the
    requested kind is already active, returns immediately.

    Example:
        curl -X POST http://pod:8000/api/v1/prepare \
          -H "x-api-key: ..." \
          -F "kind=music" -F "music_variant=medium"
    """
    verify_api_key(x_api_key)
    if kind not in _VALID_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown kind: {kind!r}. Valid: {sorted(_VALID_KINDS)}",
        )
    resolved_variant = ""
    if kind == "music" and music_variant:
        v = music_variant.strip().lower()
        if v in ("large", "facebook/musicgen-large"):
            resolved_variant = "facebook/musicgen-large"
        elif v in ("medium", "facebook/musicgen-medium"):
            resolved_variant = "facebook/musicgen-medium"
        else:
            resolved_variant = music_variant  # pass-through
    t_start = time.monotonic()
    await manager.ensure(kind, music_variant=resolved_variant)
    return {
        "prepared": True,
        "kind": kind,
        "music_variant": manager._music_variant if kind == "music" else None,
        "elapsed_s": round(time.monotonic() - t_start, 2),
        "vram_gb": _vram_stats_gb(),
    }


@app.post("/api/v1/unload")
async def unload_models(x_api_key: str = Header(None)):
    """Release all VRAM. Useful before shutdown or for operator intervention."""
    verify_api_key(x_api_key)
    result = await manager.unload_all()
    return result

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
