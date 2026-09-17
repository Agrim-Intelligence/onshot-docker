import asyncio
import json
import os

import pytest

os.environ.setdefault("AUDIO_TTS_PROVIDER", "fish")
os.environ.setdefault("AUDIO_LOG_DIR", "/tmp/onshot-audio-test-logs")
from api_server import JOB_REGISTRY, _submit_async_effect, manager


@pytest.mark.asyncio
async def test_async_submit_returns_before_generation_and_exposes_status(monkeypatch):
    JOB_REGISTRY.clear()
    started = asyncio.Event()

    async def fake_generate(kind, *, prompt, duration, music_variant, output_stem):
        started.set()
        await asyncio.sleep(0)
        return {
            "engine": "audiogen",
            "download_wav": f"/api/v1/download/{output_stem}?format=wav",
            "download_mp3": f"/api/v1/download/{output_stem}?format=mp3",
        }

    monkeypatch.setattr(manager, "generate_audio", fake_generate)
    response = await _submit_async_effect(
        kind="sfx", prompt="door", duration=2, music_model="", idempotency_key="worker-async"
    )
    body = json.loads(response.body)
    assert response.status_code == 202
    assert body["status"] == "queued"
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert JOB_REGISTRY.status(body["job_id"])["status"] == "completed"
