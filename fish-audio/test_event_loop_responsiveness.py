import asyncio
import os

import pytest

os.environ.setdefault("AUDIO_TTS_PROVIDER", "fish")
os.environ.setdefault("AUDIO_LOG_DIR", "/tmp/onshot-audio-test-logs")

import api_server


@pytest.mark.asyncio
async def test_legacy_generation_does_not_starve_health(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_generate(kind, *, prompt, duration, music_variant="", output_stem):
        started.set()
        await release.wait()
        return {
            "engine": "audiogen" if kind == "sfx" else "musicgen",
            "download_wav": f"/api/v1/download/{output_stem}?format=wav",
            "download_mp3": f"/api/v1/download/{output_stem}?format=mp3",
        }

    monkeypatch.setattr(api_server.manager, "generate_audio", slow_generate)
    task = asyncio.create_task(
        api_server.generate_sfx(prompt="door", duration=2, x_api_key=api_server.API_KEY)
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    health_result = await asyncio.wait_for(api_server.health(), timeout=0.1)
    assert "active_kind" not in health_result
    assert isinstance(health_result, dict)

    release.set()
    result = await asyncio.wait_for(task, timeout=1)
    assert result["engine"] == "audiogen"
