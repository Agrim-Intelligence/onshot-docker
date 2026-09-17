import time

import pytest

from job_registry import IdempotencyConflict, JobRegistry


def test_submit_returns_deterministic_opaque_job_and_deduplicates():
    registry = JobRegistry()
    first = registry.submit(
        idempotency_key="worker-123",
        kind="music",
        request_fingerprint="fp-1",
    )
    duplicate = registry.submit(
        idempotency_key="worker-123",
        kind="music",
        request_fingerprint="fp-1",
    )

    assert first["job_id"] == duplicate["job_id"]
    assert first["status"] == "queued"
    assert len(first["job_id"]) == 36
    assert "/" not in first["job_id"]


def test_conflicting_idempotency_key_is_rejected():
    registry = JobRegistry()
    registry.submit(idempotency_key="same", kind="sfx", request_fingerprint="one")

    with pytest.raises(IdempotencyConflict):
        registry.submit(idempotency_key="same", kind="sfx", request_fingerprint="two")


def test_status_transitions_and_failure_are_readable():
    registry = JobRegistry()
    submitted = registry.submit(idempotency_key="k", kind="sfx", request_fingerprint="fp")
    job_id = submitted["job_id"]

    registry.mark_running(job_id)
    assert registry.status(job_id)["status"] == "running"
    registry.mark_failed(job_id, "pod returned HTML 524 gateway timeout")
    failed = registry.status(job_id)
    assert failed["status"] == "failed"
    assert failed["error"] == "pod returned HTML 524 gateway timeout"


def test_unknown_job_returns_none():
    assert JobRegistry().status("00000000-0000-0000-0000-000000000000") is None


def test_terminal_records_are_bounded_without_evicting_running_jobs():
    registry = JobRegistry(max_records=2, terminal_ttl_sec=0.01)
    first = registry.submit(idempotency_key="one", kind="sfx", request_fingerprint="fp")
    registry.mark_running(first["job_id"])
    second = registry.submit(idempotency_key="two", kind="sfx", request_fingerprint="fp")
    registry.mark_failed(second["job_id"], "failed")
    time.sleep(0.02)
    third = registry.submit(idempotency_key="three", kind="sfx", request_fingerprint="fp")

    assert registry.status(first["job_id"])["status"] == "running"
    assert registry.status(third["job_id"])["status"] == "queued"
    assert registry.status(second["job_id"]) is None
