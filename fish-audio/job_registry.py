"""Small, process-local registry for asynchronous audio generation jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple


class IdempotencyConflict(ValueError):
    """Raised when a key is reused for a different request."""


@dataclass
class _Record:
    job_id: str
    idempotency_key: str
    kind: str
    fingerprint: str
    status: str = "queued"
    error: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)


class JobRegistry:
    """Thread-safe idempotency and status map.

    State is intentionally ephemeral: a pod restart forgets all jobs.  Job IDs
    are deterministic UUID5 values, which lets a client recover the status URL
    if the proxy drops the original 202 response after accepting the request.
    """

    UUID_NAMESPACE = uuid.NAMESPACE_URL
    UUID_PREFIX = "onshot-audio"

    def __init__(self, *, max_records: int = 256, terminal_ttl_sec: float = 3600.0):
        self.max_records = max(1, int(max_records))
        self.terminal_ttl_sec = max(0.0, float(terminal_ttl_sec))
        self._lock = threading.Lock()
        self._records: Dict[str, _Record] = {}
        self._keys: Dict[Tuple[str, str], str] = {}

    @classmethod
    def job_id_for(cls, *, kind: str, idempotency_key: str) -> str:
        if not idempotency_key:
            raise ValueError("idempotency_key must not be empty")
        return str(uuid.uuid5(cls.UUID_NAMESPACE, f"{cls.UUID_PREFIX}:{kind}:{idempotency_key}"))

    def submit(
        self,
        *,
        idempotency_key: str,
        kind: str,
        request_fingerprint: str,
    ) -> Dict[str, Any]:
        status, _ = self.submit_with_created(
            idempotency_key=idempotency_key,
            kind=kind,
            request_fingerprint=request_fingerprint,
        )
        return status

    def submit_with_created(
        self,
        *,
        idempotency_key: str,
        kind: str,
        request_fingerprint: str,
    ) -> Tuple[Dict[str, Any], bool]:
        if not idempotency_key:
            raise ValueError("idempotency_key must not be empty")
        with self._lock:
            self._evict_locked()
            key = (kind, idempotency_key)
            existing_id = self._keys.get(key)
            if existing_id:
                existing = self._records.get(existing_id)
                if existing is None:
                    self._keys.pop(key, None)
                elif existing.fingerprint != request_fingerprint:
                    raise IdempotencyConflict(
                        f"idempotency key already belongs to a different {kind} request"
                    )
                else:
                    return self._public(existing), False

            job_id = self.job_id_for(kind=kind, idempotency_key=idempotency_key)
            record = _Record(
                job_id=job_id,
                idempotency_key=idempotency_key,
                kind=kind,
                fingerprint=request_fingerprint,
            )
            self._records[job_id] = record
            self._keys[key] = job_id
            self._evict_locked()
            return self._public(record), True

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status="running")

    def mark_completed(self, job_id: str, **result: Any) -> None:
        self._update(job_id, status="completed", result=result, error=None)

    def mark_failed(self, job_id: str, error: str) -> None:
        detail = str(error).replace("\x00", " ")[:500]
        self._update(job_id, status="failed", error=detail)

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._evict_locked()
            record = self._records.get(job_id)
            return self._public(record) if record else None

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._keys.clear()

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = time.monotonic()
            self._evict_locked()

    def _public(self, record: _Record) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "job_id": record.job_id,
            "status": record.status,
            "error": record.error,
        }
        payload.update(record.result)
        return payload

    def _evict_locked(self) -> None:
        now = time.monotonic()
        expired = [
            record
            for record in self._records.values()
            if record.status in ("completed", "failed")
            and now - record.updated_at >= self.terminal_ttl_sec
        ]
        for record in expired:
            self._remove_locked(record)

        while len(self._records) > self.max_records:
            terminals = [
                record
                for record in self._records.values()
                if record.status in ("completed", "failed")
            ]
            if not terminals:
                break
            self._remove_locked(min(terminals, key=lambda item: item.updated_at))

    def _remove_locked(self, record: _Record) -> None:
        self._records.pop(record.job_id, None)
        self._keys.pop((record.kind, record.idempotency_key), None)
