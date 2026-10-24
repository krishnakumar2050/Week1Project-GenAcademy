"""Application use case: snapshot, execute, and attach reproducible run metadata."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from coding_review.backends.base import ReviewBackend
from coding_review.backends.factory import create_backend
from coding_review.config import ReviewConfig
from coding_review.graph import build_graph
from coding_review.sources import load_files

logger = logging.getLogger(__name__)


def review_repository(
    repository: Path,
    config: ReviewConfig,
    backend: ReviewBackend | None = None,
    files: dict[str, str] | None = None,
) -> dict:
    snapshot = (
        dict(files)
        if files is not None
        else load_files(repository, config.max_bytes, config.exclude)
    )
    if not snapshot:
        raise ValueError("No supported source files found")
    snapshot_bytes = sum(len(value.encode()) for value in snapshot.values())
    if snapshot_bytes > config.max_bytes:
        raise ValueError("Snapshot exceeds byte budget")
    run_id = str(uuid4())
    started = datetime.now(timezone.utc).isoformat()
    clock = monotonic()
    logger.info(
        "review_started run_id=%s backend=%s files=%d", run_id, config.backend, len(snapshot)
    )
    model = backend if backend is not None else create_backend(config)
    state = build_graph(model).invoke(
        {"files": snapshot, "results": []}, config={"max_concurrency": config.max_concurrency}
    )
    report = state["report"]
    report.update(
        schema_version="1.0",
        run_id=run_id,
        started_at=started,
        completed_at=datetime.now(timezone.utc).isoformat(),
        duration_seconds=round(monotonic() - clock, 3),
        backend=config.backend,
        repository=str(repository.resolve()),
        snapshot_sha256=hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        snapshot_bytes=snapshot_bytes,
    )
    requested = {a["execution"].get("requested_model") for a in report["agents"]}
    report["requested_model"] = config.model or (
        next(iter(requested)) if len(requested) == 1 else None
    )
    models = report["actual_models"]
    report["model"] = (
        models[0]
        if len(models) == 1 and all(a["execution"].get("actual_model") for a in report["agents"])
        else None
    )
    logger.info(
        "review_finished run_id=%s status=%s findings=%d duration_seconds=%s",
        run_id,
        report["status"],
        len(report["findings"]),
        report["duration_seconds"],
    )
    return report
