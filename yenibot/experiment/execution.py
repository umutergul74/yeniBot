"""Durable workflow status tracking for long-running experiment jobs."""

from __future__ import annotations

import contextvars
import functools
import json
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Callable, ParamSpec, TypeVar
from uuid import uuid4


P = ParamSpec("P")
R = TypeVar("R")

_CURRENT_JOURNAL: contextvars.ContextVar["WorkflowJournal | None"] = contextvars.ContextVar(
    "yenibot_experiment_workflow_journal",
    default=None,
)

_STATUS_WRITE_ATTEMPTS = 6
_STATUS_WRITE_BASE_DELAY_SECONDS = 0.25
_RETRYABLE_STATUS_ERRNOS = {
    5,    # transient Drive/remote I/O failure
    11,   # resource temporarily unavailable
    16,   # device or resource busy
    32,   # sharing violation on Windows-backed mounts
    103,  # software caused connection abort
    104,  # connection reset by peer
    107,  # transport endpoint not connected
    110,  # connection timed out
    116,  # stale file handle
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def _retryable_status_error(error: OSError) -> bool:
    return (
        isinstance(error, (ConnectionAbortedError, ConnectionResetError, TimeoutError))
        or error.errno in _RETRYABLE_STATUS_ERRNOS
        or getattr(error, "winerror", None) in {32, 53, 64, 121}
    )


class WorkflowJournal:
    """Atomically persist the current stage and terminal workflow outcome."""

    def __init__(self, workflow: str, path: Path) -> None:
        self.workflow = workflow
        self.path = path
        self.started_at = _utc_now()
        self.started_clock = monotonic()
        self.payload: dict[str, Any] = {
            "workflow": workflow,
            "status": "running",
            "current_stage": "starting",
            "started_at": self.started_at,
            "updated_at": self.started_at,
            "history": [],
        }
        self._write()

    def bind(self, path: Path) -> None:
        """Move subsequent status updates to a run-specific path."""

        old_path = self.path
        self.path = path
        persisted = self._write()
        if persisted and old_path != path:
            try:
                if old_path.exists():
                    old_path.unlink()
            except OSError as error:
                self._warn_persistence_failure("cleanup", old_path, error)

    def checkpoint(self, stage: str, **context: Any) -> None:
        now = _utc_now()
        event = {"stage": stage, "timestamp": now, **_json_value(context)}
        self.payload["status"] = "running"
        self.payload["current_stage"] = stage
        self.payload["updated_at"] = now
        self.payload["history"] = [*self.payload.get("history", []), event][-100:]
        self._write()

    def complete(self) -> None:
        now = _utc_now()
        self.payload.update(
            {
                "status": "completed",
                "current_stage": "completed",
                "updated_at": now,
                "completed_at": now,
                "duration_seconds": monotonic() - self.started_clock,
            }
        )
        self._write()

    def fail(self, error: BaseException) -> None:
        now = _utc_now()
        self.payload.update(
            {
                "status": "failed",
                "updated_at": now,
                "failed_at": now,
                "duration_seconds": monotonic() - self.started_clock,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(
                        traceback.format_exception(type(error), error, error.__traceback__)
                    )[-12000:],
                },
            }
        )
        self._write()

    def _warn_persistence_failure(
        self,
        operation: str,
        path: Path,
        error: OSError,
    ) -> None:
        event = {
            "operation": operation,
            "path": str(path),
            "timestamp": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        self.payload["persistence_warnings"] = [
            *self.payload.get("persistence_warnings", []),
            event,
        ][-20:]
        warnings.warn(
            "Workflow status persistence failed after retries; the model workflow "
            f"will continue and retry at the next checkpoint. path={path} "
            f"error={type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=3,
        )

    def _write(self) -> bool:
        serialized = json.dumps(_json_value(self.payload), indent=2, sort_keys=True)
        last_error: OSError | None = None

        for attempt in range(_STATUS_WRITE_ATTEMPTS):
            temporary = self.path.with_name(
                f".{self.path.name}.{uuid4().hex}.tmp"
            )
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(serialized, encoding="utf-8")
                temporary.replace(self.path)
                return True
            except OSError as error:
                last_error = error
                try:
                    if temporary.exists():
                        temporary.unlink()
                except OSError:
                    pass
                if (
                    not _retryable_status_error(error)
                    or attempt + 1 >= _STATUS_WRITE_ATTEMPTS
                ):
                    break
                sleep(
                    min(
                        _STATUS_WRITE_BASE_DELAY_SECONDS * (2**attempt),
                        2.0,
                    )
                )

        assert last_error is not None
        self._warn_persistence_failure("write", self.path, last_error)
        return False


def traced_workflow(
    workflow: str,
    initial_path: Callable[P, Path],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Track a workflow without forcing its implementation into one giant try block."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            journal = WorkflowJournal(workflow, initial_path(*args, **kwargs))
            token = _CURRENT_JOURNAL.set(journal)
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                journal.fail(error)
                raise
            else:
                journal.complete()
                return result
            finally:
                _CURRENT_JOURNAL.reset(token)

        return wrapped

    return decorator


def workflow_checkpoint(
    stage: str,
    *,
    status_path: str | Path | None = None,
    **context: Any,
) -> None:
    journal = _CURRENT_JOURNAL.get()
    if journal is None:
        return
    if status_path is not None:
        journal.bind(Path(status_path))
    journal.checkpoint(stage, **context)


def training_status_path(
    frame: Any,
    config: dict[str, Any],
    *,
    checkpoint_dir: str | Path,
    **_: Any,
) -> Path:
    del frame, config
    return Path(checkpoint_dir) / "experiments" / "_training_workflow_status.json"


def diagnostics_status_path(
    *,
    output_dir: str | Path,
    **_: Any,
) -> Path:
    return Path(output_dir) / "experiments" / "_diagnostics_workflow_status.json"


__all__ = [
    "WorkflowJournal",
    "diagnostics_status_path",
    "traced_workflow",
    "training_status_path",
    "workflow_checkpoint",
]
