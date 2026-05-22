"""Recorder implementations for consolidating latency metrics."""

from __future__ import annotations

import copy
import time
from typing import Iterable, List, Optional

from .records import BeamDetail, IterationLatencyRecord, StepRecord


class LatencyRecorder:
    """Collects latency metrics for a single environment instance."""

    def __init__(self, question_id: Optional[str] = None) -> None:
        self.reset(question_id=question_id)

    def reset(self, question_id: Optional[str] = None) -> None:
        self.question_id = question_id
        self._question_start: Optional[float] = None
        self._question_latency: float = 0.0
        self._steps: List[StepRecord] = []
        self._pending_step: Optional[StepRecord] = None
        self.rm_latency_history: List[float] = []

    def clone(self) -> "LatencyRecorder":
        cloned = LatencyRecorder(question_id=self.question_id)
        cloned._question_start = self._question_start
        cloned._question_latency = self._question_latency
        cloned._steps = [copy.deepcopy(step) for step in self._steps]
        cloned._pending_step = copy.deepcopy(self._pending_step)
        cloned.rm_latency_history = list(self.rm_latency_history)
        return cloned

    # ------------------------------------------------------------------
    # Question-level helpers
    # ------------------------------------------------------------------
    def start_question(self) -> float:
        self._question_start = time.time()
        self._question_latency = 0.0
        return self._question_start

    def finish_question(self) -> float:
        if self._question_start is not None:
            self._question_latency = time.time() - self._question_start
            self._question_start = None
        return self._question_latency

    @property
    def question_latency(self) -> float:
        return self._question_latency

    # ------------------------------------------------------------------
    # Step-level helpers
    # ------------------------------------------------------------------
    def _ensure_pending_step(self) -> StepRecord:
        if self._pending_step is None:
            index = len(self._steps)
            self._pending_step = StepRecord(index=index)
        return self._pending_step

    def begin_step(self) -> None:
        if self._pending_step is not None:
            self._steps.append(self._pending_step)
        index = len(self._steps)
        self._pending_step = StepRecord(index=index)

    def record_lm_latency(self, latency: float) -> None:
        record = self._ensure_pending_step()
        record.lm = float(latency)

    def record_rm_latency(self, latency: float) -> None:
        record = self._ensure_pending_step()
        record.rm += float(latency)
        self.rm_latency_history.append(float(latency))

    def record_step(
        self,
        *,
        total: float,
        wait: float = 0.0,
        tokens: int = 0,
        prob: float = 0.0,
        model: str = "",
    ) -> StepRecord:
        record = self._ensure_pending_step()
        record.total = float(total)
        record.wait = float(wait)
        record.num_tokens = int(tokens)
        record.prob = float(prob)
        record.model = model
        self._steps.append(record)
        self._pending_step = None
        return record

    def update_last_step(
        self,
        *,
        wait: Optional[float] = None,
        tokens: Optional[int] = None,
        prob: Optional[float] = None,
        model: Optional[str] = None,
    ) -> None:
        record = self.last_step_record
        if record is None:
            return
        if wait is not None:
            record.wait = float(wait)
        if tokens is not None:
            record.num_tokens = int(tokens)
        if prob is not None:
            record.prob = float(prob)
        if model is not None:
            record.model = model

    @property
    def last_step_record(self) -> Optional[StepRecord]:
        if self._pending_step is not None:
            return self._pending_step
        if not self._steps:
            return None
        return self._steps[-1]

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def _all_records(self) -> Iterable[StepRecord]:
        if self._pending_step is not None:
            return self._steps + [self._pending_step]
        return list(self._steps)

    @property
    def step_latency_history(self) -> List[float]:
        return [float(record.total) for record in self._steps]

    @property
    def step_lm_latency_history(self) -> List[float]:
        return [float(record.lm) for record in self._steps]

    @property
    def step_rm_latency_history(self) -> List[float]:
        return [float(record.rm) for record in self._steps]

    @property
    def step_wait_history(self) -> List[float]:
        return [float(record.wait) for record in self._steps]

    def to_serializable(self) -> List[dict]:
        return [record.to_dict() for record in self._steps]

    def iterations_as_dicts(self) -> List[dict]:
        return []


class SearchLatencyRecorder:
    """Aggregates iteration-level latency records."""

    def __init__(self) -> None:
        self._iterations: List[IterationLatencyRecord] = []

    def reset(self) -> None:
        self._iterations.clear()

    def record(self, record: IterationLatencyRecord) -> None:
        self._iterations.append(record)

    @property
    def iterations(self) -> List[IterationLatencyRecord]:
        return list(self._iterations)

    def to_dicts(self) -> List[dict]:
        return [record.to_dict() for record in self._iterations]


class NullLatencyRecorder(LatencyRecorder):
    """No-op recorder for scenarios where metrics are disabled."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, question_id: Optional[str] = None) -> None:  # type: ignore[override]
        self.question_id = question_id
        self._question_start = None
        self._question_latency = 0.0
        self._steps = []
        self._pending_step = None
        self.rm_latency_history = []

    def clone(self) -> "LatencyRecorder":  # type: ignore[override]
        return NullLatencyRecorder()

    def start_question(self) -> float:  # type: ignore[override]
        return time.time()

    def finish_question(self) -> float:  # type: ignore[override]
        return 0.0

    def _ensure_pending_step(self) -> StepRecord:  # type: ignore[override]
        if self._pending_step is None:
            index = len(self._steps)
            self._pending_step = StepRecord(index=index)
        return self._pending_step

    def record_lm_latency(self, latency: float) -> None:  # type: ignore[override]
        return

    def record_rm_latency(self, latency: float) -> None:  # type: ignore[override]
        return

    def record_step(
        self,
        *,
        total: float,
        wait: float = 0.0,
        tokens: int = 0,
        prob: float = 0.0,
        model: str = "",
    ) -> StepRecord:  # type: ignore[override]
        record = StepRecord(
            index=len(self._steps),
            total=total,
            wait=wait,
            num_tokens=tokens,
            prob=prob,
            model=model,
        )
        self._steps.append(record)
        self._pending_step = None
        return record

    def to_serializable(self) -> List[dict]:  # type: ignore[override]
        return []

    def iterations_as_dicts(self) -> List[dict]:  # type: ignore[override]
        return []
