"""Structured payload definitions for metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StepRecord:
    """Latency breakdown for a single reasoning step."""

    index: int
    total: float = 0.0
    lm: float = 0.0
    rm: float = 0.0
    wait: float = 0.0
    num_tokens: int = 0
    prob: float = 0.0
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "total": float(self.total),
            "lm": float(self.lm),
            "rm": float(self.rm),
            "wait": float(self.wait),
            "num_tokens": int(self.num_tokens),
            "prob": float(self.prob),
            "model": self.model,
        }


@dataclass
class BeamDetail:
    """Detailed timing for a single beam element within an iteration."""

    beam_idx: int
    node_id: str
    parent_node_id: str
    value: float
    parent_value: float
    total_time: float = 0.0
    lm_latency: float = 0.0
    rm_latency: float = 0.0
    step_wait: float = 0.0
    num_tokens: int = 0
    prob: float = 0.0
    lm_tokens: int = 0
    lm_time_per_token: float = 0.0
    rm_time_per_token: float = 0.0
    kept: bool = False
    is_terminal: bool = False
    text_state: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beam_idx": self.beam_idx,
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "value": float(self.value),
            "parent_value": float(self.parent_value),
            "total_time": float(self.total_time),
            "lm_latency": float(self.lm_latency),
            "rm_latency": float(self.rm_latency),
            "step_wait": float(self.step_wait),
            "num_tokens": int(self.num_tokens),
            "prob": float(self.prob),
            "lm_tokens": int(self.lm_tokens),
            "lm_time_per_token": float(self.lm_time_per_token),
            "rm_time_per_token": float(self.rm_time_per_token),
            "kept": bool(self.kept),
            "is_terminal": bool(self.is_terminal),
            "text_state": self.text_state,
        }


@dataclass
class IterationLatencyRecord:
    """Top-level timing information for one beam-search iteration."""

    iteration: int
    step_latency: float
    step_wait: float
    num_active_beams: int
    num_completed_beams: int
    num_expanded_beams: int
    num_prefix_groups: int
    total_model_time: float
    ideal_batch_time: float
    parallelism_efficiency: float
    beam_details: List[BeamDetail] = field(default_factory=list)
    parent_child_mapping: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "step_latency": float(self.step_latency),
            "step_wait": float(self.step_wait),
            "num_active_beams": int(self.num_active_beams),
            "num_completed_beams": int(self.num_completed_beams),
            "num_expanded_beams": int(self.num_expanded_beams),
            "num_prefix_groups": int(self.num_prefix_groups),
            "total_model_time": float(self.total_model_time),
            "ideal_batch_time": float(self.ideal_batch_time),
            "parallelism_efficiency": float(self.parallelism_efficiency),
            "beam_details": [detail.to_dict() for detail in self.beam_details],
            "parent_child_mapping": {
                key: list(value) for key, value in self.parent_child_mapping.items()
            },
        }
