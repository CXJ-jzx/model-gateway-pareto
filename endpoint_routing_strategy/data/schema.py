"""Immutable facts. Predictions and simulation policies never replace outcomes."""

from dataclasses import dataclass
from datetime import datetime

from endpoint_routing_strategy.models import Observation


@dataclass(frozen=True)
class CanonicalAttempt:
    attempt_id: str
    endpoint_id: str
    sent_at: datetime | None
    finished_at: datetime | None
    result: str
    http_status: int | None
    input_tokens: int | None
    output_tokens: int | None
    e2e_ms: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    flags: tuple[str, ...]

    def observation(self, stream: bool) -> Observation | None:
        # Incomplete attempts do not publish a future result at arrival/send time.
        if self.finished_at is None:
            return None
        return Observation(
            occurred_at=self.finished_at, endpoint_id=self.endpoint_id,
            is_stream=stream, success=self.result == "success" and self.http_status == 200,
            result=self.result, http_status=self.http_status,
            e2e_ms=self.e2e_ms, ttft_ms=self.ttft_ms, tpot_ms=self.tpot_ms,
        )


@dataclass(frozen=True)
class CanonicalRequest:
    request_id: str
    model_id: str
    arrived_at: datetime
    is_stream: bool
    actual_input_tokens: int | None
    actual_output_tokens: int | None
    predicted_input_tokens: int | None
    predicted_output_tokens: int | None
    task_type: str | None
    priority: str | None
    slo_tier: str | None
    final_endpoint_id: str | None
    gateway_result: str
    attempts: tuple[CanonicalAttempt, ...]
    flags: tuple[str, ...]
    source_line: int
    user_id: str | None = None

    @classmethod
    def from_dict(cls, value):
        from .normalize import parse_time

        fields = dict(value)
        fields["arrived_at"] = parse_time(fields["arrived_at"])
        fields["flags"] = tuple(fields["flags"])
        attempts = []
        for raw in fields["attempts"]:
            attempt = dict(raw)
            attempt["sent_at"] = parse_time(attempt["sent_at"])
            attempt["finished_at"] = parse_time(attempt["finished_at"])
            attempt["flags"] = tuple(attempt["flags"])
            attempts.append(CanonicalAttempt(**attempt))
        fields["attempts"] = tuple(attempts)
        return cls(**fields)


@dataclass(frozen=True)
class IncomingRequest:
    """Whitelist of arrival-time inputs. No execution result fields are accepted."""

    schema_version: str
    request_id: str
    user_id: str | None
    model_id: str
    arrived_at: datetime
    is_stream: bool
    predicted_input_tokens: int
    predicted_output_tokens: int
    input_prediction_source: str
    output_prediction_source: str
    task_type: str | None
    priority: str | None
    slo_tier: str | None
    split: str
    source_line: int
    reconstruction: str = "arrival_metadata_from_completion_log"
    payload_available: bool = False

    @classmethod
    def from_dict(cls, value):
        from .normalize import parse_time

        fields = dict(value)
        fields["arrived_at"] = parse_time(fields["arrived_at"])
        return cls(**fields)
