from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineDelta:
    metric: str
    latest: float
    baseline: float

    @property
    def delta(self) -> float:
        return self.latest - self.baseline


def compare_ndjson_snapshots(latest: dict[str, float], baseline: dict[str, float]) -> list[BaselineDelta]:
    return [
        BaselineDelta(metric=name, latest=value, baseline=baseline.get(name, 0.0))
        for name, value in latest.items()
    ]
