from __future__ import annotations
from typing import Dict, Any

# Stubs for future auto-thresholds; can compute p01/p99 etc.

def suggest_threshold(metric: str, sample_stats: Dict[str, Any]) -> Any:
    if metric == "null_pct":
        return 0.5
    if metric == "join_coverage":
        return 99.0
    return 1