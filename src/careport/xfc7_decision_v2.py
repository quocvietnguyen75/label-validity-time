"""XF-C7 protocol v2.0 inference rule (rejection design, after the 2026-10-22 check).

Primary (additive): H is rejected in camera c if the 95 % CI of delta_H,c lies entirely outside [-(5 + B_c), +(5 + B_c)] ms.
Secondary (pair, reported only): CI entirely outside [-5, +5] AND |delta_H,c| > B_c.
H is ruled out in a run if rejected in at least two cameras; the reported verdict requires "ruled out" in both the
primary run and the margin-flag-excluded run, otherwise "association-sensitive, not ruled out". Not being ruled out is
never reported as confirmation.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

EstCI = Tuple[float, float, float]   # (estimate, lower, upper)


def rejected(est_ci: EstCI, budget: float, margin: float = 5.0, rule: str = "additive") -> bool:
    est, lo, hi = est_ci
    if rule == "additive":
        t = margin + budget
        return bool(lo > t or hi < -t)
    if rule == "pair":
        return bool((lo > margin or hi < -margin) and abs(est) > budget)
    raise ValueError(rule)


def decide_rejection(delta: Mapping[str, Mapping[str, EstCI]], budgets: Mapping[str, float], cameras: Sequence[str],
                     margin: float = 5.0, min_cameras: int = 2, rule: str = "additive") -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for h, by_cam in delta.items():
        rej = [c for c in cameras if c in by_cam and rejected(by_cam[c], budgets[c], margin, rule)]
        out[h] = {"rejected_in": rej, "ruled_out": len(rej) >= min_cameras}
    return out


def combine_runs(primary: Mapping[str, Mapping[str, object]], margin_excluded: Mapping[str, Mapping[str, object]]) -> Dict[str, str]:
    out = {}
    for h in primary:
        a, b = bool(primary[h]["ruled_out"]), bool(margin_excluded.get(h, {}).get("ruled_out", False))
        out[h] = "ruled out" if (a and b) else ("association-sensitive, not ruled out" if (a or b) else "not ruled out")
    return out
