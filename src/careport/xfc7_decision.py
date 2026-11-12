"""XF-C7 decision rule (protocol v1.0 sections 5 and 9).

H is supported if, in every decision camera, the 95 % CI of delta_H,c lies within [-5, +5] ms, and, in every readout
direction whose pooled slope cell passed G5, the 95 % CI of beta_H,r lies within [-10, +10] ms per unit xi. At least one
slope cell must pass G5; fewer than two decision cameras stops the study.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

CI = Tuple[float, float]
ORDER = ["frame", "lidar", "pose", "cam"]


def _inside(ci: CI, margin: float) -> bool:
    return -margin <= ci[0] and ci[1] <= margin


def decide(delta_ci: Dict[str, Dict[str, CI]], slope_ci: Dict[str, Dict[str, CI]], decision_cameras: Sequence[str],
           slope_cells: Sequence[str], margin_ms: float = 5.0, slope_margin: float = 10.0) -> Dict[str, object]:
    if len(decision_cameras) < 2:
        return {"supported": [], "outcome": "stop: fewer than two decision cameras", "detail": {}}
    if len(slope_cells) == 0:
        return {"supported": [], "outcome": "no slope cell passed G5: no hypothesis can be supported", "detail": {}}
    detail: Dict[str, Dict[str, bool]] = {}
    supported: List[str] = []
    for h in [x for x in ORDER if x in delta_ci]:
        cams_ok = {c: _inside(delta_ci[h][c], margin_ms) for c in decision_cameras}
        slopes_ok = {r: _inside(slope_ci[h][r], slope_margin) for r in slope_cells}
        detail[h] = {**{f"delta:{c}": v for c, v in cams_ok.items()}, **{f"slope:{r}": v for r, v in slopes_ok.items()}}
        if all(cams_ok.values()) and all(slopes_ok.values()):
            supported.append(h)
    if len(supported) == 1:
        outcome = f"exactly one supported: {supported[0]}"
    elif not supported:
        outcome = "none supported"
    else:
        outcome = "more than one supported: " + ", ".join(supported)
    return {"supported": supported, "outcome": outcome, "detail": detail}
