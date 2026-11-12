"""Hash table for the T2 v1.0 freeze record: the transitive careport import set of the run and of every pre-freeze
step (derived from the source by AST, not listed by hand), the tier-2 tests, the scripts, records and diagnostics
of tier2_2026-11-01, the audit files, the config, and the data tables (by env-relative path). Prints Markdown.
Usage (repository root, env loaded): python <this> <out.md>"""
import ast
import glob
import hashlib
import os
import sys

F = "newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13"
T2 = f"{F}/tier2_2026-11-01"
AUD = f"{F}_protocol_audit"
ROOTS = ["xfc7_t2_run", "xfc7_t2_static", "xfc7_t2_av2_labels", "xfc7_t2_detect", "xfc7_t2_ts_labels",
         "xfc7_t2_ts_detect", "xfc7_t2_pcd"]


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def careport_imports(mod):
    tree = ast.parse(open(f"src/careport/{mod}.py", encoding="utf-8").read())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("careport."):
            out.add(n.module.split(".")[1])
        elif isinstance(n, ast.Import):
            out.update(a.name.split(".")[1] for a in n.names if a.name.startswith("careport."))
    return out


def closure(roots):
    seen, todo = set(), list(roots)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo.extend(careport_imports(m) - seen)
    return sorted(seen)


def env_path(p):
    for k in ("AV2_SENSOR_ROOT", "TRUCKSCENES_DERIVED"):
        if p.startswith(f"${k}"):
            return os.environ[k] + p[len(k) + 1:]
    return p


def rows(paths):
    return [f"| `{p}` | {os.path.getsize(env_path(p)):,} | `{sha(env_path(p))}` |" for p in paths]


def main(argv):
    mods = closure(ROOTS)
    out = ["## Code (transitive careport import set of the run and of every pre-freeze step, derived by AST)", "",
           "| File | Bytes | SHA-256 |", "|---|---|---|"]
    out += rows([f"src/careport/{m}.py" for m in mods])
    out += ["", "## Tests", "", "| File | Bytes | SHA-256 |", "|---|---|---|"]
    out += rows(sorted(glob.glob("tests/test_xfc7_t2_*.py")))
    out += ["", "## Config", "", "| File | Bytes | SHA-256 |", "|---|---|---|"]
    out += rows(["config/preanalysis/xfc7_t2_v1_0.json"])
    recs = sorted(p.replace("\\", "/") for p in glob.glob(f"{T2}/*") if os.path.isfile(p))
    recs += sorted(p.replace("\\", "/") for p in glob.glob(f"{T2}/diagnostics/*") if os.path.isfile(p))
    recs += sorted(p.replace("\\", "/") for p in glob.glob(f"{T2}/superseded/*") if os.path.isfile(p))
    out += ["", "## Scripts, records, diagnostics and superseded summaries (`tier2_2026-11-01/`)", "",
            "| File | Bytes | SHA-256 |", "|---|---|---|"]
    out += rows([p for p in recs if not p.endswith(".pyc")])
    out += ["", "## Audit files", "", "| File | Bytes | SHA-256 |", "|---|---|---|"]
    out += rows(sorted(p.replace("\\", "/") for p in glob.glob(f"{AUD}/T2_V1_0_*") + glob.glob(f"{AUD}/t2_*")))
    data = ["$AV2_SENSOR_ROOT/derived/t2_labels_all50_label_side.parquet",
            "$AV2_SENSOR_ROOT/derived/t2_labels_label_side.parquet",
            "$AV2_SENSOR_ROOT/derived/t2_det/raw/run_manifest.json"]
    for s in ("050", "030"):
        for n in ("reference_table", "association", "moving_analysis_set_t2", "static_all", "moving_assigned",
                  "static_zero_lined"):
            data.append(f"$AV2_SENSOR_ROOT/derived/t2_static/all50_s{s}/{n}.parquet")
    data += ["$TRUCKSCENES_DERIVED/t2_labels_rowthird_label_side.parquet", "$TRUCKSCENES_DERIVED/t2_labels_label_side.parquet"]
    for n in ("reference_table", "association", "moving_analysis_set_t2", "static_all", "moving_assigned", "static_zero_lined"):
        data.append(f"$TRUCKSCENES_DERIVED/t2_static/s050_rowthird/{n}.parquet")
    out += ["", "## Data (outside Git; paths relative to the dataset roots)", "", "| File | Bytes | SHA-256 |", "|---|---|---|"]
    out += rows(data)
    with open(argv[1], "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("modules:", ", ".join(mods))


if __name__ == "__main__":
    main(sys.argv)
