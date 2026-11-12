# Provenance

Built on 2026-11-12 by `make_public_repo.py` (kept with the study records, outside this repository) from the
study's working folders. Every file is a byte-for-byte copy; the hashes below were checked in a fresh clone.

## Code versions

The code here is each file's latest version, not a snapshot of each step. The AV2 replication's code is
exactly the frozen bundle: all 52 hashes in
`records/t3_replication_2026-11-04/cluster_bundle_2026-11-05/BUNDLE311_SHA256SUMS` match files here.

For the earlier steps, the records quote the hashes the code had when each step ran, and some files changed
later:

- Two modules of the WOD primary analysis, `src/careport/xfc7_estimators.py` (`5d13d860...` at the v2.0 freeze)
  and `src/careport/xfc7_primary_v2_run.py` (`ed0010e1...`), were extended during the post hoc revisions. On
  1 Nov 2026 the v2.0 human-reference result was re-run with the later code and reproduced all 20 per-camera
  estimates and interval ends exactly (recorded in the detector rules' freeze record).
- The files below are named in the study's records, but their current hash is quoted in none of them: they
  changed after the last hash the records give for them.

  - `records/paper_v7/make_figures.py`
  - `records/pilot/download_av2.sh`
  - `records/pilot/download_waymo.sh`
  - `records/pilot/g4_label_only.py`
  - `records/pilot/g8_projection_smoke.py`
  - `records/pilot/g8_synced_column_law.py`
  - `records/pilot/segment_offsets.py`
  - `records/pilot/verify_listing.py`
  - `src/careport/xfc7_display_run.py`
  - `src/careport/xfc7_exploratory_run.py`
  - `src/careport/xfc7_g5_a3_run.py`
  - `src/careport/xfc7_g8e_run.py`
  - `src/careport/xfc7_gate.py`
  - `src/careport/xfc7_masks.py`
  - `src/careport/xfc7_revision_run.py`
  - `src/careport/xfc7_revision_shift_run.py`
  - `src/careport/xfc7_robustness_run.py`
  - `src/careport/xfc7_s1_run.py`
  - `tests/test_xfc7_exploratory.py`
  - `tests/test_xfc7_kinematics.py`
  - `tests/test_xfc7_masks.py`
  - `tests/test_xfc7_pregate.py`
  - `tests/test_xfc7_revision2.py`
  - `tests/test_xfc7_t3_posthoc_v6r.py`

## Machine paths

Scripts, protocols, configs and logs are kept exactly as they were run or fixed, so they contain the paths of
the machines they ran on (a workstation and a GPU cluster). Several of them are hashed in the supplement's hash
log and cannot be edited without breaking those hashes.

## Left out

- `CLAIM_CHECKS.csv`: topic-selection novelty checks.
- `DATASET_SCREEN.md`: topic-selection dataset screen.
- `PAPERS.csv`: topic-selection literature table.
- `RESPONSE_TO_CROSS_CHECK.md`: topic-selection cross-check response.
- `REVIEW.md`: topic-selection review (choice among candidate studies), not part of the study.
- `SEARCH_LOG.json`: literature and documentation search log (summarized in the supplement, Section A).
- `SEARCH_PLAN.md`: topic-selection search plan.
- `candidates/`: topic-selection candidate cards.
- `dataset_docs_log.jsonl`: dataset documentation look-ups (Section A).
- `every Markdown note in the study folder`: the written protocols and the running records (freeze records, decisions, plans and their write-ups, audits); their hashes are in the supplement, Section B, and the texts are available on request.
- `extension_2026-10-31/NOVELTY_REPORT.md`: literature novelty report.
- `extension_2026-10-31/SEARCH_PLAN.md`: literature search plan.
- `extension_2026-10-31/search_log.jsonl`: literature search log (Section A).
- `paper/`: manuscript drafts, packages and reviews.
- `paper_v4/`: manuscript drafts, packages and reviews.
- `paper_v5/`: manuscript drafts, packages and reviews.
- `paper_v6/`: manuscript drafts, packages and reviews.
- `paper_v7/ (all but the figure scripts and Fig. 2)`: manuscript, supplement build, packages, reviews and correspondence.
- `pilot/e2_synced_cells.csv`: names Waymo Open Dataset segments (the paper keeps the WOD segment list for registered users).
- `pilot/g8_checks_all_segment_medians.csv`: names Waymo Open Dataset segments (as above).
- `presubmission_recheck_2026-11-02/`: pre-submission novelty search (Section A).
- `search_log_raw.jsonl`: raw literature search log (Section A).
- `t3_replication_2026-11-04/download_av2_t3_2026-11-04.log`: download log of the public AV2 files (not an analysis log).
- `t3_replication_2026-11-04/download_av2_t3_2026-11-04_attempt1.log`: download log (as above).
- `t3_replication_2026-11-04/watchdog_e_drive.ps1`: monitor of a personal external drive.
- `t3_replication_2026-11-04/watchdog_e_drive_2026-11-04.log`: monitor of a personal external drive.
- `the protocol audits (a sibling folder)`: audit reports; available on request.
- Dataset files, per-observation values, cached detector boxes and the WOD segment list: see README.

## The supplement's hash log, checked against this repository

Each SHA-256 printed in Section B of the supplementary material, and where to recompute it. Files marked
"not published" are the written protocols and running records; their texts are available from the
corresponding author on request. Every row, published or not, was recomputed from the study's files when
this repository was built.

| Hash (first 16) | What | File in this repository | How to check |
|---|---|---|---|
| `1ddc751a241509f0` | rule set | `records/PILOT_PROTOCOL_XF-C7_v1_0.md` (not published) | text available on request |
| `e4042d64325fe039` | rule set | `records/PILOT_PROTOCOL_XF-C7_v0_1.md` (not published) | text available on request |
| `4190077fe0a0c505` | rule set | `records/PILOT_PROTOCOL_XF-C7_v2_0.md` (not published) | text available on request |
| `8c7b8d098c7027ce` | rule set | `records/PILOT_PROTOCOL_XF-C7_EXT_v1_0.md` (not published) | text available on request |
| `62336ffa37d7b213` | rule set | `records/PILOT_PROTOCOL_XF-C7_T2_v1_0.md` (not published) | text available on request |
| `3571112815965513` | rule set | `records/PILOT_PROTOCOL_XF-C7_T3_v1_0.md` (not published) | text available on request |
| `d1930d5bfece07a0` | rule set | `records/PILOT_PROTOCOL_XF-C7_T3_v1_0_DRAFT.md` (not published) | text available on request |
| `f38ac3d390ce9b21` | rule set | `records/t3_replication_2026-11-04/cluster_bundle_2026-11-05/BUNDLE311_SHA256SUMS` | whole file |
| `27c6a4da3320a3f5` | rule set (machine-readable) | `config/preanalysis/xfc7_protocol_v1_0.json` | whole file |
| `a3642eef827b3725` | rule set (machine-readable) | `config/preanalysis/xfc7_protocol_v2_0.json` | whole file |
| `691d5aa11515979d` | rule set (machine-readable) | `config/preanalysis/xfc7_ext_v1_0.json` | whole file |
| `394607c9c6edca4b` | rule set (machine-readable) | `config/preanalysis/xfc7_t2_v1_0.json` | whole file |
| `9277a3f8db428a46` | rule set (machine-readable) | `config/preanalysis/xfc7_t3_v1_0.json` | whole file |
| `a4cbff9205445f5b` | plan file | `records/t3_replication_2026-11-04/posthoc_2026-11-08/V7R_PLAN_2026-11-08.md` (not published) | text available on request |
| `a6bdf59199a1bb8e` | plan file | `records/t3_replication_2026-11-04/posthoc_2026-11-08/V7R_PLAN_AMENDMENT_A1_2026-11-08.md` (not published) | text available on request |
| `a275ee489caa0409` | plan file | `records/t3_replication_2026-11-04/posthoc_2026-11-10/V7C_PLAN_2026-11-10.md` (not published) | text available on request |
| `ecc926290d24ec40` | Nov. 2 post hoc pin | `records/pilot/PILOT_RECORD_XF-C7.md` (not published) | text available on request |
| `3e67c9fe9d7d7620` | Nov. 3 post hoc pin | `records/pilot/PILOT_RECORD_XF-C7.md` (not published) | text available on request |
| `3c8cc61705f493ed` | Nov. 5 post hoc pin (V5R) | `records/pilot/PILOT_RECORD_XF-C7.md` (not published) | text available on request |
| `07f5f7a80cc27b0f` | Nov. 5 post hoc pin (V5S) | `records/pilot/PILOT_RECORD_XF-C7.md` (not published) | text available on request |
| `df688a71bdc4cc45` | Nov. 8 plan as first fixed | `records/t3_replication_2026-11-04/posthoc_2026-11-08/V6R_PLAN_2026-11-08.md` (not published) | text available on request |
| `6564eee6b4aa861d` | Nov. 8 plan with its three addenda | `records/t3_replication_2026-11-04/posthoc_2026-11-08/V6R_PLAN_2026-11-08.md` (not published) | text available on request |
| `dd69338a24b8d738` | detector weights (COCO) | not included: torchvision's published file `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` | hash that file |
