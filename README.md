# label-validity-time

Code, machine-readable rule sets, run logs, figure scripts and aggregate results for the paper

> Q. V. Nguyen, P. T. Duong, D. H. A. Nguyen, D. V. Dinh, N. S. Vo and Z. Qin, "Testing the Validity Time of
> 3D Box Labels in Driving Datasets Against Image Evidence," manuscript for *IEEE Transactions on Intelligent
> Vehicles*, 2026.

The study tests at which instant the released 3D box labels of the Waymo Open Dataset (WOD) and Argoverse 2 (AV2)
are valid, against image evidence that did not produce them: human 2D boxes (WOD) and a COCO-trained 2D detector
(WOD, AV2, MAN TruckScenes). Each rule set was fixed, with a SHA-256 hash in a local log, before the estimates it
governs. A local hash fixes content but not time, so the order is self-reported. The supplementary material
(Section B) prints the hash log.

## Layout

| Folder | Contents |
|---|---|
| `src/careport/` | the study code (`xfc7_*` modules) |
| `tests/` | unit tests (`test_xfc7_*`), all synthetic |
| `config/preanalysis/` | the five machine-readable rule sets (`xfc7_*.json`) |
| `records/` | from the study folder: the scripts run in each step, aggregate results (JSON), run logs, cluster job scripts, the AV2 log lists and the list of the 52 code files run in the AV2 replication, with their hashes |

`records/` keeps the study folder's own relative paths. Every file is a byte-for-byte copy of the study's file, and
Git is told not to convert line endings (`.gitattributes`). The code is each file's latest version.

The written protocols and the running records of the study (freeze records, decisions, plans and their write-ups)
are not included. Their SHA-256 hashes are printed in the supplementary material, Section B, and the texts are
available from the corresponding author on request. The five machine-readable rule sets in `config/preanalysis/` and
the code-hash list can be checked here against that hash log. `PROVENANCE.md` gives the details.

## What runs from this repository

- **The unit tests** run as they are (all synthetic; no data needed). With Python 3.10 or newer and NumPy, pandas,
  PyArrow and SciPy:

  ```
  PYTHONPATH=src python -m unittest discover -s tests -p "test_xfc7_*.py"
  ```

- **The analyses** need the datasets (WOD, AV2, MAN TruckScenes), which are not redistributed here, and the detector
  pass needs a GPU. The scripts carry the paths of the machines they ran on; point them to your own copies.
- **Three steps also need a text that is not included.** The AV2 runs `careport.xfc7_t2_run` and
  `careport.xfc7_t3_run` start only when given their frozen protocol file, which they check by name and SHA-256.
  `careport.xfc7_v7c` checks its plan file the same way. The two config builders
  (`records/tier2_2026-11-01/build_t2_run_config.py`, `records/t3_replication_2026-11-04/build_t3_run_config.py`)
  also read the protocol text. These texts are available from the corresponding author on request.

The detector pass also needs PyTorch and torchvision; the WOD steps need TensorFlow and `waymo-open-dataset`
(Waymo's camera model is called, not reimplemented, except its column-time formula in
`src/careport/xfc7_timing_model.py`).

## Data

No dataset files and no per-observation values are included. The scripts in `records/` are archived exactly as
they were run: they read dataset roots from environment variables or from the machine paths of the computers they
ran on, which appear in them and in the run logs. To run them elsewhere, point those paths to your own copies of
the datasets.

- **WOD:** the segment list and per-observation values are available only to registered WOD users, on request to
  the corresponding author.
- **AV2:** the drawn log lists are in `records/tier2_2026-11-01/` and `records/t3_replication_2026-11-04/`.

This publication was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License
Agreement for Non-Commercial Use, available at waymo.com/open/terms, and your access and use of such work are
governed by the terms and conditions therein. AV2 and MAN TruckScenes are used under CC BY-NC-SA 4.0. Aggregate
results derived from these datasets are provided only to let readers check the paper.

## License

The code is released under the MIT license (`LICENSE`).

The study used generative AI tools as described in the paper's Acknowledgment.
