# `fixed/notebooks/` — replication notebooks

Two Jupyter notebooks reproduce the figures in the paper using outputs
of the `fixed/` package:

* `visualization.ipynb` — Figures **2, 4, 6**
  (per-subject intra/cross/surrogate bars, mask-intensity sweep,
  electrode coverage).
* `visualize_sample_signals.ipynb` — Figure **3**
  (sample original-vs-reconstructed trace panels with scale bars).

Both notebooks read from the `results_dir` configured in
`configs/device_path.yaml`; no machine-specific paths anywhere.

## Step-by-step run order

The notebooks only *plot*; the underlying training has to be done first.
Run the commands below in this order from the `fixed/` directory. Edit
`configs/settings.yaml` before each command to set `dataset_type` and
`mode` as listed; `main.py` overrides `mask_intensity` and
`train_subject` automatically.

```text
# 1. Set up paths once.
cp configs/device_path.example.yaml configs/device_path.yaml
edit configs/device_path.yaml          # point at your data / results dirs

# 2. Intra-subject sweep (Fig 4 and the 'Intra-Subject' column of Fig 2).
# In settings.yaml: dataset_type='iEEG', mode='train'
python main.py
# In settings.yaml: dataset_type='EEG',  mode='train'
python main.py

# 3. Surrogate runs (the 'Surrogate' column of Fig 2). Use the
# phase-shuffle variant; the table in the paper also reports IAAFT
# and block-shuffle if you want extra columns.
# settings.yaml: dataset_type='Surrogate_iEEG', mode='train'
python main.py
# settings.yaml: dataset_type='Surrogate_EEG',  mode='train'
python main.py

# 4. Cross-subject (the 'Cross-Subject' column of Fig 2 and panels B,D of Fig 3).
# EEG path:
# settings.yaml: dataset_type='EEG', mode='test_new'
python main.py
# iEEG path:
# settings.yaml: dataset_type='iEEG', mode='train'  (the per-subject adj matrices must exist)
python generate_adj_matrix_ieeg_unseen.py
python Test_new_subject_iEEG.py

# 5. Coverage experiment (Fig 6).
# settings.yaml: dataset_type='iEEG', mode='train'
python run_coverage_experiment.py
# settings.yaml: dataset_type='EEG',  mode='train'
python run_coverage_experiment.py
```

Once everything in step 2–5 finishes, open the two notebooks (`jupyter
lab notebooks/` or `jupyter notebook notebooks/`) and run all cells.
Each notebook writes its plots into `notebooks/figures/` as both `.png`
and `.svg`.

## Notebook → figure → command quick reference

| Notebook section | Figure | Reads | Produced by |
|---|---|---|---|
| `visualization.ipynb` § Fig 4 (iEEG) | Fig 4A | `<results_dir>/iEEG_train/sub_*/.../intesity_*/DCORR_*.npy` | `main.py` (sweeps 0/25/50/75/100%) |
| `visualization.ipynb` § Fig 4 (EEG)  | Fig 4B | `<results_dir>/EEG_train/sub_*/.../intesity_*/DCORR_*.npy` | `main.py` |
| `visualization.ipynb` § Fig 2 (EEG)  | Fig 2B | intra: `EEG_train/...`; cross: `EEG_test_new/...`; surrogate: `Surrogate_EEG_train/...` | `main.py` (three runs) |
| `visualization.ipynb` § Fig 2 (iEEG) | Fig 2A | intra: `iEEG_train/...`; cross: `<results_dir>/plot/sub_*/best_Dcorr.npy`; surrogate: `Surrogate_iEEG_train/...` | `main.py` + `generate_adj_matrix_ieeg_unseen.py` + `Test_new_subject_iEEG.py` |
| `visualization.ipynb` § Fig 6 | Fig 6A,B | `<results_dir>/coverage/<dataset>/sub_*/...cond-*/DCORR_*.npy` | `run_coverage_experiment.py` |
| `visualize_sample_signals.ipynb` | Fig 3A–D | intra: `Mean_Channel_*.npz`; iEEG cross: `Best_Channel_*.npy` | `main.py` + `Test_new_subject_iEEG.py` |

## Editing the panel selection (Figure 3)

`visualize_sample_signals.ipynb` has a `PANELS` list near the top:

```python
PANELS = [
    ('iEEG', 'train',    0, 30, 1.0, 'A. iEEG intra-subject'),
    ('iEEG', 'test_new', 0, 35, 1.0, 'B. iEEG cross-subject'),
    ('EEG',  'train',    0,  1, 1.0, 'C. EEG intra-subject'),
    ('EEG',  'test_new', 0,  2, 1.0, 'D. EEG cross-subject'),
]
```

Each tuple is `(dataset, mode, subject, channel, mask_intensity, title)`.
Change subject/channel to pick a more representative trace; the notebook
locates the file automatically through the helper in `_helpers.py`.

## How the helpers work

`_helpers.py` exposes:

* `load_paths()` — instantiates `Paths` from `configs/`.
* `find_dcorr_file(...)`, `load_dcorr(...)`, `mean_var_dcorr(...)` —
  locate and aggregate `DCORR_*.npy` from `main.py`'s output.
* `find_mean_channel_file(...)`, `load_mean_channel(...)` — locate the
  trial-averaged signals saved by `visualize.save_mse_all_trials`.
* `coverage_masks(...)` — builds the three masks used by
  `run_coverage_experiment.py` (Local / Non-Local / All).

All helpers are tolerant of multiple training timestamps under the same
`(subject, intensity, ...)` cell: they return the **most recent** match.
