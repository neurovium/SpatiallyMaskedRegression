# Spatially Masked Regression — code release

This directory is the self-contained code accompanying:

**Spatially Masked Regression Reveals Local and Distributed Predictability in Electrophysiological Recordings**
Maryam Ostadsharif Memar and Nima Dehghani.
[![arXiv](https://img.shields.io/badge/arXiv-2606.11415-b31b1b.svg)](https://arxiv.org/abs/2606.11415)
[![PDF](https://img.shields.io/badge/PDF-2606.11415-blue.svg)](https://arxiv.org/pdf/2606.11415)
[![HTML](https://img.shields.io/badge/HTML-2606.11415-green.svg)](https://arxiv.org/html/2606.11415v1)

## What's in here

| File | Purpose |
|---|---|
| `main.py` | Top-level driver: iterates over subjects, mask intensities, and (optionally) model types. |
| `train.py` | Intra-subject training and cross-subject evaluation loops. |
| `Test_new_subject_iEEG.py` | Cross-subject iEEG evaluation: per-electrode donor selection. |
| `generate_adj_matrix_ieeg_unseen.py` | CBEM: builds cross-subject Pearson similarity (Hungarian, strict 1-to-1) and the transferred adjacency matrices. |
| `run_coverage_experiment.py` | Trains each subject three times (Local / Non-Local / All coverage) for Figure 6. |
| `Load_preprocess_data.py` | Loads pickled per-subject data; can generate phase / IAAFT / block-shuffle surrogates. |
| `neighborhood.py` | Local-neighborhood definition (`'atlas'` or `'knn'`) and `find_index_mask`. |
| `model.py` | Model factory dispatching on `model_type` ∈ {instantaneous, lagged}. |
| `ReconModel.py` | Core PyTorch modules (`ReconModel`, `LaggedReconModel`). |
| `similarity.py` | Cross-subject Pearson similarity on full concatenated time series. |
| `visualize.py` | Reconstruction quality plots, weight stem-plots, DistCorr metric. |
| `utils.py` | Environment / RNG setup, settings serialization, iEEG preprocessing, Dandiset extraction. |
| `path.py` | All filesystem locations, loaded from `configs/device_path.yaml`. |
| `setting.py` | Experiment settings, loaded from `configs/settings.yaml`. |
| `notebooks/` | Jupyter notebooks that reproduce Figures 2, 3, 4, 6 from the pipeline outputs. |

**Documentation:**

* `METHODS.md` — comprehensive map from every equation in the paper's
  *Material and Methods* section to the function/module that implements
  it. Read this side-by-side with the manuscript to verify the code.


## Setup

1. **Python.** Tested with Python ≥ 3.9 and PyTorch ≥ 1.13. The package
   uses numpy, scipy, scikit-learn, pandas, matplotlib, pyyaml, tqdm, and
   (optionally, for the Dandiset path) dandi, h5py, mne, pynwb, remfile.

2. **Local paths.** Copy
   `configs/device_path.example.yaml` to
   `configs/device_path.yaml` and edit it so the keys point at the
   directories on your machine. No code in this package contains
   hardcoded user-specific paths — everything is resolved through
   `Paths` (see `path.py`).

3. **Experiment settings.** Edit `configs/settings.yaml` to choose the
   dataset, mode, neighborhood method, model type, etc. Every key in
   that file is documented in-line.

## Running

### Intra-subject training

```bash
python main.py
```

This runs the full sweep
$m \in \{0.0,\, 0.25,\, 0.5,\, 0.75,\, 1.0\}$ across all subjects,
under the neighborhood method and model type selected in
`configs/settings.yaml`. If `run_lagged_too: true`, each
(subject, intensity) cell is also run a second time with
`model_type='lagged'`.

### Cross-subject (iEEG)

```bash
# 1. Build transferred adjacency matrices for every (src, dst) pair.
python generate_adj_matrix_ieeg_unseen.py

# 2. Evaluate per-electrode DistCorr and pick the best donor per electrode.
python Test_new_subject_iEEG.py
```

Both scripts read from / write to the directories resolved by
`Paths`. The Hungarian assignment is strictly one-to-one; unmatched
source electrodes are recorded in a `matched_mask_*.npy` next to each
adjacency matrix and are skipped by the evaluation step.

### Cross-subject (EEG)

Set `mode: 'test_new'` and `dataset_type: 'EEG'` in
`configs/settings.yaml`, then run `python main.py`. The driver loads
each donor's `model_type='lagged' || 'instantaneous'` checkpoint from
`Paths.donor_checkpoint_file` and reports DistCorr on the target
subject.

### Coverage experiment (Figure 6)

```bash
python run_coverage_experiment.py
```

Trains each subject three times under three explicit masking
configurations — *Local* (predictors restricted to
$\mathcal{N}^{(s)}(i)$), *Non-Local* (predictors restricted to the
complement of $\mathcal{N}^{(s)}(i)$), and *All* (only the target is
masked). Outputs land under
`<results_dir>/coverage/<dataset_type>/sub_<i>/.../cond-<local|non_local|all>/`
and are picked up by the Figure 6 cell of `notebooks/visualization.ipynb`.

## Reproducing the paper figures (notebooks)

The `notebooks/` folder contains the two Jupyter notebooks that turn the
saved `DCORR_*.npy` / `Mean_Channel_*.npz` / `Best_Channel_*.npy`
artifacts into the paper figures:

| Notebook | Reproduces | Reads |
|---|---|---|
| `notebooks/visualization.ipynb` | Figures 2, 4, 6 (per-subject bars, mask-intensity sweep, coverage) | outputs of `main.py`, `Test_new_subject_iEEG.py`, `run_coverage_experiment.py` |
| `notebooks/visualize_sample_signals.ipynb` | Figure 3 (sample original-vs-reconstructed traces with scale bars) | `Mean_Channel_*.npz` and `Best_Channel_*.npy` |

The notebooks resolve all paths through `notebooks/_helpers.py`, which
locates the right run for a given `(subject, mask_intensity,
neighborhood_method, model_type)` automatically (most-recent timestamp
wins when several training runs match). No path needs to be hardcoded
in the notebooks.

**Run order.** `notebooks/README.md` walks the student through the
exact `settings.yaml` edits and command sequence needed to populate the
inputs of each figure; the short version is:

```bash
# 1. Intra-subject sweeps for both modalities (Fig 4 + intra column of Fig 2)
#    settings.yaml: dataset_type='iEEG' (then 'EEG'), mode='train'
python main.py
python main.py

# 2. Surrogate runs (surrogate column of Fig 2)
#    settings.yaml: dataset_type='Surrogate_iEEG' (then 'Surrogate_EEG'), mode='train'
python main.py
python main.py

# 3. Cross-subject (cross column of Fig 2 + panels B,D of Fig 3)
#    EEG path:
#    settings.yaml: dataset_type='EEG', mode='test_new'
python main.py
#    iEEG path:
python generate_adj_matrix_ieeg_unseen.py
python Test_new_subject_iEEG.py

# 4. Coverage experiment (Fig 6) -- one run per modality
python run_coverage_experiment.py
python run_coverage_experiment.py

# 5. Open the notebooks and run all cells.
jupyter lab notebooks/
```

Figures land in `notebooks/figures/` as both `.png` and `.svg`.

## Reproducibility notes

* The full intensity sweep covers `[0.0, 0.25, 0.5, 0.75, 1.0]`;
  $m=0.0$ leaves the whole neighborhood available as predictors, and
  $m=1.0$ masks the entire neighborhood plus the target. Behaviour
  matches both endpoints described in Section *Analysis of the Effect of
  Local Information* of the manuscript.
* Train / validation / test split is 64% / 16% / 20% (two nested 80/20
  random splits with a fixed seed). Early stopping is based on
  validation loss, and the checkpoint with the lowest validation loss is
  retained for evaluation.
* The reconstruction loss is L1 (mean absolute error); regularisation
  uses fixed $\lambda_1=10^{-5}$, $\lambda_2=10^{-4}$ on all trainable
  parameters.
* DistCorr is computed per-trial for intra-subject, lagged, and
  surrogate columns of Tables `tab:ieeg` and `tab:eeg`; for the
  cross-subject column it is computed once on trial-averaged signals.
* The CBEM similarity matrix $C_{ss'}$ is computed once per electrode
  pair on the within-subject concatenation of trial signals (no
  per-trial pairing). If two subjects have different total recording
  lengths after concatenation, the longer side is truncated; the
  truncation is recorded in `sim_info_*.npy`.
* Cross-subject electrode mapping uses the Hungarian (Jonker–Volgenant)
  algorithm; assignment is strictly one-to-one with no greedy
  destination reuse.

## Tests

The unit tests are self-contained (they stub `torch`/`numpy` where
needed) and can be run individually:

```bash
python test_neighborhood.py
python verify_mask_intensity.py
python test_similarity.py
python test_hungarian.py
python test_model_factory.py
```

## Data availability

* AJILE12 iEEG: <https://dandiarchive.org/dandiset/000055/0.220127.0436>
* Upper Limb Movements EEG: <https://bnci-horizon-2020.eu/database/data-sets>

## Citation
If you use this code, please cite the accompanying paper:

BIBTEX:
  @article{memarDehghani2026spatiallymaskedregressionreveals,
        title={Spatially Masked Regression Reveals Local and Distributed Predictability in Electrophysiological Recordings}, 
        author={Maryam Ostadsharif Memar and Nima Dehghani},
        year={2026},
        eprint={2606.11415},
        archivePrefix={arXiv},
        primaryClass={q-bio.NC},
        url={https://arxiv.org/abs/2606.11415}, 
  }
