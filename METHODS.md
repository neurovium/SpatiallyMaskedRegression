# Paper Methods → Code map

This document walks through the *Material and Methods* section of the
manuscript and points each equation, algorithm, and procedure to the
specific function (file + symbol) in this package that implements it.
The aim is so that any user can open the paper next to this file and
use/verify the code in a single pass.

The math notation matches the paper exactly:

* $s$ — target subject; $s'$ — reference subject.
* $N_s$ — number of channels for subject $s$.
* $T$ — number of time samples per channel.
* $\mathcal{N}^{(s)}(i)$ — neighborhood of electrode $i$ for subject $s$.
* $K_{ij}^{(s)} \in \{0,1\}$ — binary spatial mask.
* $w_{ij}^{(s)}$ — learnable weight from electrode $j$ to electrode $i$.
* $M_{s}^{\mathrm{learn}}$, $M_{s}^{\mathrm{est}}$ — learned and
  transferred relationship matrices.
* $C_{ss'}(i,j)$ — Pearson cross-subject correlation.
* $\mathbf P^{(s\leftarrow s')}$ — assignment matrix.
* $\mathcal{T}$ — set of temporal lags.

---

## 1. Dataset preprocessing

Section *Dataset* describes 2-second epochs centered on movement events
and (where applicable) per-channel preprocessing.

| Paper detail | File · symbol |
|---|---|
| iEEG preprocessing chain (DC + detrend, band-pass, notch, low-pass, artifact rejection, CAR, optional Hilbert envelope, z-score) | `utils.py · preprocess_iEEG` |
| Raw ECoG → trial epochs (MNE) | `utils.py · epoch_ECoG_MNE`, `preprocess_ecog` |
| PSD-based bad-trial rejection | `utils.py · rem_bad_trials_PSD` |
| Loading already-pickled per-subject arrays | `Load_preprocess_data.py · load_per_subject` |
| Dandiset extraction (AJILE12 fetch via DANDI/NWB) | `utils.py · extract_data_dandi` |

The intra-subject `z`-score that appears just before training is done in
`train.py::train_model` via `scipy.stats.zscore(data)` with default
`axis=0` (across trials).

---

## 2. Spatially Masked Regression Model

### 2.1 Neighborhood $\mathcal{N}^{(s)}(i)$

The paper supports two complementary constructions of the local
neighborhood. Either can be selected at run-time via
`settings.neighborhood_method`.

**Atlas-based (`'atlas'`):**

$$
\mathcal{N}^{(s)}_{\mathrm{atlas}}(i) = \{\, j \neq i \;:\; \mathrm{label}(j) = \mathrm{label}(i) \,\},
$$

with $\mathrm{label}(\cdot)$ being the AAL anatomical region (iEEG) or
the 10-10/10-5 montage group (EEG).

| Paper detail | File · symbol |
|---|---|
| AAL iEEG neighborhood (same anatomical label as target) | `neighborhood.py · _neighborhood_ieeg_aal` |
| 10-10/10-5 EEG neighborhood (regional group containing target) | `neighborhood.py · _neighborhood_eeg_montage` (groups defined in `_EEG_MONTAGE_GROUPS`) |

**Geometric ($k$-nearest, `'knn'`):**

$$
\mathcal{N}^{(s)}_{\mathrm{knn}}(i) = \underset{\substack{S \subset \{1,\ldots,N_s\}\setminus\{i\}\\|S|=k}}{\arg\min} \sum_{j\in S} \big\lVert \Pi_{\mathrm{mod}}(\mathbf{p}^{(s)}_j) - \Pi_{\mathrm{mod}}(\mathbf{p}^{(s)}_i)\big\rVert_{2},
$$

with $\Pi_{\mathrm{mod}}$ the modality-specific 2-D projection
($\Pi_{\mathrm{EEG}}(x,y,z) = (x,y)$;
$\Pi_{\mathrm{iEEG}}(x,y,z) = (x,z)$).

| Paper detail | File · symbol |
|---|---|
| $k$-nearest electrodes by projected Euclidean distance | `neighborhood.py · _neighborhood_knn` |
| Common dispatch | `neighborhood.py · find_neighborhood` |

### 2.2 Binary spatial mask

$$
K_{ij}^{(s)} =
\begin{cases}
1, & j \in \mathcal{N}^{(s)}(i),\\
0, & \text{otherwise.}
\end{cases}
$$

The masking-intensity sweep of Section *Analysis of the Effect of Local
Information* samples a fraction $m \in \{0, 0.25, 0.5, 0.75, 1.0\}$ of
$\mathcal{N}^{(s)}(i)$ to mask, together with the target electrode $i$:

$$
\bigl|\mathrm{mask}(i,\,m)\bigr| = \bigl\lceil m\,|\mathcal{N}^{(s)}(i)|\bigr\rfloor + 1.
$$

| Paper detail | File · symbol |
|---|---|
| Build the per-electrode index set returned to the trainer | `neighborhood.py · find_index_mask` |
| Drive the full sweep across subjects, intensities, and (optionally) model types | `main.py · main` (uses `MASK_INTENSITY_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]`) |
| Apply the mask inside the forward pass (drop masked channels before the linear projection) | `ReconModel.py · ReconModel.forward` and `model.py · LaggedReconModelConfigurable.forward` |

### 2.3 Predicted (reconstructed) signal

$$
\widehat{x}_i^{(s)} = \sum_{j=1}^{N_s} (1 - K_{ij}^{(s)})\, w_{ij}^{(s)}\, x_j^{(s)}.
$$

| Paper detail | File · symbol |
|---|---|
| Linear reconstruction across the unmasked channels | `ReconModel.py · ReconModel.forward` |
| Equivalent for the lagged variant | `model.py · LaggedReconModelConfigurable.forward`, `ReconModel.py · LaggedReconModel.forward` |

---

## 3. Model Training

### 3.1 Objective

Per the manuscript prose, the data term is the L1 loss and the
regularization is Elastic Net. For target electrode $i$:

$$
\min_{\theta_i^{(s)}} \;
\frac{1}{T}\sum_{t=1}^{T}
\Bigg| x_i^{(s)}(t) - \sum_{j=1}^{N_s} (1-K_{ij}^{(s)})\,w_{ij}^{(s)}\,x_j^{(s)}(t) \Bigg|
+ \lambda_1\lVert \theta_i^{(s)}\rVert_1 + \lambda_2\lVert \theta_i^{(s)}\rVert_2^2,
\quad \text{s.t. } w_{ij}^{(s)}=0\ \forall j\in\mathcal{N}^{(s)}(i),
$$

where $\theta_i^{(s)}=(\mathbf w_i^{(s)},\,b_i^{(s)})$ collects the
linear-layer weight and bias.

| Paper detail | File · symbol |
|---|---|
| L1 data term | `train.py · train_model` (`criterion = nn.L1Loss()`) |
| Elastic Net penalty on all trainable parameters | `train.py · train_model` (the `l1_penalty` / `l2_penalty` block) |
| Fixed coefficients $\lambda_1=10^{-5}$, $\lambda_2=10^{-4}$ | `train.py · train_model` (`l1_lambda`, `l2_lambda`) |
| Adam optimizer | `train.py · train_model` (`torch.optim.Adam`) |

### 3.2 Train / validation / test split, early stopping

The implementation uses a 64 / 16 / 20 split obtained from two nested
80/20 random splits with a fixed seed; early stopping is on the
validation loss; the lowest-val-loss checkpoint is restored before
evaluation.

| Paper detail | File · symbol |
|---|---|
| Nested 80/20 splits | `train.py · train_model` (two `sklearn.model_selection.train_test_split(...)`) |
| Validation-loss early stopping with patience | `train.py · train_model` (`patience_counter` / `best_val_loss`) |
| Restore best checkpoint at the end of training | `train.py · train_model` (`model.load_state_dict(torch.load(...))`) |
| z-score across trials before splitting | `train.py · train_model` (`zscore(data)`) |

### 3.3 Intra-subject pseudocode (Algorithm 1 in the paper)

```
for i = 1 to N_s:
    initialize w_i; train under masking;
    update with Adam; stop on val-loss patience;
    M_s_learn[i, :] <- w_i
return M_s_learn
```

| Paper detail | File · symbol |
|---|---|
| Per-electrode loop + weight collection into $M_s^{\mathrm{learn}}$ | `train.py · train_model` (outer `for i in range(...)` and the post-training `weights = model.state_dict()[fc_weight_key]` block) |
| Saved $M_s^{\mathrm{learn}}$ on disk | `train.py · train_model` (writes `adj_mat_<subject>.npy`) |

---

## 4. Cross-subject evaluation

### 4.1 Cross-subject correlation matrix

$$
C_{ss'}(i,j) = \frac{\mathrm{cov}\!\big(x_i^{(s)},\,x_j^{(s')}\big)}{\sigma\!\big(x_i^{(s)}\big)\,\sigma\!\big(x_j^{(s')}\big)},
$$

where $x_i^{(s)}$ is the **full** within-subject time series of
electrode $i$ (trials concatenated). When the two subjects' total
recording lengths differ, both are truncated to the common length.

| Paper detail | File · symbol |
|---|---|
| Vectorised Pearson on concatenated trial signals | `similarity.py · cross_subject_pearson` |
| Concatenate-trials helper | `similarity.py · _concat_trials` |
| Pairwise Pearson between rows | `similarity.py · _pearson_rows` |
| Length-truncation policy + diagnostics | `similarity.py · cross_subject_pearson` (`length_policy` arg, returned `info` dict) |
| Legacy per-trial-mean version (for comparison / reproducibility audits) | `similarity.py · cross_subject_pearson_legacy_per_trial_mean` |

### 4.2 Rectangular assignment

$$
\mathbf P^{(s\leftarrow s')} = \arg\max_{\mathbf P \in \mathcal{P}} \sum_{r,c} P[r,c]\,C_{ss'}[r,c],
\qquad
\mathcal{P} = \{\,\mathbf P \in \{0,1\}^{N_s\times N_{s'}} : \mathbf P\mathbf 1 \le \mathbf 1,\ \mathbf P^\top\mathbf 1 \le \mathbf 1\,\}.
$$

Solved by the Hungarian (Jonker–Volgenant) algorithm. The assignment is
strictly one-to-one: when $N_s \ne N_{s'}$, only
$\min(N_s,\,N_{s'})$ source electrodes receive a donor; unmatched rows
remain empty.

| Paper detail | File · symbol |
|---|---|
| Strict 1-to-1 Hungarian via `scipy.optimize.linear_sum_assignment` | `generate_adj_matrix_ieeg_unseen.py · hungarian_strict` |
| Boolean `matched_mask` of source rows that received a donor | `generate_adj_matrix_ieeg_unseen.py · hungarian_strict` (second return value) |
| NaN-safe cost matrix (NaN replaced with min finite before Hungarian) | `generate_adj_matrix_ieeg_unseen.py · hungarian_strict` |

### 4.3 Transferring the learned matrix

$$
M_{s\leftarrow s'}^{\mathrm{est}} = \mathbf P^{(s\leftarrow s')}\,M_{s'}^{\mathrm{learn}}\,\mathbf P^{(s\leftarrow s')\top}.
$$

| Paper detail | File · symbol |
|---|---|
| Row-/column-permutation of donor matrix into target electrode space | `generate_adj_matrix_ieeg_unseen.py · build_cross_subject_adjacency` (the inner `for m, n in range(n_src): ...` loop using `mapping`) |
| Unmatched rows/columns left at zero (per `matched_mask`) | same loop (the `continue` guards) |
| Output files: `adj_mat_corr_sub<src>_base_sub<dst>.npy`, `matched_mask_sub<src>_base_sub<dst>.npy`, `sim_info_sub<src>_base_sub<dst>.npy` | `build_cross_subject_adjacency` (the `np.save(...)` calls) |

### 4.4 Per-electrode donor selection (Algorithm 2)

For each target electrode $i$, the donor $s'_i$ giving the highest
DistCorr on a held-out validation half of the target subject is selected
for evaluation:

$$
s'_i = \arg\max_{s' \neq s} \mathrm{DistCorr}\!\big(\overline{x}_{i,\mathrm{val}}^{(s)},\ M_{s\leftarrow s'}^{\mathrm{est}}[i,:]\,\overline{X}_{\mathrm{val}}^{(s)}\big),
\qquad
M_s^{\mathrm{est}}[i,:] = M_{s\leftarrow s'_i}^{\mathrm{est}}[i,:].
$$

| Paper detail | File · symbol |
|---|---|
| 50/50 validation/test split of the target subject's trials | `Test_new_subject_iEEG.py · run_cross_subject_eval` (`train_test_split(data_src, test_size=0.5, random_state=42)`) |
| Trial-averaged $\overline{x}_i^{(s)}$ | same function (`val_mean = np.mean(val_data, axis=0)`, `test_mean = np.mean(test_data, axis=0)`) |
| Per-electrode validation DistCorr | same function (inner loop, `Dcorr_val = distance_correlation(...)`) |
| Per-electrode test DistCorr (reported in the *Cross-Subject* column) | same function (`Dcorr_test = distance_correlation(...)`) |
| Best-donor pick per electrode and saved plot | `Test_new_subject_iEEG.py · _plot_cross_subject` (uses `val_Dcorr` to select `idx_best`) |
| Skipping unmatched electrodes (zero rows from strict Hungarian) | `run_cross_subject_eval` (the `if not matched_mask[m]: continue` block) |

### 4.5 EEG-specific shortcut

For EEG the electrode layout is shared, so the EEG cross-subject path
simply loads a donor checkpoint (no Hungarian step) and computes
DistCorr on the target subject.

| Paper detail | File · symbol |
|---|---|
| Random donor selection | `train.py · evaluate_model` (`random.choice(...)`) |
| Donor checkpoint loaded from `Paths.donor_checkpoint_file` | same function |
| Per-trial DistCorr reported | `visualize.py · save_mse_all_trials` |

---

## 5. Lagged variation of SMR

$$
\widehat{x}_i^{(s)}(t) = \sum_{\tau \in \mathcal{T}\cup\{0\}}\sum_{j=1}^{N_s} (1 - K_{ij}^{(s)})\, w_{ij}^{(s)}[\tau]\, x_j^{(s)}(t-\tau),
$$

with $\mathcal{T}=\{20, 30, 50, 60\}$ ms in the manuscript runs and the
$\tau=0$ branch always present as the instantaneous term. Sample offsets
are obtained as $\lceil (\tau/1000)\,f_s\rceil$.

| Paper detail | File · symbol |
|---|---|
| Instantaneous baseline (model_type = `'instantaneous'`) | `ReconModel.py · ReconModel` |
| Lagged variant with $\mathcal{T}\cup\{0\}$ | `ReconModel.py · LaggedReconModel`, `model.py · LaggedReconModelConfigurable` |
| Forward pass: sum over `fc_main` (lag 0) and each `fc_lags[lag]` | `model.py · LaggedReconModelConfigurable.forward` |
| Lag-set configuration | `settings.lags_ms`, propagated via `model.py · build_model` |
| Optional pickling of the full per-lag weight tensor | `train.py · train_model` (writes `lagged_weights/lagged_weights_node_<i>.npy` via `model.collect_lagged_weights`) |
| Choice between instantaneous and lagged | `model.py · build_model` (dispatches on `settings.model_type`) |
| Bundled "run both variants" sweep | `main.py · main` (the `run_lagged_too` branch) |

---

## 6. Distance correlation (DistCorr)

For $u = x_i^{(s)}$, $v = \widehat{x}_i^{(s)}$ of length $T$:

$$
a_{mn} = |u_m - u_n|,\qquad b_{mn} = |v_m - v_n|,
$$

$$
A_{mn} = a_{mn} - \bar a_{m\cdot} - \bar a_{\cdot n} + \bar a_{\cdot\cdot},\qquad
B_{mn} = b_{mn} - \bar b_{m\cdot} - \bar b_{\cdot n} + \bar b_{\cdot\cdot},
$$

$$
\mathrm{dCov}^2(u,v) = \frac{1}{T^2}\sum_{m,n} A_{mn} B_{mn},\quad
\mathrm{dVar}^2(u) = \frac{1}{T^2}\sum_{m,n} A_{mn}^2,\quad
\mathrm{dVar}^2(v) = \frac{1}{T^2}\sum_{m,n} B_{mn}^2,
$$

$$
\mathrm{DistCorr}(u,v) = \frac{\mathrm{dCov}(u,v)}{\sqrt{\mathrm{dVar}(u)\,\mathrm{dVar}(v)}},
$$

with the convention $\mathrm{DistCorr}=0$ when the denominator is zero.

| Paper detail | File · symbol |
|---|---|
| Pairwise distance matrices $a, b$ | `visualize.py · distance_correlation` (`a = np.abs(x[:, None] - x[None, :])`, similarly for `b`) |
| Double-centered matrices $A, B$ | same function (`A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()`) |
| dCov, dVar, DistCorr | same function (`dcov`, `dvar_x`, `dvar_y`, final ratio) |
| Independent copy used by the cross-subject script (so it can run without importing torch) | `Test_new_subject_iEEG.py · distance_correlation` |

### 6.1 Per-trial vs. trial-averaged granularity

For intra-subject / lagged / surrogate columns of
Tables `tab:ieeg` / `tab:eeg`, DistCorr is computed **per trial** and
averaged across trials. For the cross-subject column it is computed
**once on the trial-averaged signals** (validation and test halves
averaged separately).

| Paper detail | File · symbol |
|---|---|
| Per-trial DistCorr averaged across trials | `visualize.py · save_mse_all_trials` (returned list `dcorr_all`) |
| Trial-averaged DistCorr (cross-subject) | `Test_new_subject_iEEG.py · run_cross_subject_eval` (the `recon_test = zscore(np.sum(weights * test_mean, axis=0))` block) |

---

## 7. Surrogate data

### 7.1 Phase-shuffled

Preserves the amplitude spectrum and randomises positive-frequency
phases with Hermitian symmetry:

$$
\mathcal{F}_{\mathrm{surr}} = |\mathcal{F}_S|\,\exp(i\phi'),\qquad \phi'[-k] = -\phi'[k],
$$

with DC (and Nyquist when applicable) phase pinned to zero.

| Paper detail | File · symbol |
|---|---|
| `np.fft.fft`-based shuffle with explicit Hermitian symmetry | `Load_preprocess_data.py · shuffle_phase` |
| Alternative `np.fft.rfft`-based version | `Load_preprocess_data.py · generate_phase_shuffled_surrogate` |
| Per-trial-per-channel application in `load_per_subject` | `Load_preprocess_data.py · load_per_subject` (the `'Surrogate' in dataset_type` branch) |

### 7.2 IAAFT

Iterative alternation between matching the target power spectrum and the
exact amplitude distribution; fixed $n_{\mathrm{iter}}=5$ iterations.

| Paper detail | File · symbol |
|---|---|
| 1-D IAAFT loop | `Load_preprocess_data.py · iaaft_1d` |
| Per-trial-per-channel application | `Load_preprocess_data.py · load_per_subject` (the `'IAAFT' in dataset_type` branch) |

### 7.3 Block-shuffle

Divides each electrode's signal into blocks of $B$ seconds and permutes
them; leftover samples are appended at the end.

| Paper detail | File · symbol |
|---|---|
| Block permutation | `Load_preprocess_data.py · block_shuffle_1d` |
| Sample-count conversion $B\,f_s$ using the dataset's sampling rate | `Load_preprocess_data.py · load_per_subject` (`block_size_samples = int(block_size_sec * fs)`; `fs` defaults to `settings.fs`) |

Block size $B$ is `block_size_sec` in `load_per_subject`; default is 0.5
seconds.

---

## 8. Hyperparameter Optimization

| Paper detail | File · symbol |
|---|---|
| `max_epoch`, `batch_size`, `lr`, `patience`, `mask_intensity`, `fs` | `setting.py · Settings` (one property each); read from `configs/settings.yaml` |
| Fixed regularisation coefficients $\lambda_1, \lambda_2$ | hard-coded in `train.py · train_model` (intentional, to match the released numbers) |
| Lag set $\mathcal{T}$ for the lagged variant | `settings.lags_ms`, consumed by `model.py · build_model` |
| Neighborhood size $k$ for the geometric construction | `settings.neighborhood_k`, consumed by `neighborhood.py · find_index_mask` |

---

## 9. Reproducibility

| Paper detail | File · symbol |
|---|---|
| Deterministic seeds across `random`, `numpy`, `torch` | `utils.py · configure_environment` |
| Fixed seed in the train/val/test split (`random_state=42`) | `train.py · train_model`; `Test_new_subject_iEEG.py · run_cross_subject_eval` |
| `Paths` writes a timestamped subdirectory with `nbhd-<method>/model-<type>/intesity_<percent>/` so atlas vs k-NN and instantaneous vs lagged runs do not overwrite each other | `path.py · Paths.create_paths` |
| Adjacency artifacts include a `neighborhood_config.npy` dump of `{method, k, model_type, lags_ms}` | `train.py · train_model` (`np.save(... 'neighborhood_config.npy' ...)`) |
| Settings JSON dump per run | `utils.py · save_settings_to_json` (called from `main.py`) |

---

## 10. File-by-file summary (quick index)

| Equation / algorithm | Module | Symbol |
|---|---|---|
| $\mathcal{N}^{(s)}_{\mathrm{atlas}}(i)$ | `neighborhood.py` | `_neighborhood_eeg_montage`, `_neighborhood_ieeg_aal` |
| $\mathcal{N}^{(s)}_{\mathrm{knn}}(i)$ | `neighborhood.py` | `_neighborhood_knn` |
| Mask + intensity sweep | `neighborhood.py`, `main.py` | `find_index_mask`, `MASK_INTENSITY_SWEEP` |
| $\widehat{x}_i^{(s)}$ (instantaneous) | `ReconModel.py` | `ReconModel.forward` |
| $\widehat{x}_i^{(s)}$ (lagged) | `model.py`, `ReconModel.py` | `LaggedReconModelConfigurable.forward`, `LaggedReconModel.forward` |
| L1 + Elastic Net objective | `train.py` | `train_model` (loss block) |
| 64/16/20 split + val-loss early stopping | `train.py` | `train_model` |
| $M_s^{\mathrm{learn}}$ assembly | `train.py` | `train_model` (the `adj_mat[i, j] = weights[0, m]` block) |
| $C_{ss'}$ | `similarity.py` | `cross_subject_pearson` |
| $\mathbf P^{(s\leftarrow s')}$ (strict 1-to-1 Hungarian) | `generate_adj_matrix_ieeg_unseen.py` | `hungarian_strict` |
| $M_{s\leftarrow s'}^{\mathrm{est}}$ | `generate_adj_matrix_ieeg_unseen.py` | `build_cross_subject_adjacency` |
| Per-electrode donor selection | `Test_new_subject_iEEG.py` | `run_cross_subject_eval` |
| DistCorr | `visualize.py` | `distance_correlation` |
| Phase-shuffle / IAAFT / Block-shuffle | `Load_preprocess_data.py` | `shuffle_phase`, `iaaft_1d`, `block_shuffle_1d` |
| Driver entry-point | `main.py` | `main` |
| All filesystem paths | `path.py` | `Paths` |
| All experiment knobs | `setting.py`, `configs/settings.yaml` | `Settings` |

---

## 11. Test mapping

| Methods item under test | Test file |
|---|---|
| Neighborhood definitions (atlas + k-NN) and mask construction | `test_neighborhood.py` |
| End-to-end intensity sweep behavior matches $\lceil m\,|\mathcal{N}|\rfloor+1$ | `verify_mask_intensity.py` |
| Cross-subject Pearson on concatenated trials | `test_similarity.py` |
| Strict 1-to-1 Hungarian assignment | `test_hungarian.py` |
| Model factory dispatch + lagged forward shape | `test_model_factory.py` |

Each test is self-contained and can be invoked directly
(`python <test_file>`) once the package's runtime dependencies are
installed.
