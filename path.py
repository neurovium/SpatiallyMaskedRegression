"""
Path management for the SMR pipeline.

All filesystem locations are loaded from ``configs/device_path.yaml``.
The release ships with a placeholder file at that location; users replace
the placeholders with paths on their own machine.

The yaml keys consumed by ``Paths`` are:

    data_dir            -- root of the project's data directory
    preprocessed_dir    -- directory holding preprocessed pickles
                            (defaults to ``<data_dir>/preprocessed``)
    raw_dir             -- directory holding channel-name files,
                            electrode coordinates, etc.
                            (defaults to ``<data_dir>/raw``)
    results_dir         -- directory under which all training and
                            cross-subject artifacts are written
                            (defaults to ``<package>/results``)
    donor_models_dir    -- directory of saved donor checkpoints used by
                            ``evaluate_model`` in 'test_new' mode
                            (optional; defaults to ``<results_dir>``)

All of these can be absolute or relative to the package root.
"""

import datetime
import os
from pathlib import Path

import yaml


class Paths:
    """Resolves all filesystem locations used by the pipeline."""

    def __init__(self, settings):
        self.settings = settings

        # Loaded from device_path.yaml.
        self.data_dir = None
        self.preprocessed_dir = None
        self.raw_dir = None
        self.results_dir = None
        self.donor_models_dir = None

        # Built by ``create_paths`` from the above + the current settings.
        self.base_path = None
        self.result_path = None
        self.folder_name = None

    # ---------------------------------------------------- IO ----------
    def load_device_paths(self):
        """Read ``configs/device_path.yaml`` into the attributes above."""
        working_folder = Path(__file__).resolve().parents[0]
        config_path = working_folder / 'configs' / 'device_path.yaml'

        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError as e:
            raise Exception(
                "Could not load 'configs/device_path.yaml'. Copy "
                "'configs/device_path.example.yaml' to 'configs/device_path.yaml' "
                "and edit the paths."
            ) from e

        for key in ('data_dir', 'preprocessed_dir', 'raw_dir',
                    'results_dir', 'donor_models_dir'):
            if key in cfg and cfg[key]:
                setattr(self, key, str(cfg[key]))

        # Sensible defaults relative to data_dir / package root.
        pkg_root = working_folder
        if self.data_dir is None:
            self.data_dir = str(pkg_root / 'data')
        if self.preprocessed_dir is None:
            self.preprocessed_dir = os.path.join(self.data_dir, 'preprocessed')
        if self.raw_dir is None:
            self.raw_dir = os.path.join(self.data_dir, 'raw')
        if self.results_dir is None:
            self.results_dir = str(pkg_root / 'results')
        if self.donor_models_dir is None:
            self.donor_models_dir = self.results_dir

        self.create_paths()

    # -------------------------- Result-path layout --------------------
    def create_paths(self):
        """Construct ``self.result_path`` from settings and ``results_dir``.

        Layout::

            <results_dir>/
                <dataset_type>_<mode>/
                    sub_<subject>/
                        <timestamp>/
                            nbhd-<atlas|knn>/
                                model-<instantaneous|lagged>/
                                    intesity_<percent>/
        """
        self.folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        subject = (self.settings.train_subject
                   if self.settings.mode == 'train'
                   else self.settings.test_subject)

        nbhd = getattr(self.settings, 'neighborhood_method', 'atlas')
        mtype = getattr(self.settings, 'model_type', 'instantaneous')

        rel = os.path.join(
            f'{self.settings.dataset_type}_{self.settings.mode}',
            f'sub_{subject}',
            self.folder_name,
            f'nbhd-{nbhd}',
            f'model-{mtype}',
            f'intesity_{self.settings.mask_intensity * 100}',
        )
        self.base_path = os.path.join(self.results_dir, rel) + os.sep
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        self.result_path = self.base_path

    # ------------------------------- Convenience accessors ------------
    def eeg_channel_name_file(self) -> str:
        """Path to the EEG channel-name array (numpy)."""
        return os.path.join(self.raw_dir, 'EEG', 'ch_name_eeg.npy')

    def eeg_channel_coord_file(self) -> str:
        """Path to the EEG electrode-coordinate array (numpy)."""
        return os.path.join(self.raw_dir, 'EEG', 'coordinate_eeg.npy')

    def ieeg_channel_names_file(self) -> str:
        """Path to the iEEG channel-name pickle (per-subject list)."""
        return os.path.join(self.raw_dir, 'iEEG', 'ch_names.pkl')

    def ieeg_electrode_coord_file(self) -> str:
        """Path to the iEEG electrode-coordinate pickle (per-subject)."""
        return os.path.join(self.raw_dir, 'iEEG', 'elec_coor_all_move_rest.pkl')

    def eeg_subject_file(self, sub_idx_0based: int) -> str:
        """Path to a preprocessed EEG subject pickle (1-based filename)."""
        return os.path.join(
            self.preprocessed_dir, 'EEG', f'S{sub_idx_0based + 1}.pkl'
        )

    def ieeg_subject_file(self, sub_idx_0based: int) -> str:
        """Path to a preprocessed iEEG subject pickle (1-based filename)."""
        return os.path.join(
            self.preprocessed_dir, 'iEEG',
            f'patient_{sub_idx_0based + 1}_reformat.pkl',
        )

    def cross_subject_adj_dir(self, src_sub_0based: int) -> str:
        """Output directory for CBEM-transferred adjacency matrices."""
        return os.path.join(
            self.results_dir, 'adj_mat_generated', f'sub_{src_sub_0based}'
        )

    def donor_adj_matrix_file(self, donor_sub_0based: int) -> str:
        """Path to the donor subject's intra-subject adjacency matrix."""
        return os.path.join(
            self.donor_models_dir,
            f'sub_{donor_sub_0based}', 'Regu2', 'intesity_100',
            f'adj_mat_{donor_sub_0based}.npy',
        )

    def donor_checkpoint_file(self,
                              donor_sub_0based: int,
                              channel: int) -> str:
        """Path to a donor subject's per-channel checkpoint."""
        return os.path.join(
            self.donor_models_dir,
            f'sub_{donor_sub_0based}', 'Regu', 'intesity_100',
            'best_model', f'best_model_node_{channel}.pt',
        )
