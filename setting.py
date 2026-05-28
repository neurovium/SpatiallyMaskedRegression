"""
Experiment configuration loaded from ``configs/settings.yaml``.

Properties added beyond ``new/setting.py``:

    * neighborhood_method  -- 'atlas' or 'knn'
    * neighborhood_k       -- int, used by 'knn'
    * model_type           -- 'instantaneous' or 'lagged'
    * lags_ms              -- list[int], used by 'lagged'
    * run_lagged_too       -- bool; if True, the driver runs each
                              (subject, mask intensity) cell twice
                              ('instantaneous' then 'lagged')
"""

from pathlib import Path

import yaml


class Settings:
    """Settings container with strict validation in property setters."""

    def __init__(self):
        # -- Data ----------------------------------------------------------
        self.__dataset_type = None
        self.__mode = None
        self.__num_subject = None
        self.__fs = None
        self.__train_subject = None
        self.__test_subject = None

        # -- Training ------------------------------------------------------
        self.__max_epoch = None
        self.__batch_size = None
        self.__lr = None
        self.__patience = None
        self.__mask_intensity = None

        # -- Neighborhood / model selection -------------------------------
        self.__neighborhood_method = 'atlas'
        self.__neighborhood_k = 9
        self.__model_type = 'instantaneous'
        self.__lags_ms = [20, 30, 50, 60]
        self.__run_lagged_too = False

        # -- Misc ----------------------------------------------------------
        # Path to a directory of saved donor checkpoints; consumed by
        # ``evaluate_model`` in ``train.py``. Left blank by default; users
        # set it in ``configs/settings.yaml`` only if they need cross-subject
        # evaluation.
        self.__path_save_model = ''

    # --------------------------------------------------------------- IO --
    def load_settings(self):
        working_folder = Path(__file__).resolve().parents[0]
        file_path = working_folder / 'configs' / 'settings.yaml'
        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise Exception("Could not load 'configs/settings.yaml'.") from e

        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception(
                    f"'{key}' is not a valid attribute of the Settings class!"
                )

    # ----------------------------------------------------- Data --------
    @property
    def dataset_type(self): return self.__dataset_type

    @dataset_type.setter
    def dataset_type(self, value):
        valid = ('EEG', 'iEEG',
                 'Surrogate_EEG', 'Surrogate_iEEG',
                 'IAAFT_EEG', 'IAAFT_iEEG',
                 'Block_EEG', 'Block_iEEG')
        if value in valid:
            self.__dataset_type = value
        else:
            raise ValueError(f"dataset_type must be one of {valid}")

    @property
    def mode(self): return self.__mode

    @mode.setter
    def mode(self, value):
        if value in ('train', 'test_new', 'compare', 'Evaluate_dandiset'):
            self.__mode = value
        else:
            raise ValueError("mode must be 'train', 'test_new', "
                             "'compare', or 'Evaluate_dandiset'")

    @property
    def num_subject(self): return self.__num_subject

    @num_subject.setter
    def num_subject(self, value):
        if isinstance(value, int) and value > 0:
            self.__num_subject = value
        else:
            raise ValueError("num_subject must be a positive integer")

    @property
    def fs(self): return self.__fs

    @fs.setter
    def fs(self, value):
        if isinstance(value, int) and value > 0:
            self.__fs = value
        else:
            raise ValueError("fs must be a positive integer")

    @property
    def train_subject(self): return self.__train_subject

    @train_subject.setter
    def train_subject(self, value):
        if isinstance(value, int) and value >= 0:
            self.__train_subject = value
        else:
            raise ValueError("train_subject must be a non-negative integer")

    @property
    def test_subject(self): return self.__test_subject

    @test_subject.setter
    def test_subject(self, value):
        if isinstance(value, int) and value >= 0:
            self.__test_subject = value
        else:
            raise ValueError("test_subject must be a non-negative integer")

    # ---------------------------------------------------- Training -----
    @property
    def max_epoch(self): return self.__max_epoch

    @max_epoch.setter
    def max_epoch(self, value):
        if isinstance(value, int) and value > 0:
            self.__max_epoch = value
        else:
            raise ValueError("max_epoch must be a positive integer")

    @property
    def batch_size(self): return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        if isinstance(value, int) and value > 0:
            self.__batch_size = value
        else:
            raise ValueError("batch_size must be a positive integer")

    @property
    def lr(self): return self.__lr

    @lr.setter
    def lr(self, value):
        if isinstance(value, float) and value > 0:
            self.__lr = value
        else:
            raise ValueError("lr must be a positive float")

    @property
    def patience(self): return self.__patience

    @patience.setter
    def patience(self, value):
        if isinstance(value, int) and value >= 0:
            self.__patience = value
        else:
            raise ValueError("patience must be a non-negative integer")

    @property
    def mask_intensity(self): return self.__mask_intensity

    @mask_intensity.setter
    def mask_intensity(self, value):
        if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
            self.__mask_intensity = float(value)
        else:
            raise ValueError("mask_intensity must be in [0, 1]")

    # ------------------------------- Neighborhood / model selection -----
    @property
    def neighborhood_method(self): return self.__neighborhood_method

    @neighborhood_method.setter
    def neighborhood_method(self, value):
        if value in ('atlas', 'knn'):
            self.__neighborhood_method = value
        else:
            raise ValueError("neighborhood_method must be 'atlas' or 'knn'")

    @property
    def neighborhood_k(self): return self.__neighborhood_k

    @neighborhood_k.setter
    def neighborhood_k(self, value):
        if isinstance(value, int) and value > 0:
            self.__neighborhood_k = value
        else:
            raise ValueError("neighborhood_k must be a positive integer")

    @property
    def model_type(self): return self.__model_type

    @model_type.setter
    def model_type(self, value):
        if value in ('instantaneous', 'lagged'):
            self.__model_type = value
        else:
            raise ValueError("model_type must be 'instantaneous' or 'lagged'")

    @property
    def lags_ms(self): return self.__lags_ms

    @lags_ms.setter
    def lags_ms(self, value):
        if (isinstance(value, (list, tuple))
                and all(isinstance(v, int) and v >= 0 for v in value)):
            self.__lags_ms = list(value)
        else:
            raise ValueError("lags_ms must be a list of non-negative integers")

    @property
    def run_lagged_too(self): return self.__run_lagged_too

    @run_lagged_too.setter
    def run_lagged_too(self, value):
        if isinstance(value, bool):
            self.__run_lagged_too = value
        else:
            raise ValueError("run_lagged_too must be bool")

    # ---------------------------------------------------- Misc ---------
    @property
    def path_save_model(self): return self.__path_save_model

    @path_save_model.setter
    def path_save_model(self, value):
        if isinstance(value, str):
            self.__path_save_model = value
        else:
            raise ValueError("path_save_model must be a string")
