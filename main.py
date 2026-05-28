"""
Driver script for the SMR pipeline with the unified neighborhood
construction (``fixed/neighborhood.py``) and the corrected mask-intensity
sweep.

Differences vs. ``new/main.py``:
  1. The training sweep iterates over ``[0.0, 0.25, 0.5, 0.75, 1.0]``,
     so the 100% condition reported in Fig. 4 of the manuscript actually
     runs. The previous ``[0, 0.25, 0.5, 0.75]`` sweep silently skipped it.
  2. The neighborhood method (``'atlas'`` or ``'knn'``) is honoured for the
     full sweep regardless of which one is selected in
     ``configs/settings.yaml``; ``find_index_mask`` samples
     ``round(|N(i)| * m)`` neighbors at every intensity ``m``, so the
     intensity knob behaves the same way under either neighborhood.
"""

from train import train_model, evaluate_model, train_model_dandiset
from setting import Settings
from path import Paths
from utils import save_settings_to_json, extract_data_dandi


# Mask intensities described in Section "Analysis of the Effect of Local
# Information" of the manuscript. The 1.0 endpoint was missing from
# `new/main.py`; including it here closes that gap.
MASK_INTENSITY_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    settings = Settings()
    settings.load_settings()

    if settings.mode == 'test_new':
        # Cross-subject evaluation: a single intensity (full local mask).
        for i in range(settings.num_subject):
            settings.test_subject = i
            settings.mask_intensity = 1.0
            paths = Paths(settings)
            paths.load_device_paths()
            save_settings_to_json(settings, file_path=paths.result_path)
            evaluate_model(settings, paths)

    elif settings.mode == 'Evaluate_dandiset':
        settings.train_subject = 0
        settings.mask_intensity = 1.0
        paths = Paths(settings)
        paths.load_device_paths()
        save_settings_to_json(settings, file_path=paths.result_path)
        data = extract_data_dandi()
        train_model_dandiset(settings, paths, data)

    else:
        # Standard training: iterate over all subjects and all five
        # mask intensities, under whichever neighborhood method is set in
        # settings.yaml (default: 'atlas').
        #
        # If `settings.run_lagged_too: true`, the loop runs each
        # (subject, intensity) cell a second time with
        # `settings.model_type = 'lagged'`, so the *Lagged* column of
        # tab:ieeg / tab:eeg is produced by the same driver as the
        # *Intra-Subject* column.
        baseline_model_type = getattr(settings, 'model_type', 'instantaneous')
        run_lagged_too = bool(getattr(settings, 'run_lagged_too', False))

        for i in range(settings.num_subject):
            settings.train_subject = i
            for mask_intensity in MASK_INTENSITY_SWEEP:
                settings.mask_intensity = mask_intensity

                model_types = [baseline_model_type]
                if run_lagged_too and baseline_model_type != 'lagged':
                    model_types.append('lagged')

                for model_type in model_types:
                    settings.model_type = model_type
                    paths = Paths(settings)
                    paths.load_device_paths()
                    save_settings_to_json(settings, file_path=paths.result_path)
                    train_model(settings, paths)

        # Restore the user-configured value so a subsequent call to main()
        # in the same process doesn't see the side-effect.
        settings.model_type = baseline_model_type


if __name__ == '__main__':
    main()
