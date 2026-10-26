"""Explore saved participant data using production response epoching.

Tasks:
- Select the first participant in the prepared dataset.
- Load existing pre-ICA EEG and the saved ICA model.
- Recreate cleaned epochs and ERN using the production routine.
Inputs: Prepared-data record, saved EEG/ICA and current settings.
Outputs: Overwritten epoch FIF and ERN CSV; completion reports are not refreshed.
Usage: Run deliberately after individual processing; this is not a test suite.
"""

from __future__ import annotations
import mne
import pandas as pd
from constants import PathConstants, ICAPrep
from bids_correction import prepared_data
from analysis_validation import participant_paths
from ica import make_epochs


def main() -> mne.Epochs:
    """Rebuild one prepared participant's epochs using the production routine."""
    path_result = PathConstants().FOLDER_RESULT
    folder_data = prepared_data(path_result)

    if folder_data is None:
        raise FileNotFoundError("Prepare BIDS data with scripts/main.py first.")

    participant = pd.read_csv(
        folder_data / "participants.tsv",
        sep="\t"
    ).participant_id.iloc[0]
    paths = participant_paths(folder_data, path_result, participant)
    raw = mne.io.read_raw_fif(paths["preica"], preload=True)
    ica = mne.preprocessing.read_ica(paths["ica"])

    return make_epochs(
        raw,
        ica,
        ICAPrep(),
        participant,
        paths["epochs"].parent,
        path_result
    )


if __name__ == "__main__":
    main()
