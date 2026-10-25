import pandas as pd
import numpy as np

import mne
import matplotlib.pyplot as plt
from ica import*

from constants import*

def make_epochs(data_ica, ica, p_id) -> mne.epochs.Epochs:
    response_idx = np.array([
        i for i, extra in enumerate(data_ica.annotations.extras)
        if extra.get("Event") == "Response"
    ], dtype=int)
    samples = np.round(
        data_ica.annotations.onset[response_idx] * data_ica.info["sfreq"]
    ).astype(int)
    events = np.column_stack([
        samples,
        np.zeros(len(samples), dtype=int),
        np.ones(len(samples), dtype=int),
    ])

    metadata = pd.DataFrame([dict(data_ica.annotations.extras[i]) for i in response_idx])
    metadata["annotation_description"] = data_ica.annotations.description[response_idx]
    metadata["annotation_index"] = response_idx

    for col in ["Trial", "Congruency", "RT", "ACC"]:
        metadata[col] = pd.to_numeric(metadata[col], errors="coerce")

    epochs = mne.Epochs(
        data_ica,
        events=events,
        event_id={"response": 1},
        tmin=ica_prep.response_window[0],
        tmax=ica_prep.response_window[1],
        baseline=None,
        metadata=metadata,
        reject_by_annotation=True,
        preload=True,
    )

    ica.apply(epochs)
    epochs.apply_baseline((None,0))
    # epochs.save(path_data / f"{p_id}_task_Flanker_eeg_epochs.fif", overwrite=True)

    # Last time epochs review before ERP analysis
    epochs.plot()
    # epochs.save(path_data / f"{p_id}_task_Flanker_eeg_epochs_reviewed.fif", overwrite=True)

    return epochs

path = PathConstants()
filter_config = FilterParameters()
ica_config = ICAParameters()
ica_prep = ICAPrep()
folder_data = path.FOLDER_ROOT / "eeglabset_fix"
participant_info = pd.read_csv(folder_data / "participants.tsv", sep="\t")
participant_id = participant_info["participant_id"][0]
path_data = folder_data / participant_id / "eeg"
path_result = path.FOLDER_ROOT / "results"
path_result.joinpath(participant_id).mkdir(parents=True, exist_ok=True)

raw_ica_test = mne.io.read_raw_fif(folder_data / participant_id / "eeg" / f"{participant_id}_task_Flanker_eeg_before_ica.fif", preload=True)
# run_ica(raw_ica, ica_config, p_id=participant_id, path_data=path_data)
ica_test = mne.preprocessing.read_ica(folder_data / participant_id / "eeg" / f"{participant_id}_ica.fif")
_, ica_test = ica_eval(raw_ica_test, ica_test, ica_config, p_id=participant_id, path_result=path_result)
make_epochs(raw_ica_test, ica_test, ica_prep)


# picks = list(range(ica.n_components_))
# if "boundary" in raw_ica.annotations.description:
#     raw_ica.annotations.rename({"boundary": "BAD_boundary"})
# raw_ica.filter(
#             l_freq=1,
#             h_freq=None
#         )
# figs = ica.plot_properties(
#     raw_ica,
#     picks=picks,
#     show=False,  # 不弹出窗口
# )
#
# for idx, fig in zip(picks, figs):
#     fig.savefig(
#         path_result / f"ICA{idx:03d}_properties.png",
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.close(fig)

print("end")