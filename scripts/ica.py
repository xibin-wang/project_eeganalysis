import pandas as pd
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from pathlib import Path

from constants import*
from preprocess import main as preprocess_local
from ica_preparation import main as ica_preparation


def run_ica(
        raw_bids: mne.io.BaseRaw,
        cfg: ICAParameters,
        p_id: str,
        path_data: Path
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA]:
    raw_ica = raw_bids.copy().load_data()
    if "boundary" in raw_ica.annotations.description:
        raw_ica.annotations.rename({"boundary": "BAD_boundary"})

    # Optional if you want to train ICA with another highpass threshold
    if cfg.if_transfer:
        raw_ica.filter(
            l_freq=cfg.highpass_train,
            h_freq=None
        )

    picks = mne.pick_types(
        raw_ica.info,
        eeg=True,
        eog=False,
        exclude="bads",
    )

    ica = ICA(
        n_components=None,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
        max_iter="auto",
    )

    ica.fit(
        raw_ica,
        picks=picks,
        reject_by_annotation=True,
    )

    ica.save(path_data/ f"{p_id}_ica.fif", overwrite=True)

    return raw_ica, ica

def ica_eval(
        raw_ica: mne.io.BaseRaw,
        ica: mne.preprocessing.ICA,
        cfg: ICAParameters,
        p_id: str,
        path_result: Path
) -> tuple[bool, mne.preprocessing.ICA]:

    data = raw_ica.get_data(reject_by_annotation="omit")
    raw_clean = mne.io.RawArray(data, raw_ica.info.copy())
    if_re_run = False

    probabilities = iclabel_label_components(
        raw_clean,
        ica,
        inplace=True,
        backend="onnx",
    )
    classes = [
        "brain",
        "muscle artifact",
        "eye blink",
        "heart beat",
        "line noise",
        "channel noise",
        "other",
    ]

    predicted = probabilities.argmax(axis=1)
    results = pd.DataFrame({
        "component": np.arange(ica.n_components_),
        "name": [f"ICA{i:03d}" for i in range(ica.n_components_)],
        "label": [classes[i] for i in predicted],
        "confidence": probabilities.max(axis=1),
        "reject":(
                (probabilities.max(axis=1) >= cfg.label_threshold)
                & ~np.isin(np.array(classes)[predicted], ["brain", "other"])
                ),
        })

    for index, label in enumerate(classes):
        results[f"p_{label.replace(' ', '_')}"] = probabilities[:, index]

    print(results[["name", "label", "confidence"]].to_string(index=False))
    n_other = int(results["label"].eq("other").sum())
    n_components = len(results)
    other_fraction = n_other / n_components
    print(f"ICLabel other: {n_other}/{n_components} ({other_fraction:.1%})")
    if other_fraction > cfg.n_other_threshold:
        if_re_run = True
        print(
            f" Components labeled as 'other' has passed {cfg.n_other_threshold:.0%}, "
            f" please consider re-run ica with more strict artifact rejection."
        )

    results.to_csv(
        path_result / p_id / f"{p_id}_iclabel.tsv",
        sep="\t",
        index=False,
    )

    ica.exclude = results.loc[
        results["reject"], "component"
    ].astype(int).tolist()

    return if_re_run, ica

def analysis_individual(filter_config, ica_config, p_id, folder_data, path_result: Path):
    path_data = folder_data / p_id / "eeg"
    path_result.joinpath(p_id).mkdir(parents=True, exist_ok=True)

    # ----------------------- Data preparation  ---------------------------
    data = preprocess_local(p_id, filter_config, folder_data)
    data_before_ica = ica_preparation(data, p_id=p_id, path_data=path_data, path_result=path_result)

    # ------------------------- Run ICA -----------------------------------
    count_ica = 1
    while True:
        data_ica, ica = run_ica(data_before_ica, ica_config, p_id=p_id, path_data=path_data)
        if_re_run, ica_exclude = ica_eval(data_ica, ica, ica_config, p_id=p_id, path_result=path_result)
        if if_re_run:
            decision = input("Would you like to decide to run the ICA? "
                             "This will bring you back to artifact detection window and rerun ica (y/n): ")
            if decision.lower() == "y":
                count_ica += 1
                data_before_ica = mne.io.read_raw_fif(path_data / f"{p_id}_task_Flanker_eeg_before_ica.fif",
                                                      preload=True)
                data_before_ica.plot(
                    duration=10.0,
                    start=0.0,
                    n_channels=30,
                    scalings="auto",
                    show=True,
                    block=True,
                    title="Manual artifact review for ICA",
                )
                data_before_ica.save(path_data / f"{p_id}_task_Flanker_eeg_before_ica.fif")

                data_ica, ica = run_ica(data_before_ica, ica_config, p_id=p_id, path_data=path_data)
                if_re_run, ica_exclude = ica_eval(data_ica, ica, ica_config, p_id=p_id,
                                                  path_result=path_result)
                ica.save(path_data / f"{p_id}_task_Flanker_eeg_ica.fif")
            elif decision.lower() == "n":
                print("Using current data to continue following analysis.")
                break
        elif ~if_re_run:
            break

    return data_before_ica, data_ica, ica

def make_epochs(data_ica, ica, ica_prep_config, p_id, path_data) -> mne.epochs.Epochs:
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
        tmin=ica_prep_config.response_window[0],
        tmax=ica_prep_config.response_window[1],
        baseline=None,
        metadata=metadata,
        reject_by_annotation=True,
        preload=True,
    )

    ica.apply(epochs)
    epochs.apply_baseline((None,0))
    epochs.save(path_data / f"{p_id}_task_Flanker_eeg_epochs.fif", overwrite=True)

    return epochs
