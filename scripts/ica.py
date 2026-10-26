"""Perform individual ICA, response epoching and ERN extraction.

Tasks:
- Run continuous preprocessing and artifact preparation.
- Fit ICA, classify components and optionally repeat validated review.
- Apply ICA/baseline to response epochs and calculate ERN measures.

Inputs: Participant ID, data/result paths and filter/ICA/epoch settings.

Outputs: Pre-ICA EEG, ICA model, cleaned epochs, ICLabel and ERN tables.

Usage: Call analysis_individual(); supply epoch_config for final epochs/ERN.
"""

from __future__ import annotations

from pipeline_logging import print, timed_step, timed_stage, participant_log
from analysis_validation import match_bids_responses
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from pathlib import Path

from constants import*
from preprocess import main as preprocess_local
from ica_preparation import main as ica_preparation, review_for_ica
from typing import Literal


@timed_step("ICA fitting")
def run_ica(
    raw_bids: mne.io.BaseRaw,
    cfg: ICAParameters,
) -> tuple[mne.io.BaseRaw, ICA]:
    """Fit ICA on a separate training copy while preserving the analysis data."""
    raw_ica = raw_bids.copy().load_data()

    if "boundary" in raw_ica.annotations.description:
        raw_ica.annotations.rename({"boundary": "BAD_boundary"})

    # Optional if you want to train ICA with another highpass threshold
    if cfg.if_transfer:
        raw_ica.filter(
            l_freq=cfg.highpass_train,
            h_freq=None,
            skip_by_annotation=("edge", "bad", "boundary"),
        )

    picks = mne.pick_types(raw_ica.info, eeg=True, eog=False, exclude="bads")

    ica = ICA(
        n_components=None,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
        max_iter="auto",
    )

    ica.fit(raw_ica, picks=picks, reject_by_annotation=True)

    return raw_ica, ica


@timed_step("ICLabel evaluation")
def ica_eval(
    raw_ica: mne.io.BaseRaw,
    ica: ICA,
    cfg: ICAParameters,
    p_id: str,
    path_result: Path,
) -> tuple[bool, ICA]:
    """Classify ICA components, save probabilities and select components for removal."""

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
        "reject":( (probabilities.max(axis=1) >= cfg.label_threshold) & ~np.isin(
            np.array(classes)[predicted],
            ["brain", "other"]
        ) ),
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

    ica.exclude = results.loc[results["reject"], "component"].astype(int).tolist()

    return if_re_run, ica


@participant_log
def analysis_individual(
    filter_config: FilterParameters,
    ica_config: ICAParameters,
    p_id: str,
    folder_data: Path,
    path_result: Path | str,
    epoch_config: ICAPrep | None = None,
    force: bool = False,
) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw, ICA] | Literal["skipped"] | None:
    """Preprocess one participant, fit ICA and optionally export final epochs and ERN."""
    path_result = Path(path_result)
    path_data = Path(folder_data) / p_id / "eeg"
    path_result.joinpath(p_id).mkdir(parents=True, exist_ok=True)

    # ----------------------- Data preparation  ---------------------------
    data = preprocess_local(p_id, filter_config, folder_data, path_result)
    data_before_ica = ica_preparation(
        data,
        p_id=p_id,
        path_data=path_data,
        path_result=path_result
    )

    # ------------------------- Run ICA -----------------------------------
    while True:
        data_ica, ica = run_ica(
            data_before_ica,
            ica_config,
            p_id=p_id,
            path_data=path_data
        )
        if_re_run, ica_exclude = ica_eval(
            data_ica,
            ica,
            ica_config,
            p_id=p_id,
            path_result=path_result
        )
        if not if_re_run or not ModeConfiguration().semi_auto:
            break

        while True:
            decision = input("Review artifacts and rerun ICA? (y/n): ").strip().lower()
            if decision in ("y", "n"):
                break
            print("Please enter y or n.")
        if decision == "n":
            break
        data_before_ica = review_for_ica(data_before_ica, p_id, path_result)

    with timed_stage("Save pre-ICA data and ICA model"):
        data_before_ica.save(
            path_data / f"{p_id}_task_Flanker_eeg_before_ica.fif",
            overwrite=True,
        )
        ica.save(path_data / f"{p_id}_ica.fif", overwrite=True)

    if epoch_config is not None:
        make_epochs(data_before_ica, ica, epoch_config, p_id, path_data, path_result)
        return None

    return data_before_ica, data_ica, ica


@timed_step("Epoch creation and ERN")
def make_epochs(
    data_ica: mne.io.BaseRaw,
    ica: ICA,
    ica_prep_config: ICAPrep,
    p_id: str,
    path_data: Path,
    path_result: Path | str,
) -> mne.Epochs:
    """Create response epochs, apply ICA and baseline, and export individual ERN measures."""
    response_idx = np.array(
        [
            i for i,
            extra in enumerate(data_ica.annotations.extras) if str(extra.get(
                "Event",
                ""
            )).strip().casefold() == "response"
        ],
        dtype=int
    )

    if response_idx.size == 0:
        raise ValueError(f"{p_id}: no response events available for epoch creation.")

    samples = np.round(
        data_ica.annotations.onset[response_idx] * data_ica.info["sfreq"]
    ).astype(int)
    events = np.column_stack([
        samples,
        np.zeros(len(samples), dtype=int),
        np.ones(len(samples), dtype=int),
    ])

    metadata = pd.DataFrame([
        dict(data_ica.annotations.extras[i])
        for i in response_idx
    ])
    metadata["annotation_description"] = data_ica.annotations.description[response_idx]
    metadata["annotation_index"] = response_idx

    for col in ["Trial", "Congruency", "RT", "ACC"]:
        metadata[col] = pd.to_numeric(
            metadata[col],
            errors="coerce"
        ) if col in metadata else np.nan

    metadata, levels = match_bids_responses(
        metadata,
        Path(path_data) / f"{p_id}_task-Flanker_events.tsv"
    )

    with timed_stage("Epoch construction and annotation rejection"):
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

    with timed_stage("Apply ICA, baseline and save epochs"):
        ica.apply(epochs)
        epochs.apply_baseline((None,0))
        epochs.save(path_data / f"{p_id}_task_Flanker_eeg_epochs.fif", overwrite=True)

    with timed_stage("Calculate and save ERN"):
        cfg_ern = GroupParameters()
        missing_roi = set(cfg_ern.roi_channels) - set(epochs.ch_names)
        bad_roi = set(cfg_ern.roi_channels) & set(epochs.info["bads"])

        if missing_roi or bad_roi or len(epochs) == 0:
            raise ValueError(f"{p_id}: cannot calculate ERN; missing ROI={missing_roi}, bad ROI={bad_roi}, epochs={len(epochs)}")
        tmin, tmax = cfg_ern.win_analysis_s
        time_mask = (epochs.times >= tmin - 1e-9) & (epochs.times <= tmax + 1e-9)
        if (
            tmin > tmax
            or tmin < epochs.times[0] - 1e-9
            or tmax > epochs.times[-1] + 1e-9
            or not time_mask.any()
        ):
            raise ValueError(f"{p_id}: ERN window is not fully covered by epochs.")
        amplitudes = epochs.get_data(
            picks=list(cfg_ern.roi_channels),
            units="uV"
        ).mean(axis=1)[:, time_mask].mean(axis=1)

        md = epochs.metadata.reset_index(drop=True).copy()
        if (
            not md["ACC"].isin([0, 1]).all()
            or not np.isfinite(md["Congruency"]).all()
            or not np.isfinite(amplitudes).all()
        ):
            raise ValueError(f"{p_id}: invalid ACC, Congruency or ERN amplitude.")

        rows = []
        for level in [None, *levels]:
            selected = np.ones(
                len(md),
                dtype=bool
            ) if level is None else md["Congruency"].eq(level).to_numpy()
            errors = amplitudes[selected & md["ACC"].eq(0).to_numpy()]
            correct = amplitudes[selected & md["ACC"].eq(1).to_numpy()]
            ern = errors.mean() if errors.size else np.nan
            crn = correct.mean() if correct.size else np.nan
            rows.append(dict(
                participant_id=p_id,
                condition="all" if level is None else f"congruency_{level:g}",
                Congruency=np.nan if level is None else level,
                n_error=errors.size,
                n_correct=correct.size,
                mean_ern_uv=ern,
                mean_crn_uv=crn,
                delta_ern_uv=ern-crn,
                roi=",".join(cfg_ern.roi_channels),
                tmin_s=tmin,
                tmax_s=tmax
            ))

        output = Path(path_result) / p_id
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            output / f"{p_id}_ern_summary.csv",
            index=False,
            na_rep="n/a"
        )

    return epochs
