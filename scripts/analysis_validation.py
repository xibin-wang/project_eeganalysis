"""Shared paths, input fingerprints and validation for EEG result reuse.

Tasks:
- Build participant paths and save run-state JSON.
- Check completed state, source changes and preprocessing compatibility.
- Match epochs to BIDS responses and validate ERN tables.

Inputs: Data/result paths, settings, saved epochs and metadata tables.

Outputs: Validated data, file signatures and state JSON; exceptions on failure.

Usage: Import helpers from individual and group processing modules.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from constants import GroupParameters

if TYPE_CHECKING:
    import mne


def write_run_state(
    path: Path | str,
    state: dict[str, Any],
) -> None:
    """Atomically save a run status or preprocessing record as JSON."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, default=str) + "\n",
        encoding="utf-8"
    )
    temporary.replace(path)


def input_signature(
    folder_data: Path | str,
    participant_id: str,
) -> dict[str, dict[str, int | str]]:
    """Compare small preprocessing metadata by content and large EEG files by stat."""
    root = Path(folder_data)
    eeg = root / participant_id / "eeg"
    eeg_files = {
        p
        for pattern in ("*.set", "*.fdt")
        for p in eeg.glob(pattern)
        if p.is_file()
    }
    metadata = {
        p
        for pattern in ("*.tsv", "*.json")
        for p in eeg.glob(pattern)
        if p.is_file()
    }
    metadata.update(
        p
        for pattern in (
            "*channels*",
            "*electrodes*",
            "*coordsystem*",
            "*events.json",
            "*eeg.json"
        )
        for p in root.glob(pattern)
        if p.is_file()
    )
    signature = {}

    for file in sorted(eeg_files):
        stat = file.stat()
        signature[str(file.resolve())] = dict(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns
        )

    for file in sorted(metadata):
        signature[str(file.resolve())] = dict(sha256=hashlib.sha256(file.read_bytes()).hexdigest())

    return signature


def read_analysis_state(
    state_path: Path | str,
    participant_id: str,
) -> dict[str, Any]:
    """Validate the machine-readable record accompanying the compact text summary."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))

    if (
        state["participant_id"] != participant_id
        or state["status"] != "completed"
        or state["error"] != ""
    ):
        raise ValueError("No successfully completed analysis state for this participant.")

    count = state["epochs"]["n_epochs"]

    if (
        not isinstance(state["config"], dict)
        or isinstance(count, bool)
        or not isinstance(count, (int, float))
        or not np.isfinite(count)
        or count <= 0
        or count != int(count)
    ):
        raise ValueError("Missing configuration or invalid retained epoch count.")

    return state


def participant_paths(
    folder_data: Path | str,
    path_result: Path | str,
    participant_id: str,
) -> dict[str, Path]:
    """Return the standard per-participant input and output paths."""
    eeg = Path(folder_data) / participant_id / "eeg"
    result = Path(path_result) / participant_id
    return dict(
        epochs=eeg / f"{participant_id}_task_Flanker_eeg_epochs.fif",
        events=eeg / f"{participant_id}_task-Flanker_events.tsv",
        state=result / f"{participant_id}_analysis_state.json",
        summary=result / f"{participant_id}_analysis_summary.txt",
        ern=result / f"{participant_id}_ern_summary.csv",
        preica=eeg / f"{participant_id}_task_Flanker_eeg_before_ica.fif",
        ica=eeg / f"{participant_id}_ica.fif"
    )


def load_completed_epochs(
    paths: dict[str, Path],
    participant_id: str,
    expected_config: dict[str, Any],
) -> tuple[mne.BaseEpochs, pd.DataFrame, list[float]]:
    """Check completed preprocessing and return lazy epochs and BIDS-matched metadata."""
    import mne

    saved = read_analysis_state(paths["state"], participant_id)
    expected = json.loads(json.dumps(expected_config))
    keys = ("filters", "period_qc", "artifact_detection", "ica", "epochs")
    mismatches = [key for key in keys if saved["config"].get(key) != expected[key]]

    if mismatches:
        raise ValueError(f"Preprocessing configuration differs: {mismatches}; rerun individual analysis.")

    missing = [
        str(paths[key])
        for key in ("epochs", "events")
        if not paths[key].is_file()
        or paths[key].stat().st_size == 0
    ]

    if missing:
        raise ValueError(f"Missing or empty preprocessing inputs: {missing}")

    root = paths["epochs"].parents[2]
    current_inputs = input_signature(root, participant_id)
    previous_inputs = saved.get("input_files")

    if not isinstance(previous_inputs, dict):
        raise ValueError("Source input record is missing; rerun individual analysis.")

    changed = [
        name
        for name in sorted(set(previous_inputs) | set(current_inputs))
        if previous_inputs.get(name) != current_inputs.get(name)
    ]

    if changed:
        names = ", ".join(Path(name).name for name in changed)
        raise ValueError(f"Source EEG/metadata changed: {names}; rerun individual analysis.")

    epochs = mne.read_epochs(paths["epochs"], preload=False, verbose="ERROR")
    stored = saved["epochs"]

    if (
        len(epochs) != stored["n_epochs"]
        or not np.isclose(epochs.info["sfreq"], stored["sfreq_hz"])
        or not np.allclose(
            np.asarray(epochs.baseline, dtype=float),
            np.asarray(stored["baseline_s"], dtype=float)
        )
    ):
        raise ValueError("Epoch count, sampling rate or baseline differs from the saved state.")

    if (
        expected.get("mode", {}).get("semi_auto")
        and not saved["config"].get("mode", {}).get("semi_auto")
    ):
        raise ValueError("Manual review requested for previously automatic results.")

    metadata, levels = match_bids_responses(epochs.metadata, paths["events"])

    return epochs, metadata, levels


def match_bids_responses(
    metadata: pd.DataFrame | None,
    event_path: Path | str,
) -> tuple[pd.DataFrame, list[float]]:
    """Match retained epochs by Trial and preserve the exact numeric BIDS condition codes."""

    if metadata is None or not {"Trial", "ACC"}.issubset(metadata.columns):
        raise ValueError("Epoch metadata requires Trial and ACC.")

    events = pd.read_csv(event_path, sep="\t")

    if not {"Event", "Trial", "ACC", "Congruency"}.issubset(events.columns):
        raise ValueError("BIDS events require Event, Trial, ACC and Congruency.")

    responses = events.loc[
        events["Event"].astype(str).str.strip().str.casefold().eq("response"),
        ["Trial", "ACC", "Congruency"]
    ].apply(pd.to_numeric, errors="raise")

    if (
        responses.empty
        or not np.isfinite(responses.to_numpy()).all()
        or responses["Trial"].duplicated().any()
        or not responses["ACC"].isin([0, 1]).all()
    ):
        raise ValueError("BIDS response Trial IDs must be unique and metadata finite with ACC 0/1.")

    md = metadata.copy().reset_index(drop=True)
    md["Trial"] = pd.to_numeric(md["Trial"], errors="raise")
    md["ACC"] = pd.to_numeric(md["ACC"], errors="raise")
    source = responses.set_index("Trial")
    md["Congruency"] = md["Trial"].map(source["Congruency"])

    if (
        not np.isfinite(md[["Trial", "ACC", "Congruency"]].to_numpy()).all()
        or not md["ACC"].eq(md["Trial"].map(source["ACC"])).all()
    ):
        raise ValueError("Retained epochs do not match BIDS response trials/accuracy.")

    if "RT" not in md:
        md["RT"] = np.nan

    return md, sorted(responses["Congruency"].unique())


def validate_ern_summary(
    ern_df: pd.DataFrame,
    participant_id: str | None,
    group_config: GroupParameters | None = None,
    metadata: pd.DataFrame | None = None,
    levels: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Validate personal CSV or model condition rows; return standardized condition rows only."""

    if "participant" in ern_df.columns and "participant_id" not in ern_df.columns:
        required = {"participant", "Congruency", "n_error", "n_correct", "ACC_diff"}
        if ern_df.empty or not required.issubset(ern_df.columns):
            raise ValueError("Model condition table is empty or required fields are missing.")
        table = ern_df.copy()
        numeric = ["Congruency", "n_error", "n_correct", "ACC_diff"]
        table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
        if (
            table["participant"].isna().any()
            or (
                participant_id is not None
                and not table["participant"].eq(participant_id).all()
            )
            or table.duplicated(["participant", "Congruency"]).any()
            or not np.isfinite(table["Congruency"]).all()
            or (levels is not None and set(table["Congruency"]) != set(levels))
        ):
            raise ValueError("Invalid participant IDs or duplicate/unexpected model conditions.")
        counts = table[["n_error", "n_correct"]].to_numpy(dtype=float)
        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
            or (counts != np.floor(counts)).any()
        ):
            raise ValueError("ERN trial counts are missing or invalid.")
        return table.reset_index(drop=True)

    table = ern_df.rename(columns={
        "participant": "participant_id",
        "ACC_diff": "delta_ern_uv"
    }).copy()
    required = {"participant_id", "Congruency", "n_error", "n_correct", "delta_ern_uv"}
    required.update(("condition", "mean_ern_uv", "mean_crn_uv"))

    if group_config is not None:
        required.update(("roi", "tmin_s", "tmax_s"))

    if table.empty or not required.issubset(table.columns):
        raise ValueError("ERN table is empty or required fields are missing.")

    if (
        table["participant_id"].isna().any()
        or (
            participant_id is not None
            and not table["participant_id"].eq(participant_id).all()
        )
    ):
        raise ValueError("ERN table participant ID mismatch.")

    numeric = [
        column
        for column in (
            "Congruency",
            "n_error",
            "n_correct",
            "mean_ern_uv",
            "mean_crn_uv",
            "delta_ern_uv"
        )
        if column in table
    ]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
    overall = table.loc[table["condition"].eq("all")]
    conditions = table.loc[~table["condition"].eq("all")].copy()

    if len(overall) != 1 or conditions.empty or table["condition"].isna().any():
        raise ValueError("Personal ERN table requires one overall row and condition rows.")

    if (
        conditions.duplicated(["participant_id", "Congruency"]).any()
        or not np.isfinite(conditions["Congruency"]).all()
    ):
        raise ValueError("Duplicate participant-condition summaries or invalid BIDS condition codes.")

    if levels is not None and set(conditions["Congruency"]) != set(levels):
        raise ValueError("CSV condition codes differ from expected/BIDS response codes.")

    if group_config is not None:
        if (
            not table["roi"].astype(str).map(lambda value: sorted(value.split(",")) == sorted(group_config.roi_channels)).all()
            or not np.allclose(
                pd.to_numeric(table["tmin_s"], errors="raise"),
                group_config.win_analysis_s[0],
                atol=1e-9,
                rtol=0
            )
            or not np.allclose(
                pd.to_numeric(table["tmax_s"], errors="raise"),
                group_config.win_analysis_s[1],
                atol=1e-9,
                rtol=0
            )
        ):
            raise ValueError("CSV ROI or ERN window differs from current group settings.")

    counts = table[["n_error", "n_correct"]].to_numpy(dtype=float)

    if (
        not np.isfinite(counts).all()
        or (counts < 0).any()
        or (counts != np.floor(counts)).any()
    ):
        raise ValueError("ERN trial counts are missing or invalid.")

    for count, mean in (("n_error", "mean_ern_uv"), ("n_correct", "mean_crn_uv")):
        observed = table[count].gt(0)
        if (
            not np.isfinite(table.loc[observed, mean]).all()
            or not table.loc[~observed, mean].isna().all()
        ):
            raise ValueError("ERN/CRN amplitude is missing despite available trials, or present without trials.")

    both = table.n_error.gt(0) & table.n_correct.gt(0)

    if (
        not np.isfinite(table.loc[both, "delta_ern_uv"]).all()
        or not table.loc[~both, "delta_ern_uv"].isna().all()
        or not np.allclose(
            table.loc[both, "delta_ern_uv"],
            table.loc[both, "mean_ern_uv"] - table.loc[both, "mean_crn_uv"]
        )
    ):
        raise ValueError("Delta ERN values are missing or inconsistent.")

    if (
        not overall.empty
        and not np.array_equal(
            overall[["n_error", "n_correct"]].iloc[0].to_numpy(),
            conditions[["n_error", "n_correct"]].sum().to_numpy()
        )
    ):
        raise ValueError("Overall trial counts differ from the sum of condition counts.")

    if metadata is not None:
        expected_levels = sorted(conditions.Congruency.unique()) if levels is None else levels
        expected_counts = metadata.groupby([ "Congruency", "ACC" ]).size().unstack(
            "ACC",
            fill_value=0
        ).reindex(index=expected_levels, columns=[0, 1], fill_value=0)
        if ((not np.array_equal(
            conditions.set_index("Congruency").loc[
                expected_levels,
                ["n_error", "n_correct"]
            ].to_numpy(),
            expected_counts.to_numpy()
        ))):
            raise ValueError("Cached counts differ from retained BIDS-matched trials.")

    return conditions.rename(columns={
        "participant_id": "participant",
        "delta_ern_uv": "ACC_diff"
    }).reset_index(drop=True)
