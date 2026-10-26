"""Prepare BIDS metadata and record the active dataset for subsequent reuse.

Tasks:
- Reuse completed preparation or explicitly rebuild from source.
- Correct supported channel/electrode and event-metadata problems.
- Validate each participant and record eligibility and errors.
Inputs: Source BIDS dataset, result root and optional force flag.
Outputs: Active data path, correction report and preparation-state JSON.
Usage: Called by main.py; run directly for preparation only.
"""

from __future__ import annotations

from pipeline_logging import print
from pathlib import Path
import json
import shutil

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from constants import PathConstants
from analysis_validation import write_run_state


def event_correction(
    folder: Path,
    participant_id: str,
    repair: bool,
) -> tuple[bool, bool, bool]:
    """Validate SET/BIDS events and repair missing durations when requested."""

    # Validate every event one-to-one, including duplicate markers.
    prefix = folder / participant_id / "eeg" / f"{participant_id}_task-Flanker"
    event_path = Path(f"{prefix}_events.tsv")
    set_path = Path(f"{prefix}_eeg.set")
    events = pd.read_csv(event_path, sep="\t", dtype={"value": str})

    if not {"onset", "value"}.issubset(events.columns):
        raise ValueError(f"{participant_id}: events.tsv requires onset and value.")

    raw_set = mne.io.read_raw_eeglab(set_path, preload=False, verbose=False)

    # Raw clips out-of-range boundaries; retain original timing and duration here.
    original = mne.read_annotations(set_path).to_data_frame(time_format=None)
    original["code"] = original["description"].str.strip().str.casefold().str.removeprefix("x")
    events["onset"] = pd.to_numeric(events["onset"], errors="raise")
    codes = events["value"].str.strip().str.casefold().str.removeprefix("x")

    if codes.isna().any():
        raise ValueError(f"{participant_id}: missing event code.")

    skipped_boundary = False

    if len(original) == len(events) + 1 and not original.empty:
        first = original.iloc[0]
        if first["code"] == "boundary" and first["onset"] < 0:
            original = original.iloc[1:].copy()
            skipped_boundary = True

    if len(original) != len(events):
        raise ValueError(f"{participant_id}: SET/TSV event counts differ.")

    order_set = original.sort_values(["onset", "code"], kind="stable").index
    expected = original.loc[order_set]
    duration = pd.to_numeric(
        events["duration"],
        errors="raise"
    ) if "duration" in events else pd.Series(np.nan, index=events.index)
    missing = duration.isna()
    tolerance = max(2e-6, 1e-6 / raw_set.info["sfreq"])
    onset_fixed = False

    # Accept existing seconds first; sample/sfreq is used only during a required repair.
    candidates = [events["onset"]]

    if repair and missing.any() and "sample" in events:
        candidates.append(pd.to_numeric(
            events["sample"],
            errors="raise"
        ) / raw_set.info["sfreq"])

    for candidate_number, onsets in enumerate(candidates):
        aligned = pd.DataFrame({
            "onset": onsets,
            "code": codes
        }).sort_values(["onset", "code"], kind="stable")
        if (
            aligned["code"].tolist() == expected["code"].tolist()
            and np.allclose(aligned["onset"], expected["onset"], atol=tolerance, rtol=0)
        ):
            onset_fixed = candidate_number > 0
            break
    else:
        raise ValueError(f"{participant_id}: SET/TSV event timing or codes differ.")

    expected_duration = pd.Series(
        expected["duration"].to_numpy(),
        index=aligned.index
    ).reindex(events.index)

    if not np.isfinite(expected_duration).all() or (expected_duration < 0).any():
        raise ValueError(f"{participant_id}: SET contains unknown or negative durations.")

    if ((not np.allclose(
        duration[~missing],
        expected_duration[~missing],
        atol=tolerance,
        rtol=0
    ))):
        raise ValueError(f"{participant_id}: existing durations disagree with SET.")

    if missing.any() and not repair:
        raise ValueError(f"{participant_id}: missing durations require correction.")

    if missing.any():
        events["onset"] = onsets
        events["duration"] = duration.fillna(expected_duration)
        events.to_csv(event_path, sep="\t", index=False, na_rep="n/a")
        json_path = Path(f"{prefix}_events.json")
        if json_path.exists():
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            metadata.setdefault("onset", {})["Units"] = "second"
            metadata.setdefault("duration", {})["Units"] = "second"
            json_path.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8"
            )

    bids_path = BIDSPath(
        subject=participant_id.removeprefix("sub-"),
        task="Flanker",
        datatype="eeg",
        root=folder,
        extension=".set"
    )
    raw_bids = read_raw_bids(bids_path, extra_params={"preload": False}, verbose=False)

    # Compare Raw annotations as well, after MNE applies the same range clipping.
    annotations_set = raw_set.annotations.to_data_frame(time_format=None)

    if skipped_boundary:
        annotations_set = annotations_set.iloc[1:]

    annotations_bids = raw_bids.annotations.to_data_frame(time_format=None)

    for frame in (annotations_set, annotations_bids):
        frame["code"] = frame["description"].str.strip().str.casefold().str.removeprefix("x")

    annotations_set = annotations_set.sort_values(
        ["onset", "code", "duration"],
        kind="stable"
    )
    annotations_bids = annotations_bids.sort_values(
        ["onset", "code", "duration"],
        kind="stable"
    )

    if (
        len(annotations_set) != len(annotations_bids)
        or annotations_set["code"].tolist() != annotations_bids["code"].tolist()
    ):
        raise ValueError(f"{participant_id}: read_raw_bids/EEGLAB event codes or counts differ.")

    if ((not np.allclose(
        annotations_set[["onset", "duration"]],
        annotations_bids[["onset", "duration"]],
        atol=tolerance,
        rtol=0
    ))):
        raise ValueError(f"{participant_id}: read_raw_bids/EEGLAB timing or duration differs.")

    return bool(missing.any()), onset_fixed, skipped_boundary


def prepared_data(path_result: Path | str) -> Path | None:
    """Reuse a completed preparation without scanning, copying or correcting BIDS again."""
    marker = Path(path_result) / "inspect" / "bids_preparation_state.json"

    if not marker.is_file():
        # Recognize a completed preparation from earlier code before the marker existed.
        root = PathConstants().FOLDER_ROOT
        candidates = []
        for folder in (root / "eeglabset_fix", root / "eeglabset"):
            report = folder / "results" / "inspect" / "bids_correction_summary.tsv"
            if report.is_file() and (folder / "participants.tsv").is_file():
                frame = pd.read_csv(report, sep="\t")
                ids = pd.read_csv(
                    folder / "participants.tsv",
                    sep="\t"
                )["participant_id"].dropna()
                if (
                    {
                        "participant_id",
                        "status",
                        "analysis_eligible",
                        "error"
                    }.issubset(frame)
                    and frame.participant_id.is_unique
                    and set(frame.participant_id) == set(ids)
                ):
                    candidates.append((report.stat().st_mtime_ns, folder, report))
        if not candidates:
            return None
        _, folder, old_report = max(candidates, key=lambda item: item[0])
        marker.parent.mkdir(parents=True, exist_ok=True)
        report = marker.parent / "bids_correction_summary.tsv"
        if old_report.resolve() != report.resolve():
            shutil.copy2(old_report, report)
        write_run_state(
            marker,
            dict(status="completed", folder_data=str(folder.resolve()))
        )

    state = json.loads(marker.read_text(encoding="utf-8"))

    if state.get("status") != "completed":
        return None

    folder = Path(state["folder_data"])
    report = Path(path_result) / "inspect" / "bids_correction_summary.tsv"

    if (
        not folder.is_dir()
        or not (folder / "participants.tsv").is_file()
        or not report.is_file()
    ):
        raise FileNotFoundError("Prepared BIDS directory/report is missing; restore it or set rebuild_bids=True.")

    print(f"Reusing prepared BIDS data: {folder}")

    return folder


def main(
    path_result: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Prepare BIDS once or explicitly rebuild it, returning the active dataset root."""

    path = PathConstants()
    path_result = Path(path_result) if path_result is not None else path.FOLDER_RESULT

    if not force:
        cached = prepared_data(path_result)
        if cached is not None:
            return cached

    output = path_result / "inspect"
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "bids_preparation_state.json"
    write_run_state(marker, dict(status="running"))
    source = path.FOLDER_ROOT / "eeglabset"
    target = path.FOLDER_ROOT / "eeglabset_fix"
    channels = pd.read_csv(source / "channels.tsv", sep="\t")
    electrodes = pd.read_csv(source / "electrodes.tsv", sep="\t")
    channel_columns = {"name", "type", "units"}
    electrode_columns = {"name", "x", "y", "z"}

    swap = electrode_columns.issubset(channels.columns) and channel_columns.issubset(electrodes.columns)

    if (
        not swap
        and not (
            channel_columns.issubset(channels.columns)
            and electrode_columns.issubset(electrodes.columns)
        )
    ):
        raise ValueError("Cannot identify channels.tsv/electrodes.tsv schemas; no correction performed.")

    participants = pd.read_csv(source / "participants.tsv", sep="\t")["participant_id"]

    if participants.empty or participants.isna().any() or not participants.is_unique:
        raise ValueError("participants.tsv requires nonmissing, unique participant IDs.")

    participants = participants.to_numpy()

    needs_duration = {}
    scan_errors = {}

    for participant_id in participants:
        event_path = source / participant_id / "eeg" / f"{participant_id}_task-Flanker_events.tsv"
        try:
            events = pd.read_csv(event_path, sep="\t")
            needs_duration[participant_id] = "duration" not in events or pd.to_numeric(
                events["duration"],
                errors="raise"
            ).isna().any()
        except Exception as exc:
            needs_duration[participant_id] = False
            scan_errors[participant_id] = str(exc)

    folder = source

    if swap or any(needs_duration.values()):
        shutil.copytree(source, target, dirs_exist_ok=True)
        folder = target
        if swap:
            channel_bytes = (target / "channels.tsv").read_bytes()
            (target / "channels.tsv").write_bytes((target / "electrodes.tsv").read_bytes())
            (target / "electrodes.tsv").write_bytes(channel_bytes)

    summary = []

    for participant_id in participants:
        row = dict(
            participant_id=participant_id,
            files_swapped=swap,
            duration_repaired=False,
            onset_repaired=False,
            negative_boundary_skipped=False,
            status="failed",
            analysis_eligible=False,
            error=""
        )
        prefix = folder / participant_id / "eeg" / f"{participant_id}_task-Flanker"
        backups = {}
        try:
            if participant_id in scan_errors:
                raise ValueError(scan_errors[participant_id])
            backups = {
                p: p.read_bytes()
                for p in (Path(f"{prefix}_events.tsv"), Path(f"{prefix}_events.json"))
                if p.exists()
            }
            repaired, onset_fixed, skipped = event_correction(
                folder,
                participant_id,
                repair=bool(needs_duration[participant_id])
            )
            row.update(
                duration_repaired=repaired,
                onset_repaired=onset_fixed,
                negative_boundary_skipped=skipped,
                status="ok",
                analysis_eligible=True
            )
        except Exception as exc:
            if needs_duration[participant_id]:
                for file_path, contents in backups.items():
                    try:
                        file_path.write_bytes(contents)
                    except OSError as restore_error:
                        row["error"] += f"Restore failed for {file_path.name}: {restore_error}; "
            row["error"] += str(exc)
        summary.append(row)
        pd.DataFrame(summary).to_csv(
            output / "bids_correction_summary.tsv",
            sep="\t",
            index=False,
            na_rep="n/a"
        )

    summary = pd.DataFrame(summary)
    summary.to_csv(
        output / "bids_correction_summary.tsv",
        sep="\t",
        index=False,
        na_rep="n/a"
    )
    print(summary.to_string(index=False))
    print(f"BIDS validation: {int(summary['analysis_eligible'].sum())}/{len(summary)} participants eligible; failed participants are excluded. Report: {output / 'bids_correction_summary.tsv'}")

    write_run_state(
        marker,
        dict(
            status="completed",
            folder_data=str(folder.resolve()),
            n_participants=len(summary)
        )
    )

    return folder


if __name__ == "__main__":
    main()
