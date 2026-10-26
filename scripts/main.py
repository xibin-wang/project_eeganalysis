"""Coordinate interactive individual processing and group analysis.

Tasks:
- Reuse prepared BIDS data and load participant eligibility.
- Process selected/all participants, with parallel automatic batches.
- Validate menu input, isolate failures and support forced reruns.
- Save batch status and hand completed candidates to group analysis.

Inputs: Configurations, prepared BIDS dataset and interactive selections.

Outputs: Batch/entry reports and the individual/group outputs it schedules.

Usage: Run python scripts/main.py; importing the module starts no analysis.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import mne
import pandas as pd

from pipeline_logging import print
from analysis_validation import read_analysis_state, participant_paths
from bids_correction import main as bids_correction, prepared_data
from group_analysis import main as group_analysis
from ica import analysis_individual
from constants import PathConstants, ModeConfiguration, FilterParameters, ICAPrep, ICAParameters, GroupParameters
from typing import Any


def select_participant(participant_ids: list[str]) -> str:
    """Prompt until an existing participant ID or displayed index is selected."""
    for index, participant_id in enumerate(participant_ids):
        print(f"{index}: {participant_id}")
    while True:
        value = input("Participant ID or displayed nonnegative index: ").strip()
        if value in participant_ids:
            return value
        if value.isdigit() and 0 <= int(value) < len(participant_ids):
            return participant_ids[int(value)]
        print("Please enter an existing participant ID or a valid displayed index.")


def main() -> dict[str, Any]:
    """Coordinate prepared BIDS data, individual processing and the group-analysis attempt."""
    paths = PathConstants()
    mode = ModeConfiguration()
    path_result = Path(paths.FOLDER_RESULT)
    path_result.mkdir(parents=True, exist_ok=True)
    folder_data = None if mode.rebuild_bids else prepared_data(path_result)

    if folder_data is None:
        folder_data = bids_correction(path_result=path_result, force=mode.rebuild_bids)

    if mode.semi_auto:
        mne.set_config("MNE_BROWSER_BACKEND", "qt")

    if (
        isinstance(mode.max_workers, bool)
        or not isinstance(mode.max_workers, int)
        or mode.max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer.")

    filters, ica_config, epoch_config, group_config = FilterParameters(), ICAParameters(), ICAPrep(), GroupParameters()
    all_participants = pd.read_csv(folder_data / "participants.tsv", sep="\t")

    if (
        all_participants.participant_id.isna().any()
        or not all_participants.participant_id.is_unique
    ):
        raise ValueError("participants.tsv requires nonmissing, unique participant IDs.")

    bids_report = pd.read_csv(
        path_result / "inspect" / "bids_correction_summary.tsv",
        sep="\t"
    )
    eligible = set(bids_report.loc[
        bids_report.status.eq("ok") & bids_report.analysis_eligible.eq(True),
        "participant_id"
    ])
    participant_info = all_participants.loc[all_participants.participant_id.isin(eligible)].copy()
    participant_ids = participant_info.participant_id.tolist()
    failures = set()
    batch = {}

    for pid in all_participants.participant_id:
        batch[pid] = dict(
            participant_id=pid,
            status="pending" if pid in eligible else "bids_failed",
            error=""
        )
        if pid not in eligible:
            errors = bids_report.loc[
                bids_report.participant_id.eq(pid),
                "error"
            ].dropna().astype(str)
            batch[pid]["error"] = "; ".join(errors) or "Not eligible in cached BIDS report."
        else:
            state_path = participant_paths(folder_data, path_result, pid)["state"]
            if state_path.is_file():
                try:
                    saved = json.loads(state_path.read_text(encoding="utf-8"))
                    batch[pid].update(
                        status=saved["status"],
                        error=saved.get("error", "")
                    )
                except (OSError, ValueError, KeyError):
                    batch[pid].update(status="invalid_state")

    def save_batch() -> None:
        """Save the latest processing status and error for every participant."""
        pd.DataFrame(batch.values()).to_csv(
            path_result / "preprocessing_batch_status.tsv",
            sep="\t",
            index=False
        )

    def record(
        pid: str,
        outcome: str | None = None,
        error: Exception | None = None,
    ) -> None:
        """Record an individual outcome and refresh the batch status table."""
        if error is None:
            failures.discard(pid)
            batch[pid].update(status="completed", error="")
            print(f"{pid}: {'verified and skipped' if outcome == 'skipped' else 'completed'}.")
        else:
            failures.add(pid)
            batch[pid].update(status="failed", error=f"{type(error).__name__}: {error}")
            print(f"{pid}: analysis failed: {error}; other participants continue.")
        save_batch()

    save_batch()

    while participant_ids:
        choice = input("1. Single participant\n2. All participants\n3. Group analysis\n4. Force rerun one participant\n").strip()
        if choice == "3":
            break
        if choice not in ("1", "2", "4"):
            print("Please enter 1, 2, 3 or 4.")
            continue
        selected = participant_ids if choice == "2" else [select_participant(participant_ids)]
        force = choice == "4" or mode.force_reprocess
        if choice == "2" and not mode.semi_auto:
            with ThreadPoolExecutor(max_workers=mode.max_workers) as executor:
                futures = {
                    executor.submit(
                        analysis_individual,
                        filters,
                        ica_config,
                        pid,
                        folder_data,
                        path_result,
                        epoch_config=epoch_config,
                        force=force
                    ): pid
                    for pid in selected
                }
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        record(pid, outcome=future.result())
                    except Exception as exc:
                        record(pid, error=exc)
        else:
            for pid in selected:
                try:
                    record(
                        pid,
                        outcome=analysis_individual(
                            filters,
                            ica_config,
                            pid,
                            folder_data,
                            path_result,
                            epoch_config=epoch_config,
                            force=force
                        )
                    )
                except Exception as exc:
                    record(pid, error=exc)

    if not participant_ids:
        print("No BIDS-eligible participants; group attempt will record no inputs.")

    checks, group_ids = [], []

    for pid in participant_ids:
        entry = dict(
            participant_id=pid,
            included=False,
            ern_csv_available=False,
            reason=""
        )
        try:
            files = participant_paths(folder_data, path_result, pid)
            entry["ern_csv_available"] = files["ern"].is_file() and files["ern"].stat().st_size > 0
            read_analysis_state(files["state"], pid)
            if pid in failures:
                raise ValueError("Individual processing failed in this session.")
            group_ids.append(pid)
            entry["included"] = True
        except Exception as exc:
            entry["reason"] = str(exc)
        checks.append(entry)

    pd.DataFrame(
        checks,
        columns=["participant_id", "included", "ern_csv_available", "reason"]
    ).to_csv(path_result / "group_entry_checks.tsv", sep="\t", index=False)

    return group_analysis(
        participant_info.loc[participant_info.participant_id.isin(group_ids)].copy(),
        group_config,
        folder_data,
        path_result
    )


if __name__ == "__main__":
    main()
