"""Manage participant logging, timing, reports and processing state.

Tasks:
- Add timestamps and isolate participant logs during parallel processing.
- Record stage durations, errors and compact MNE result statistics.
- Validate reusable outputs or track a new running/completed/failed attempt.
- Generate the preprocessing summary and verify inputs remain unchanged.

Inputs: Decorated processing calls, settings, paths and returned MNE objects.

Outputs: Per-participant log, timing TSV, summary TXT and state JSON.

Usage: Import print(), timed_stage(), timed_step() and participant_log().
"""

from __future__ import annotations

from analysis_validation import participant_paths, load_completed_epochs, validate_ern_summary, input_signature, write_run_state

import builtins
import csv
import functools
import inspect
import json
import os
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any, Literal, ParamSpec, TextIO, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_state = ContextVar("participant_analysis_log", default=None)
_console_lock = threading.RLock()


def print(
    *args: object,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Print timestamped messages and mirror participant messages into their isolated log."""
    state = _state.get()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    prefix = f"[{stamp}]" + (f" [{state['participant_id']}]" if state else "")
    message = sep.join(str(arg) for arg in args)
    message = "\n".join(f"{prefix} {line}" for line in message.split("\n"))

    with _console_lock:
        builtins.print(message, end=end, file=file, flush=flush)

    if state is not None and (file is None or file is sys.stdout or file is sys.stderr):
        state["log"].write(message + end)
        state["log"].flush()


@contextmanager
def timed_stage(name: str) -> Iterator[None]:
    """Measure a processing stage and record its duration and failure status."""
    state = _state.get()

    if state is None:
        yield
        return

    started = time.time()
    clock = time.perf_counter()
    status = "ok"
    print(f"START {name}")

    try:
        yield
    except BaseException:
        state.setdefault("failed_stage", name)
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - clock
        state["timings"].append(dict(
            stage=name,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_sec=elapsed,
            status=status
        ))
        print(f"END {name}: {elapsed:.3f} s ({status})")


def _snapshot(value: object) -> dict[str, Any] | list[Any] | bool | None:
    """Extract compact report statistics from MNE results without retaining EEG arrays."""
    import mne

    if isinstance(value, mne.io.BaseRaw):
        intervals = sorted(
            (
                max(0., float(a["onset"] - value.first_time)),
                min(
                    value.n_times / value.info["sfreq"],
                    float(a["onset"] - value.first_time + a["duration"])
                )
            )
            for a in value.annotations
            if str(a["description"]).casefold().startswith("bad")
        )
        merged = []
        for start, stop in intervals:
            if stop <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(stop, merged[-1][1])
            else:
                merged.append([start, stop])
        return dict(
            duration_sec=value.n_times/value.info["sfreq"],
            sfreq_hz=value.info["sfreq"],
            n_channels=len(value.ch_names),
            channels=value.ch_names,
            bad_channels=list(value.info["bads"]),
            bad_duration_sec=sum(b-a for a,b in merged),
            pending_review_intervals=sum(
                str(d).casefold().startswith("review")
                for d in value.annotations.description
            )
        )

    if isinstance(value, mne.BaseEpochs):
        return dict(
            n_epochs=len(value),
            n_candidates=len(value.drop_log),
            n_dropped=sum(bool(reason) for reason in value.drop_log),
            drop_reasons=dict(Counter(
                reason
                for reasons in value.drop_log
                for reason in reasons
            )),
            sfreq_hz=value.info["sfreq"],
            n_channels=len(value.ch_names),
            n_times=len(value.times),
            epoch_tmin_s=value.tmin,
            epoch_tmax_s=value.tmax,
            baseline_s=value.baseline,
            bad_channels=list(value.info["bads"])
        )

    if isinstance(value, mne.preprocessing.ICA):
        return dict(
            method=value.method,
            n_components=int(value.n_components_),
            n_iter=int(value.n_iter_),
            excluded_components=list(value.exclude),
            channels=value.ch_names
        )

    if isinstance(value, tuple):
        return [
            _snapshot(item)
            for item in value
            if isinstance(item, (mne.io.BaseRaw, mne.BaseEpochs, mne.preprocessing.ICA))
        ]

    if isinstance(value, bool):
        return value

    return None


def timed_step(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a processing function with timing and compact result snapshots."""
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        """Wrap a processing function while preserving its parameter and return types."""
        @functools.wraps(func)
        def wrapped(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            """Run the function, record elapsed time and capture its report snapshot."""

            with timed_stage(name):
                result = func(*args, **kwargs)

            state = _state.get()

            if state is not None:
                snapshot = _snapshot(result)
                if snapshot is not None:
                    state["snapshots"][name] = snapshot

            return result
        return wrapped
    return decorate


def _write_summary(
    path: Path,
    record: dict[str, Any],
    snapshots: dict[str, Any],
) -> None:
    """Write a standalone preprocessing report without embedding detailed output files."""
    import pandas as pd

    pid, cfg = record["participant_id"], record["config"]
    initial = snapshots.get("Read BIDS", {})
    fitting = snapshots.get("ICA fitting", [])
    final_raw = next(
        (item for item in fitting if "duration_sec" in item),
        snapshots.get( "Artifact preparation", snapshots.get("Preprocessing", initial) )
    )
    evaluated = snapshots.get("ICLabel evaluation", fitting)
    ica = next((item for item in evaluated if "n_components" in item), None)
    epochs = record["epochs"]
    eeg_folder = Path(record["eeg_folder"])

    try:
        eeg_location = Path(os.path.relpath(eeg_folder, path.parent)).as_posix()
    except ValueError:
        eeg_location = eeg_folder.as_posix()

    lines = [
        f"Participant: {pid}",
        f"Status: {record['status']}",
        f"Error: {record['error'] or 'none'}",
        f"Failed stage: {record['failed_stage'] or 'none'}",
        f"Started (UTC): {record['started_utc']}",
        f"Ended (UTC): {record['ended_utc']}",
        f"Total elapsed seconds: {record['duration_sec']:.3f}",
        "Software: " + "; ".join(
            f"{key}: {value}" for key,
            value in record["software"].items()
        ),
        "",
        "PREPROCESSING SETTINGS",
        f"Bandpass (Hz): {cfg['filters']['highpass_freq']} to {cfg['filters']['lowpass_freq']}",
        f"Line noise: {cfg['filters']['line_freq']} Hz; detected: {snapshots.get('Line-noise detection', 'not evaluated')}",
        "Reference: average EEG, excluding MASTl, MASTr, VOGbelow and marked bad channels.",
        f"ICA training highpass (Hz): {cfg['ica']['highpass_train'] if cfg['ica']['if_transfer'] else cfg['filters']['highpass_freq']}",
        f"ICLabel rejection: non-brain/non-other class with confidence >= {cfg['ica']['label_threshold']}",
        f"Manual artifact review completed: {'Artifact review or automatic pass-through' in snapshots and cfg['mode']['semi_auto']}"
    ]

    if initial:
        lines.extend([
            "",
            "DATA AND CHANNELS",
            f"Input: {initial['duration_sec']:.3f} s; {initial['sfreq_hz']:g} Hz; {initial['n_channels']} channels"
        ])

    if final_raw:
        removed = sorted(set(initial.get("channels", [])) - set(final_raw["channels"]))
        bads = final_raw["bad_channels"]
        restored = sorted(set(final_raw["channels"]) - set(initial.get("channels", [])))
        bad_seconds = final_raw["bad_duration_sec"]
        lines.extend([
            f"Removed channels: {len(removed)}; {', '.join(removed) or 'none'}",
            "Interpolated channels: 0 (not performed)",
            f"Marked bad channels: {len(bads)}; {', '.join(bads) or 'none'} (excluded from ICA/reference)",
            f"Added reference channels: {', '.join(restored) or 'none'}",
            f"BAD annotation duration: {bad_seconds:.3f} s ({100 * bad_seconds / final_raw['duration_sec']:.2f}%)",
            f"Pending REVIEW intervals: {final_raw['pending_review_intervals']}"
        ])

    lines.extend(["", "ICA"])

    if ica:
        excluded = ica["excluded_components"]
        lines.extend([
            f"Method: {ica['method']}; fitted components: {ica['n_components']}",
            f"Removed components: {len(excluded)}; indices: {excluded}"
        ])
    else:
        lines.append("ICA results unavailable for this run.")

    lines.extend(["", "EPOCHS AND ERN"])

    if epochs:
        counts = pd.read_csv(path.parent / f"{pid}_ern_summary.csv")
        lines.extend([
            f"Epoch window (s): {epochs['epoch_tmin_s']} to {epochs['epoch_tmax_s']}; baseline (s): {epochs['baseline_s']}",
            f"Candidate epochs: {epochs['n_candidates']}; retained: {epochs['n_epochs']}; dropped: {epochs['n_dropped']}",
            f"Drop reasons: {epochs['drop_reasons']}",
            counts[[ "condition", "n_error", "n_correct" ]].to_string(index=False)
        ])
    else:
        lines.append("Final epochs/ERN unavailable for this run.")

    lines.extend([
        f"ERN ROI: {', '.join(cfg['ern']['roi_channels'])}; measurement window (s): {cfg['ern']['win_analysis_s']}",
        "Delta ERN: incorrect minus correct, in microvolts; overall pools retained trials.",
        "A condition with no incorrect or correct trials has an unavailable Delta ERN.",
        "",
        "OUTPUTS (relative to this summary)",
        f"ERN values: {pid}_ern_summary.csv",
        f"State/configuration: {pid}_analysis_state.json",
        f"Log: {pid}_analysis.log; stage timings: {pid}_step_timings.tsv",
        f"EEG: {eeg_location}/{pid}_task_Flanker_eeg_before_ica.fif",
        f"ICA: {eeg_location}/{pid}_ica.fif",
        f"Final epochs: {eeg_location}/{pid}_task_Flanker_eeg_epochs.fif",
        "Output paths identify expected files; failed runs may leave incomplete/older outputs."
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def participant_log(func: Callable[P, R]) -> Callable[P, R | Literal["skipped"]]:
    """Manage completed-result reuse, isolated logging and state for one participant."""
    @functools.wraps(func)
    def wrapped(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R | Literal["skipped"]:
        """Validate reusable outputs or run one participant with persistent status and reports."""

        import mne
        import numpy as np
        import pandas as pd
        from constants import ModeConfiguration, PeriodQCConfig, ICAPrep, GroupParameters

        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        pid = values["p_id"]
        folder = Path(values["path_result"]) / pid
        paths = participant_paths(values["folder_data"], values["path_result"], pid)
        summary_path, state_path = paths["summary"], paths["state"]
        configs = dict(
            mode=asdict(ModeConfiguration()),
            filters=asdict(values["filter_config"]),
            period_qc=asdict(PeriodQCConfig()),
            artifact_detection=asdict(ICAPrep()),
            ica=asdict(values["ica_config"]),
            epochs=asdict(values["epoch_config"]) if values["epoch_config"] is not None else None,
            ern=asdict(GroupParameters())
        )
        configs = json.loads(json.dumps(configs))

        if (
            state_path.is_file()
            and not ModeConfiguration().force_reprocess
            and not values.get("force", False)
        ):
            try:
                epochs, metadata, levels = load_completed_epochs(paths, pid, configs)
                missing = [
                    str(paths[key])
                    for key in ("preica", "ica", "ern", "summary")
                    if not paths[key].is_file()
                    or paths[key].stat().st_size == 0
                ]
                if missing:
                    raise ValueError(f"Missing or empty individual outputs: {missing}")
                ern = pd.read_csv(paths["ern"])
                conditions = validate_ern_summary(
                    ern,
                    pid,
                    GroupParameters(),
                    metadata=metadata,
                    levels=levels
                )
                if ((set(ern["condition"]) != {
                    "all",
                    *(f"congruency_{level:g}" for level in levels)
                })):
                    raise ValueError("ERN table condition labels are incomplete.")
                if int(conditions[["n_error", "n_correct"]].sum().sum()) != len(epochs):
                    raise ValueError("State epoch counts and ERN trial counts disagree.")
            except Exception as exc:
                print(f"{pid}: incomplete or failed state ({exc}); restarting all analysis steps.")
            else:
                print(f"{pid}: successful state and complete outputs verified; skipping individual analysis.")
                return "skipped"

        folder.mkdir(parents=True, exist_ok=True)
        started = time.time()
        clock = time.perf_counter()
        timing_path = folder / f"{pid}_step_timings.tsv"
        log_path = folder / f"{pid}_analysis.log"

        record = dict(
            participant_id=pid,
            status="running",
            error="",
            failed_stage="",
            started_utc=time.strftime( "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started) ),
            config=configs,
            epochs={},
            input_files=input_signature( values["folder_data"], pid ),
            eeg_folder=str(paths["epochs"].parent.resolve()),
            software=dict(
                Python=sys.version.split()[0],
                MNE=mne.__version__,
                NumPy=np.__version__,
                pandas=pd.__version__
            )
        )
        write_run_state(state_path, record)

        with log_path.open("w", encoding="utf-8") as stream:

            state = dict(participant_id=pid, log=stream, timings=[], snapshots={})
            token = _state.set(state)
            status = "failed"
            error = ""

            try:
                print("START individual analysis")
                print("No valid completed state; restarting all steps from BIDS input and overwriting previous step outputs.")
                result = func(*args, **kwargs)
                if input_signature(values["folder_data"], pid) != record["input_files"]:
                    raise RuntimeError("Source EEG/metadata changed during processing; rerun this participant.")
                status = "completed" if values["epoch_config"] is not None else "completed_before_epochs"
                return result
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(traceback.format_exc())
                raise
            finally:
                try:
                    elapsed = time.perf_counter() - clock
                    print(f"END individual analysis: {elapsed:.3f} s ({status})")
                    with timing_path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=[
                                "stage",
                                "started_utc",
                                "ended_utc",
                                "duration_sec",
                                "status"
                            ],
                            delimiter="\t"
                        )
                        writer.writeheader()
                        writer.writerows(state["timings"])
                    record.update(
                        status=status,
                        error=error,
                        failed_stage=state.get(
                            "failed_stage",
                            "individual analysis" if error else ""
                        ),
                        ended_utc=time.strftime( "%Y-%m-%dT%H:%M:%SZ", time.gmtime() ),
                        duration_sec=elapsed,
                        epochs=state["snapshots"].get( "Epoch creation and ERN", {} )
                    )
                    _write_summary(summary_path, record, state["snapshots"])
                    write_run_state(state_path, record)
                except Exception as report_error:
                    record.update(
                        status="failed",
                        error=error or f"{type(report_error).__name__}: {report_error}",
                        failed_stage=record["failed_stage"] or "Write analysis reports"
                    )
                    write_run_state(state_path, record)
                    print(f"Could not finish analysis reports: {report_error}")
                    if not error:
                        raise
                finally:
                    _state.reset(token)

    return wrapped
