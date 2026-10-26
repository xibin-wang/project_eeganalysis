"""Prepare continuous BIDS EEG for artifact classification and ICA.

Tasks:
- Load EEG and normalize boundary annotations.
- Run period/bad-channel QC and restore the missing FCz reference.
- Apply bandpass, conditional notch and EEG re-reference.

Inputs: Participant ID, FilterParameters, active data root and path_result.

Outputs: Preprocessed MNE Raw and the participant's period-QC report.

Usage: Called by ica.py; direct execution preprocesses the first prepared subject.
"""

from __future__ import annotations

from pipeline_logging import print, timed_step
import pandas as pd
import mne
from pathlib import Path
from scipy.signal import welch

from mne_bids import BIDSPath, read_raw_bids
import numpy as np

from constants import PathConstants, FilterParameters, PeriodQCConfig
from ica_preparation import detect_period_artifacts, get_excluded_start_ranges


@timed_step("Read BIDS")
def read_data(
    file_path: Path,
    p_id: str,
) -> mne.io.BaseRaw:
    """Load one participant from BIDS and normalize discontinuity annotation labels."""
    # Check file path
    if not file_path.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {file_path}")

    # Read data with BIDS
    bids_path = BIDSPath(
        subject=p_id.removeprefix("sub-"),
        task="Flanker",
        datatype="eeg",
        root=file_path,
        extension=".set"
    )
    raw_bids = read_raw_bids(bids_path, extra_params={"preload": True})

    raw_bids.annotations.description[:] = [
        desc.removeprefix("/") for desc in raw_bids.annotations.description
    ]

    raw_bids.annotations.rename({
        desc: "BAD_boundary"
        for desc in raw_bids.annotations.description
        if "boundary" in desc.casefold()
        or desc.casefold().startswith("edge")
    })

    return raw_bids


@timed_step("Line-noise detection")
def has_line_noise(
    raw_data: mne.io.BaseRaw,
    line_freq: float = 50,
    threshold_db: float = 8,
    min_fraction: float = 0.20,
    window_sec: float = 4,
    step_sec: float = 2,
) -> bool:
    """Detect persistent power-line peaks using valid EEG analysis windows."""

    sample_rate = raw_data.info['sfreq']
    picks = mne.pick_types(raw_data.info, eeg=True)

    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must bigger than 0.")

    if not 0 < min_fraction <= 1:
        raise ValueError("min_fraction must in (0, 1].")

    win = round(sample_rate * window_sec)
    step = round(sample_rate * step_sec)

    if win < 2 or step < 1:
        raise ValueError("Detection requires at least two samples per window and one per step.")

    if sample_rate / 2 <= line_freq + 5:
        raise ValueError("Nyquist frequency is too low.")

    if len(picks) == 0 or raw_data.n_times < win:
        raise ValueError("Insufficient EEG channels or data.")

    results = []
    excluded_ranges = get_excluded_start_ranges(raw_data, win, include_periods=True)

    for start in range(0, raw_data.n_times - win + 1, step):
        if any(left < start < right for left, right in excluded_ranges):
            continue
        data = raw_data.get_data(picks=picks, start=start, stop=start + win)

        if not np.isfinite(data).all():
            continue
        freqs, psd = welch(
            data,
            fs=sample_rate,
            nperseg=min(int(2*sample_rate),win),
            axis=-1
        )

        line_band = (
            (freqs >= line_freq - 0.5) &
            (freqs <= line_freq + 0.5)
        )

        background = (
            ((freqs >= line_freq - 5) & (freqs <= line_freq -2)) |
            ((freqs >= line_freq + 2) & (freqs <= line_freq + 5))
        )

        if not line_band.any() or not background.any():
            raise ValueError("Window is too short to resolve line and background frequency bands.")
        peak = np.max(psd[:, line_band], axis=1)
        noise = np.median(psd[:, background], axis=1)

        eps = np.finfo(float).tiny
        ratio_db = 10 * np.log10((peak + eps) / (noise + eps))

        results.append(ratio_db > threshold_db)

    if not results:
        raise ValueError("No valid windows available for line-noise detection.")

    fractions = np.mean(results, axis=0)

    return bool(np.any(fractions >= min_fraction))


@timed_step("Bandpass and optional notch")
def power_filter(
    raw_data: mne.io.BaseRaw,
    config: FilterParameters,
) -> mne.io.BaseRaw:
    """Apply the configured bandpass and a notch only when line noise is detected."""

    if not 0 < config.highpass_freq < config.lowpass_freq < raw_data.info["sfreq"] / 2:
        raise ValueError("Require 0 < highpass < lowpass < Nyquist.")

    data_filtered = raw_data.copy()

    data_filtered.filter(
        l_freq=config.highpass_freq,
        h_freq=config.lowpass_freq,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_design="firwin",
        skip_by_annotation=("edge", "bad", "boundary"),
        verbose=True
    )
    print(f"Data filtered to {config.highpass_freq}Hz and {config.lowpass_freq}Hz.")

    if config.highpass_freq < config.line_freq < config.lowpass_freq:
        if has_line_noise(
                data_filtered,
                line_freq=config.line_freq,
                threshold_db=config.threshold_db,
                min_fraction=config.min_fraction,
                window_sec=config.window_sec,
                step_sec=config.step_sec
        ):
            print(f"Found line noise at {config.line_freq}Hz.")
            data_filtered.notch_filter(
                freqs=config.line_freq,
                picks="eeg",
                method="spectrum_fit",
                skip_by_annotation=("edge", "bad", "boundary"),
            )
        else:
            print(f"Found no line noise at {config.line_freq}Hz.")

    return data_filtered


@timed_step("Restore FCz")
def restore_fcz(raw_data: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Restore the missing FCz reference channel and its head-space location."""

    if any(ch.casefold() == "fcz" for ch in raw_data.ch_names):
        return raw_data

    montage = raw_data.get_montage()

    if montage is None or montage.get_positions()["coord_frame"] != "head":
        raise ValueError("Restoring FCz requires a montage in head coordinates.")

    positions = montage.get_positions()
    ch_pos = positions["ch_pos"].copy()

    raw_data = mne.add_reference_channels(raw_data, ref_channels=["FCz"], copy=False)

    ch_pos["FCz"] = np.array([ 0.0, 0.0388, 0.0922 ])

    new_montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=positions["nasion"],
        lpa=positions["lpa"],
        rpa=positions["rpa"],
        coord_frame="head"
    )

    raw_data.set_montage(new_montage)

    print(f"Restored FCz at {raw_data.get_montage().get_positions()['ch_pos']["FCz"]}.")

    return raw_data


@timed_step("EEG reference")
def ref_data(
    raw_data: mne.io.BaseRaw,
    mast_exclude: bool = True,
    ref_channel: str = "average",
) -> mne.io.BaseRaw:
    """Re-reference EEG using valid channels and the configured exclusion policy."""

    if mast_exclude:
      exclude = {"MASTl", "MASTr", "VOGbelow"}
    else:
      exclude = {"VOGbelow"}

    exclude |= set(raw_data.info["bads"])

    if ref_channel == "average":
        ref_channels = [
            ch for ch in raw_data.ch_names
            if raw_data.get_channel_types(picks=[ch])[0] == "eeg"
            and ch not in exclude
        ]

        if not ref_channels:
            raise ValueError("No valid EEG reference channels.")
        raw_avg = raw_data.copy().set_eeg_reference(
            ref_channels=ref_channels,
            ch_type="eeg",
        )
    else:
        channel_name = next(
            ch for ch in raw_data.ch_names
            if ch.lower() == ref_channel.lower()
        )
        if channel_name in exclude:
            raise ValueError(f"Reference channel {channel_name} is excluded or marked bad.")
        raw_avg = raw_data.copy().set_eeg_reference(
            ref_channels=[channel_name],
            ch_type="eeg"
        )

    return raw_avg


@timed_step("Preprocessing")
def main(
    p_id: str,
    cfg_filter: FilterParameters,
    path_data: Path,
    path_result: Path | str,
) -> mne.io.BaseRaw:
    """Run period QC, restore FCz, filter and re-reference one participant."""
    raw = read_data(file_path=path_data, p_id=p_id)
    raw, report_period = detect_period_artifacts(raw, PeriodQCConfig())
    output = Path(path_result) / p_id
    output.mkdir(parents=True, exist_ok=True)
    report_period.to_csv(
        output / "bad_period_results.tsv",
        sep="	",
        index=False,
        na_rep="n/a"
    )

    if report_period["data_unusable"].any():
        raise ValueError(f"{p_id}: record failed period QC; see {output}")

    irrelevant = [
        i for i,
        scope in enumerate(raw.annotations.ch_names) if scope and set(scope).issubset(raw.info["bads"]) and raw.annotations.description[i].casefold().startswith((
            "bad",
            "review"
        ))
    ]
    raw.annotations.delete(irrelevant)
    raw_restored = restore_fcz(raw)
    raw_filtered = power_filter(raw_restored, cfg_filter)
    raw_ref = ref_data(raw_filtered, mast_exclude=True, ref_channel="average")

    print("Preprocessed before artifact detection finished.")

    return raw_ref

if __name__ == "__main__":
    path = PathConstants()
    filter_config = FilterParameters()
    from bids_correction import prepared_data
    folder_data = prepared_data(path.FOLDER_RESULT)
    if folder_data is None:
        raise FileNotFoundError("Prepare BIDS data with scripts/main.py first.")
    participant_info = pd.read_csv(folder_data / "participants.tsv", sep="\t")
    participant_id = participant_info["participant_id"][0]
    main(participant_id, filter_config, folder_data, path.FOLDER_RESULT)


