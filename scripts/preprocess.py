"""
Script for preprocessing EEGlab set data until ICA

"""

import pandas as pd
import mne
from pathlib import Path
from scipy.signal import welch

from mne_bids import BIDSPath, read_raw_bids
import numpy as np

from constants import PathConstants, FilterParameters


def read_data(file_path: Path, p_id: str) -> mne.io.BaseRaw:
    # Check file path
    if not file_path.exists():
        print("Folder data does not exist")
        exit()

    # Read data with BIDS
    bids_path = BIDSPath(
        subject=p_id.removeprefix("sub-"),
        task="Flanker",
        datatype="eeg",
        root=file_path,
        extension=".set"
    )
    raw_bids = read_raw_bids(
        bids_path,
        extra_params={"preload": True},
    )

    raw_bids.annotations.description[:] = [
        desc.removeprefix("/") for desc in raw_bids.annotations.description
    ]

    return raw_bids


def has_line_noise(
        raw_data: mne.io.BaseRaw,
        line_freq: float=50,
        threshold_db: int=8,
        min_fraction: float=0.20,
        window_sec: float=4,
        step_sec: float=2,
) -> bool:

    sample_rate = raw_data.info['sfreq']
    picks = mne.pick_types(
        raw_data.info,
        eeg=True
    )

    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must bigger than 0.")
    if not 0 < min_fraction <= 1:
        raise ValueError("min_fraction must in (0, 1].")

    win = int(sample_rate * window_sec)
    step = int(sample_rate * step_sec)

    if sample_rate / 2 <= line_freq + 5:
        raise ValueError("Nyquist frequency is too low.")
    if len(picks) == 0 or raw_data.n_times < win:
        raise ValueError("Insufficient EEG channels or data.")

    results = []

    for start in range(0, raw_data.n_times - win + 1, step):
        data = raw_data.get_data(
            picks=picks,
            start=start,
            stop=start + win,
        )

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
            ((freqs >= line_freq - 5) &
             (freqs <= line_freq -2)) |
            ((freqs >= line_freq + 2) &
             (freqs <= line_freq + 5))
        )

        peak = np.max(psd[:, line_band], axis=1)
        noise = np.median(psd[:, background], axis=1)

        eps = np.finfo(float).tiny
        ratio_db = 10 * np.log10(
            (peak + eps) / (noise + eps)
        )

        results.append(ratio_db > threshold_db)

    if not results:
        print("No noise found.")

    fractions = np.mean(results, axis=0)

    return bool(np.any(fractions >= min_fraction))


def power_filter(
        raw_data: mne.io.BaseRaw,
        config: FilterParameters
) -> mne.io.BaseRaw:
    data_filtered = raw_data.copy()

    data_filtered.filter(
        l_freq=config.highpass_freq,
        h_freq=config.lowpass_freq,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=True
    )
    print(f"Data filtered to {config.highpass_freq}Hz and {config.lowpass_freq}Hz.")

    if config.highpass_freq > config.line_freq + 5:
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
                method="spectrum_fit"
            )
        else:
            print(f"Found no line noise at {config.line_freq}Hz.")

    return data_filtered


def restore_fcz(raw_data: mne.io.BaseRaw) -> mne.io.BaseRaw:
    montage = raw_data.get_montage()
    positions = montage.get_positions()
    ch_pos = positions["ch_pos"].copy()

    raw_data = mne.add_reference_channels(
        raw_data,
        ref_channels=["FCz"],
        copy=False
    )

    ch_pos["FCz"] = np.array([
        0.0,
        0.0388,
        0.0922
    ])

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

def ref_data(
        raw_data: mne.io.BaseRaw,
        mast_exclude: bool = True,
        ref_channel: str = "average"
) -> mne.io.BaseRaw:

    if mast_exclude:
      exclude = {"MASTl", "MASTr", "VOGbelow"}
    else:
      exclude = {"VOGbelow"}

    if ref_channel == "average":
        ref_channels = [
            ch for ch in raw_data.ch_names
            if raw_data.get_channel_types(
                picks=[ch]
            )[0] == "eeg"
            and ch not in exclude
        ]

        raw_avg = raw_data.copy().set_eeg_reference(
            ref_channels=ref_channels,
            ch_type="eeg",
        )
    else:
        channel_name = next(
            ch for ch in raw_data.ch_names
            if ch.lower() == ref_channel.lower()
        )
        raw_avg = raw_data.copy().set_eeg_reference(
            ref_channels=channel_name,
            ch_type="eeg"
        )

    return raw_avg

def main(p_id, cfg_filter, path_data) -> mne.io.BaseRaw:
    raw = read_data(file_path=path_data, p_id=p_id)
    raw_restored = restore_fcz(raw)
    raw_filtered = power_filter(raw_restored, cfg_filter)
    raw_ref = ref_data(raw_filtered, mast_exclude=True, ref_channel="average")

    print("Preprocessed before artifact detection finished.")

    return raw_ref

if __name__ == "__main__":
    path = PathConstants()
    filter_config = FilterParameters()
    folder_data = path.FOLDER_ROOT / "eeglabset_fix"
    participant_info = pd.read_csv(folder_data / "participants.tsv", sep="\t")
    participant_id = participant_info["participant_id"][0]
    main(participant_id, filter_config, folder_data)


