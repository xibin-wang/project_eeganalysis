"""
Correction of data in form of BIDS structure for following processing

Main functions:
1. Make a copy of all original files into folder eeglabset_fix
2. Swap data from channels.tsv and electrodes.tsv files
    useful while read_raw_bids() function reports KeyError, swap data stored in channels.tsv and electrodes.tsv
3. Change onset values' unit from sample into seconde in _events.tsv files
    useful while read_raw_bids() function gets only one event (inconsistency of onset units between BIDS event file
    and EEGLab .set file)
4. Change duration and event value in _event.tsv files according to .set fils relevant data

"""

from pathlib import Path
import shutil

import json
import pandas as pd
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids

from constants import PathConstants

def make_copy(source_dir:Path, target_dir:Path):
    """make copy of eeglabset_fix and swap channels.tsv and electrodes.tsv files

    """
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    old_channels = target_dir/"electrodes.tsv"
    old_electrodes = target_dir/"channels.tsv"

    temp = target_dir/"_temp_swap.tsv"
    old_channels.rename(temp)
    old_electrodes.rename(target_dir/"electrodes.tsv")
    temp.rename(target_dir/"channels.tsv")

    # Check if 3d coordination and electrodes' type information are stored properly followed BIDS structure
    old_channels = source_dir/"channels.tsv"
    old_electrodes = source_dir/"electrodes.tsv"
    channels_df = pd.read_csv(old_channels, sep="\t")
    electrodes_df = pd.read_csv(old_electrodes, sep="\t")
    assert {"name", "x", "y", "z"}.issubset(channels_df.columns)
    print(f"3d coordination information stored in {old_channels}.")
    assert {"name", "type", "units"}.issubset(electrodes_df.columns)
    print(f"Electrodes' type information stored in {old_electrodes}.")

    new_channels = target_dir/"channels.tsv"
    new_electrodes = target_dir/"electrodes.tsv"
    channels_df = pd.read_csv(new_channels, sep="\t")
    electrodes_df = pd.read_csv(new_electrodes, sep="\t")
    assert {"name", "type", "units"}.issubset(channels_df.columns)
    print(f"Electrodes' type information stored in {new_channels}.")
    assert {"name", "x", "y", "z"}.issubset(electrodes_df.columns)
    print(f"3d coordination information stored in {new_electrodes}.")

def change_units(target_dir:Path,):
    """Change onset unit in every _events.tsv

    """
    participant_info = pd.read_csv(target_dir/"participants.tsv", sep="\t")
    for _, row in participant_info.iterrows():
        participant_id = row["participant_id"]
        event_path = target_dir/participant_id/"eeg"/f"{participant_id}_task-Flanker_events.tsv"
        json_path = target_dir/participant_id/"eeg"/f"{participant_id}_task-Flanker_eeg.json"
        with open(json_path) as f:
            eeg_info = json.load(f)
        freq = eeg_info["SamplingFrequency"]
        df = pd.read_csv(event_path, sep="\t")

        # verify onset and sample exist
        required = {"onset","sample"}
        missing = required - set(df.columns)

        if missing:
            print(f"Skipping {participant_id} due to missing data {missing}.")
            continue

        corrected_onset = pd.to_numeric(df["sample"])/freq
        df["onset"] = corrected_onset

        df.to_csv(
            event_path,
            sep="\t",
            index=False,
            na_rep="nan",
        )

def event_inspect(target_dir:Path):
    """This function inspect inconsistency between BIDS event files and set file event annotations.

    Check inspect results in results/inspect/inspect_results.txt.
    If results indicate any kind of inconsistency, please check manually using event files extracted in same directory.
    """
    participant_info = pd.read_csv(target_dir/"participants.tsv", sep="\t")
    id_list = participant_info["participant_id"].tolist()
    output_path = target_dir.resolve().parent/"results"/"inspect"
    output_path.mkdir(parents=True, exist_ok=True)
    list_result = []
    for participant_id in id_list:
        participant_path = target_dir/participant_id/"eeg"/f"{participant_id}_task-Flanker_eeg.set"
        # Read data with .set file
        raw_set = mne.io.read_raw_eeglab(
            participant_path,
            preload=True
        )
        # Read data with BIDS
        bids_path = BIDSPath(
            subject=participant_id.removeprefix("sub-"),
            task="Flanker",
            datatype="eeg",
            root=target_dir,
            extension=".set"
        )
        raw_bids = read_raw_bids(
            bids_path,
            extra_params={"preload": True},
        )

        df_eeglab = raw_set.annotations.to_data_frame(
            time_format=None
        )
        df_eeglab.to_csv(output_path/f"{participant_id}_set_annotations.tsv", sep="\t", index=False)
        df_bids = raw_bids.annotations.to_data_frame(
            time_format=None
        )
        df_bids.to_csv(output_path/f"{participant_id}_bids_annotations.tsv", sep="\t", index=False)

        if len(df_eeglab) == len(df_bids):
            if np.allclose(
                    df_eeglab["onset"].to_numpy(),
                    df_bids["onset"].to_numpy(),
                    atol=2e-6,
                    rtol=0
            ):
                string_inspect = f"Subject {participant_id} BIDS structure data equals to corresponding set file data."
                list_result.append(string_inspect)
            else:
                string_inspect = f"Subject {participant_id} BIDS structure data DOES NOT equal corresponding set file data, please check."
                list_result.append(string_inspect)
        elif df_eeglab["onset"].isin(df_bids["onset"]).all():
            list_inspect = list(set(df_bids["onset"]) - set(df_eeglab["onset"]))
            string_inspect = (f"Inconsistent BIDS structure data for subject {participant_id}: missing data {list_inspect} "
                              f"in .set file.")
            list_result.append(string_inspect)
        elif df_bids["onset"].isin(df_eeglab["onset"]).all():
            list_inspect = list(set(df_eeglab["onset"]) - set(df_bids["onset"]))
            string_inspect = (f"Inconsistent BIDS structure data for subject {participant_id}: missing data {list_inspect}"
                  f"in BIDS _event file.")
            list_result.append(string_inspect)

    with open(output_path/"inspect_results.txt", "w") as f:
        f.write("\n".join(list_result))

def event_correction(target_dir:Path):
    """This function use set file stored information to restore duration. Make sure that inconsistency indicated by
    event_inspect() function is correctly handled.

    Some inconsistency occurs due to only one additional boundary event at the beginning of the set file. You can
    ignore it and run this function. The following processing will use only read_raw_bids() function.

    """
    participant_info = pd.read_csv(target_dir/"participants.tsv", sep="\t")
    id_list = participant_info["participant_id"].tolist()
    for participant_id in id_list:
        path_event = target_dir/participant_id/"eeg"/f"{participant_id}_task-Flanker_events.tsv"
        participant_path = target_dir/participant_id/"eeg"/f"{participant_id}_task-Flanker_eeg.set"
        # Read data with .set file
        raw_set = mne.io.read_raw_eeglab(
            participant_path,
            preload=True
        )
        df_eeglab = raw_set.annotations.to_data_frame(
            time_format=None
        )

        event_info = pd.read_csv(path_event,sep="\t")
        event_info["value"] = event_info["value"].str.removeprefix("x")
        event_info["trial_type"] = event_info["trial_type"].fillna("")

        for index, row in event_info.iterrows():
            onset = row["onset"]
            description = row["value"]

            matches = np.isclose(
                onset,
                df_eeglab["onset"],
                atol=2e-6,
                rtol=0
            )
            if matches.sum() > 1:
                matches = matches & (df_eeglab["description"] == description)

            if matches.sum() == 1:
                event_info.loc[index, "duration"] = (
                    df_eeglab.loc[matches, "duration"].iloc[0]
                )
            elif matches.sum() == 0:
                print(f"No matching onset: {onset}")
                event_info.loc[index, "duration"] = 0
            else:
                print(f"Multiple matches for onset: {onset} for {description}.")


        event_info.to_csv(
            target_dir/participant_id/"eeg"/f"{participant_id}_task-Flanker_events.tsv",
            sep="\t",
            index=False
        )

def main():
    path = PathConstants()
    source = path.FOLDER_ROOT / "eeglabset"
    target = path.FOLDER_ROOT / "eeglabset_fix"
    make_copy(source, target)
    change_units(target)
    event_inspect(target)
    event_correction(target)

    print("BIDS data correction finished.")


if __name__ == "__main__":
    main()