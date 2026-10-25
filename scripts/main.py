from bids_correction import main as bids_correction
from group_analysis import main as group_analysis
from ica import*

from constants import*

if __name__ == "__main__":

    # ----------------------- General configuration ------------------------
    bids_correction()
    mne.set_config("MNE_BROWSER_BACKEND", "qt")
    path = PathConstants()
    filter_config = FilterParameters()
    ica_prep = ICAPrep()
    ica_config = ICAParameters()
    group_config = GroupParameters()
    folder_data = path.FOLDER_ROOT / "eeglabset_fix"
    participant_info = pd.read_csv(folder_data / "participants.tsv", sep="\t")
    path_result = path.FOLDER_ROOT / "results"
    path_result.mkdir(parents=True, exist_ok=True)


    # ----------------------- Data path check ------------------------
    check_result = []
    for participant_id in participant_info.participant_id.unique():
        if folder_data.joinpath(participant_id).is_dir():
            continue
        elif ~folder_data.joinpath(participant_id).is_dir():
            check_result.append(participant_id)
    if check_result:
        print(f"Can not find data folder for following participants: {",".join(check_result)}. Please check before proceeding.")
        exit()


    # ----------------------- Individual analysis ------------------------
    while True:
        mode_analysis = input("Would you like to analyze the data of specific participant or all participants? \n"
                              "1. Single participant ica \n"
                              "2. All participants ica\n"
                              "3. Skip to group analysis\n")
        if mode_analysis.isdigit():
            mode_analysis = int(mode_analysis)
            if mode_analysis == 1:
                while True:
                    index = input("Which participant you want to analyse? Just give number:")
                    if int(index) < len(participant_info["participant_id"]):
                        index = int(index)
                        participant_id = participant_info["participant_id"][index]
                        break
                    else:
                        print("Please enter a number.")
                path_data = folder_data / participant_id / "eeg"
                data_original, data_train, ica_results = analysis_individual(filter_config, ica_config, participant_id,
                                                                             folder_data, path_result)
                make_epochs(data_original, ica_results,ica_prep, participant_id, path_data)
            elif mode_analysis == 2:
                for participant_id in participant_info.participant_id.unique():
                    path_data = folder_data / participant_id / "eeg"
                    path_result.joinpath(participant_id).mkdir(parents=True, exist_ok=True)
                    data_original, data_train, ica_results = analysis_individual(filter_config, ica_config, participant_id,
                                                                                 folder_data, path_result)
                    make_epochs(data_original, ica_results, ica_prep, participant_id, path_data)
            elif mode_analysis == 3:
                break
        else:
            print("Please enter a valid number.")

    # ----------------------- Group ERP analysis ------------------------
    group_analysis(participant_info, group_config, folder_data, path_result)
