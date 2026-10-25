import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mne
import statsmodels.formula.api as sm

from constants import *

def compute_roi_erps(
    trial_df,
    erp_trials,
    diff_df,
    cong_levels=(0.0, 0.33, 0.66, 1.0),
):
    erp_lists = {
        (cong, acc): []
        for cong in cong_levels
        for acc in (0, 1)
    }

    if len(trial_df) != len(erp_trials):
        raise ValueError(
            "trial_df and erp_trials rows do not match"
        )
    included_cells = set(
        zip(
            diff_df["participant"].astype(str),
            diff_df["Congruency"].astype(float),
        )
    )
    erp_data = {}

    participants = trial_df["participant"].astype(str).to_numpy()
    congruency = trial_df["Congruency"].to_numpy(dtype=float)
    accuracy = trial_df["ACC"].to_numpy(dtype=float)

    for sub in np.unique(participants):
        sub_mask = participants == sub
        for cong in cong_levels:
            if (sub, cong) not in included_cells:
                continue
            condition_mask = (
                sub_mask
                & np.isclose(congruency, cong, atol=0.005)
            )

            mask_correct = condition_mask & (accuracy == 1)
            mask_error = condition_mask & (accuracy == 0)

            # Both conditions must have valid trials
            if not mask_correct.any() or not mask_error.any():
                continue

            # Mean ERP per participant and condition
            erp_lists[(cong, 1)].append(
                erp_trials[mask_correct].mean(axis=0)
            )
            erp_lists[(cong, 0)].append(
                erp_trials[mask_error].mean(axis=0)
            )

    erp_data = {
        key: (
            np.stack(waves, axis=0)
            if waves
            else np.empty((0, erp_trials.shape[1]), dtype=float)
        )
        for key, waves in erp_lists.items()
    }
    return erp_data

def plot_roi_erps(
    erp_data,
    times_ms,
    group_config,
    path_result=None,
):
    cong_levels = (0.0, 0.33, 0.66, 1.0)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, cong in zip(axes.flat, cong_levels):

        for acc, color, label in [
            (1, "tab:blue", "Correct"),
            (0, "tab:red", "Incorrect"),
        ]:
            waves = erp_data[(cong, acc)]
            if len(waves) == 0:
                continue
            mean_wave = waves.mean(axis=0)
            ax.plot(
                times_ms,
                mean_wave,
                color=color,
                linewidth=2,
                label=f"{label} (N={len(waves)})",
            )

        # LMM analysis window
        tmin, tmax = group_config.win_analysis_s

        ax.axvspan(
            tmin * 1000,
            tmax * 1000,
            color="gray",
            alpha=0.15,
        )

        ax.axvline(0, color="gray", linestyle="--")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"Congruency = {cong:.2f}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend()

    fig.suptitle(
        f"ROI ERP: {', '.join(group_config.roi_channels)}"
    )

    if path_result is not None:
        path_result = Path(path_result)
        path_result.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            path_result / "ROI_ERP_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return fig

def read_roi_based(group_config, participant_info, folder_data) -> tuple[pd.DataFrame, np.ndarray, float]:

    trial_data = []
    erp_data = []
    times_ms = None

    for participant_id in participant_info["participant_id"].unique():
        path_data = folder_data / participant_id / "eeg" / f"{participant_id}_task_Flanker_eeg_epochs.fif"
        epochs = mne.read_epochs(
            path_data,
            preload=True,
        )

        missing_roi = set(group_config.roi_channels) - set(epochs.ch_names)
        bad_roi = set(group_config.roi_channels) & set(epochs.info["bads"])

        if missing_roi or bad_roi:
            raise ValueError(
                f"{participant_id}: missing ROI={missing_roi}, bad ROI={bad_roi}"
            )

        data = epochs.get_data(
            picks=group_config.roi_channels,
            units="uV",
        )

        roi_waveforms = data.mean(axis=1)
        tmin, tmax = group_config.win_analysis_s
        time_mask = (
            (epochs.times >= tmin - 1e-9)
            & (epochs.times <= tmax + 1e-9)
        )
        if not time_mask.any():
            raise ValueError("No valid samples between analysis window.")

        amplitude = roi_waveforms[:, time_mask].mean(axis=1)

        md = epochs.metadata.copy().reset_index(drop=True)
        md["participant"] = participant_id
        md["amplitude"] = amplitude
        trial_data.append(
            md[[
                "participant",
                "Congruency",
                "ACC",
                "RT",
                "amplitude",
            ]]
        )

        erp_data.append(roi_waveforms)
        current_times = epochs.times * 1000
        if times_ms is None:
            times_ms = current_times.copy()
        elif (
                times_ms.shape != current_times.shape
                or not np.allclose(times_ms, current_times)
        ):
            raise ValueError(f"Inconsistent time axis: {participant_id}")

    trial_df = pd.concat(
        trial_data,
        ignore_index=True,
    )

    erp_trails = np.concatenate(erp_data, axis=0)

    trial_df.to_csv(
        folder_data / "ROI_based_trial_amplitudes.tsv",
        sep="\t",
        index=False,
    )

    return trial_df, erp_trails, times_ms

def roi_based_analysis(participant_info, data_roi_df, erp_trials, times_ms, path_result, group_config):

    #--------------------- Data preparation ----------------------------
    participant_info = participant_info.rename(
    columns={
        "participant_id": "participant",
    }
)
    assert participant_info["participant"].is_unique
    means = (
        data_roi_df
        .groupby(["participant", "Congruency", "ACC"])["amplitude"]
        .mean()
        .unstack("ACC")
    )

    diff_df = (
        (means[0] - means[1])
        .rename("ACC_diff")
        .reset_index()
    )
    diff_df = diff_df.merge(
        participant_info[["participant", "PSWQ", "Sex"]],
        on="participant",
        how="left",
        validate="many_to_one",
    )

    #--------------------- Model fit ----------------------------
    diff_df["Congruency"] = pd.Categorical(
        diff_df["Congruency"],
        categories=[0.0, 0.33, 0.66, 1.0],
    )
    diff_df["PSWQ"] = pd.to_numeric(
        diff_df["PSWQ"],
        errors="raise"
    )
    diff_df["Sex"] = diff_df["Sex"].astype("category")

    diff_df = diff_df.dropna(
        subset=["ACC_diff", "PSWQ", "Sex"]
    ).copy()

    model = sm.mixedlm(
        "ACC_diff ~ C(Congruency) + PSWQ + C(Sex)",
        data=diff_df,
        groups=diff_df["participant"],
        re_formula="1",
    )

    result = model.fit(reml=True)
    print(result.summary())

    fe_names = result.fe_params.index
    sig_effects = []
    result_df = pd.DataFrame({
        "Effect": fe_names,
        "Beta": result.fe_params.values,
        "SE": result.bse_fe.values,
        "z": result.tvalues.loc[fe_names].values,
        "p": result.pvalues.loc[fe_names].values,
        "CI_lower": result.conf_int().loc[fe_names, 0].values,
        "CI_upper": result.conf_int().loc[fe_names, 1].values,
    })

    result.save(path_result/"LMM_model.pickle")
    result_df.to_csv(
        path_result/"LMM_results.tsv",
        sep="\t",
        index=False
    )

    for effect in fe_names:
        if effect == "Intercept":
            continue
        p = result.pvalues[effect]
        beta = result.fe_params[effect]

        if np.isfinite(p) and p < group_config.alpha:
            sig_effects.append(effect)

            print(
                f"Significant effect: {effect}\n"
                f"Beta = {beta:.4f}\n"
                f"p = {p:.4g}\n"
            )

    if not sig_effects:
        print("No significant fixed effects (p < 0.05).")
    else:
        erp_data = compute_roi_erps(
            trial_df=data_roi_df,
            erp_trials=erp_trials,
            diff_df=diff_df,
        )

        plot_roi_erps(
            erp_data=erp_data,
            times_ms=times_ms,
            group_config=group_config,
            path_result=path_result,
        )

    return result, diff_df

def main(participant_info, group_config, folder_data, path_result):

    data_roi_df, erp_trials, times_ms = read_roi_based(
        group_config=group_config,
        participant_info=participant_info,
        folder_data=folder_data,
    )

    roi_based_analysis(
        participant_info=participant_info,
        data_roi_df=data_roi_df,
        erp_trials=erp_trials,
        times_ms=times_ms,
        path_result=path_result,
        group_config=group_config,
    )


# if __name__ == "__main__":
#     path = PathConstants()
#     group_config = GroupParameters()
#     folder_data = path.FOLDER_ROOT / "eeglabset_fix"
#     participant_info = pd.read_csv(folder_data / "participants.tsv", sep="\t").set_index("participant_id")
#     path_result = path.FOLDER_ROOT / "results"

