"""Combine completed participant results for group ERP and statistical analysis.

Tasks:
- Validate inputs and reuse ERN CSV files, with epoch-based fallback.
- Export trial amplitudes, compact participant ERPs and comparison plots.
- Fit the random-intercept LMM and pooled Delta ERN-PSWQ correlation.
- Record exclusions, diagnostics and current group-output status.

Inputs: Participant table, group settings, active data root and path_result.

Outputs: Group TSV/NPZ tables, ERP figure, model and run-state JSON.

Usage: Call main() from main.py or import individual analysis functions.
"""

from __future__ import annotations

from pipeline_logging import print
from analysis_validation import participant_paths, load_completed_epochs, validate_ern_summary, write_run_state
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mne
import statsmodels.formula.api as sm
import warnings
import json
import hashlib
import time
from scipy.stats import pearsonr
from dataclasses import asdict

from constants import *
from typing import Any, TypeAlias
from numpy.typing import NDArray
from matplotlib.figure import Figure
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper

SubjectERPs: TypeAlias = dict[tuple[str, float, int], NDArray[np.float64]]
ConditionERPs: TypeAlias = dict[tuple[float, int], NDArray[np.float64]]


def compute_roi_erps(subject_erps: SubjectERPs) -> ConditionERPs:
    """Group subject-level ROI means by Congruency and accuracy for plotting."""
    levels = sorted({key[1] for key in subject_erps})
    n_times = len(next(iter(subject_erps.values()))) if subject_erps else 0
    result = {}

    for level in levels:
        for acc in (0, 1):
            waves = [
                wave for (_, cong, accuracy),
                wave in subject_erps.items() if cong == level and accuracy == acc
            ]
            result[(level, acc)] = np.stack(waves) if waves else np.empty((0, n_times))

    return result


def plot_roi_erps(
    erp_data: ConditionERPs,
    times_ms: NDArray[np.float64],
    group_config: GroupParameters,
    path_result: Path | str,
) -> Figure:
    """Save descriptive ROI ERP curves with participant counts for each condition."""
    cong_levels = sorted({key[0] for key in erp_data})

    if not cong_levels:
        raise ValueError("No Congruency levels available for ERP plotting.")

    path_result = Path(path_result)
    path_result.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        int(np.ceil(len(cong_levels) / 2)), 2,
        figsize=(12, 4 * int(np.ceil(len(cong_levels) / 2))),
        squeeze=False,
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

        ax.axvspan(tmin * 1000, tmax * 1000, color="gray", alpha=0.15)

        ax.axvline(0, color="gray", linestyle="--")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"Congruency = {cong:g}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend()

    for ax in axes.flat[len(cong_levels):]:
        ax.set_visible(False)

    fig.suptitle(f"ROI ERP: {', '.join(group_config.roi_channels)}")

    try:
        fig.savefig(
            path_result / "ROI_ERP_comparison.png",
            dpi=300,
            bbox_inches="tight"
        )
    finally:
        plt.close(fig)

    return fig


def read_roi_based(
    group_config: GroupParameters,
    participant_info: pd.DataFrame,
    folder_data: Path | str,
    path_result: Path | str | None = None,
) -> tuple[pd.DataFrame, SubjectERPs, NDArray[np.float64]]:
    """Return condition-level model rows and compact subject-condition ERP means."""

    path_result = Path(path_result) if path_result is not None else PathConstants().FOLDER_RESULT
    path_result.mkdir(parents=True, exist_ok=True)
    expected = dict(
        filters=asdict(FilterParameters()),
        period_qc=asdict(PeriodQCConfig()),
        artifact_detection=asdict(ICAPrep()),
        ica=asdict(ICAParameters()),
        epochs=asdict(ICAPrep())
    )
    expected = json.loads(json.dumps(expected))

    config = dict(
        group=asdict(group_config),
        preprocessing_expected=expected,
        formula="ACC_diff ~ C(Congruency) + PSWQ + C(Sex)",
        units="uV",
        delta_direction="error-minus-correct",
        source_policy="validated personal CSV first; compatible epochs fallback",
        congruency_source="BIDS response events matched by Trial",
        epoch_fingerprint="size and mtime_ns; no full EEG hash",
        waveform_policy="one participant at a time; retain condition means only"
    )
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    (path_result / "group_analysis_config.json").write_text(
        json.dumps(
            dict(
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                config_sha256=config_hash,
                mne_version=mne.__version__,
                numpy_version=np.__version__,
                pandas_version=pd.__version__,
                **config
            ),
            indent=2
        ) + "\n",
        encoding="utf-8"
    )
    columns = [
        "participant",
        "Congruency",
        "n_error",
        "n_correct",
        "mean_ern_uv",
        "mean_crn_uv",
        "ACC_diff"
    ]
    cells, compact_erps, file_checks = [], {}, []
    times_ms = None
    trial_output = path_result / "ROI_based_trial_amplitudes.tsv"
    pd.DataFrame(columns=[
        "participant",
        "Trial",
        "Congruency",
        "ACC",
        "RT",
        "amplitude"
    ]).to_csv(trial_output, sep="\t", index=False)

    for participant_id in participant_info["participant_id"].dropna().unique():
        paths = participant_paths(folder_data, path_result, participant_id)
        inputs = {
            key: paths[key]
            for key in ("epochs", "events", "summary", "state", "ern")
        }
        record = dict(
            participant_id=participant_id,
            **{key: str(file.resolve()) for key, file in inputs.items()},
            config_sha256=config_hash,
            included=False,
            n_epochs=0,
            model_source="",
            cache_reason="",
            reason=""
        )
        epochs = data = roi_waveforms = None

        try:
            for label, file in inputs.items():
                record[f"{label}_exists"] = file.is_file()
                if file.is_file():
                    stat = file.stat()
                    record[f"{label}_size"] = stat.st_size
                    record[f"{label}_mtime_ns"] = str(stat.st_mtime_ns)
                    if label != "epochs":
                        record[f"{label}_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            epochs, md, levels = load_completed_epochs(paths, participant_id, expected)
            missing_roi = set(group_config.roi_channels) - set(epochs.ch_names)
            bad_roi = set(group_config.roi_channels) & set(epochs.info["bads"])

            if missing_roi or bad_roi or not len(epochs):
                raise ValueError(f"Invalid ROI or empty epochs: missing={missing_roi}, bad={bad_roi}, n={len(epochs)}")

            try:
                condition_df = validate_ern_summary(
                    pd.read_csv(paths["ern"]),
                    participant_id,
                    group_config,
                    md,
                    levels
                )
                record["model_source"] = "personal_ern_csv"
            except Exception as exc:
                record["cache_reason"] = str(exc)
                condition_df = None
                record["model_source"] = "epochs_recomputed"
                print(f"{participant_id}: ERN CSV unavailable/incompatible ({exc}); recomputing this participant only.")
            tmin, tmax = group_config.win_analysis_s
            time_mask = (epochs.times >= tmin - 1e-9) & (epochs.times <= tmax + 1e-9)
            if (
                tmin > tmax
                or tmin < epochs.times[0] - 1e-9
                or tmax > epochs.times[-1] + 1e-9
                or not time_mask.any()
            ):
                raise ValueError("Analysis window is not fully covered by epochs.")
            current_times = epochs.times * 1000
            if (
                times_ms is not None
                and (
                    times_ms.shape != current_times.shape
                    or not np.allclose(times_ms, current_times)
                )
            ):
                raise ValueError("Epoch time axis differs from already included participants.")
            data = epochs.get_data(picks=list(group_config.roi_channels), units="uV")
            if not np.isfinite(data).all():
                raise ValueError("ROI waveforms contain non-finite samples.")

            roi_waveforms = data.mean(axis=1)
            md["participant"] = participant_id
            md["amplitude"] = roi_waveforms[:, time_mask].mean(axis=1)
            participant_erps = {
                (
                    participant_id,
                    float(level),
                    int(acc)
                ): roi_waveforms[rows.index].mean(axis=0)
                for (level, acc), rows in md.groupby(["Congruency", "ACC"])
            }
            if condition_df is None:
                grouped = md.groupby(["Congruency", "ACC"])["amplitude"]
                counts = grouped.size().unstack(
                    "ACC",
                    fill_value=0
                ).reindex(index=levels, columns=[0, 1], fill_value=0)
                means = grouped.mean().unstack("ACC").reindex(
                    index=levels,
                    columns=[0, 1]
                )
                condition_df = pd.DataFrame(dict(
                    participant=participant_id,
                    Congruency=levels,
                    n_error=counts[0].to_numpy(),
                    n_correct=counts[1].to_numpy(),
                    mean_ern_uv=means[0].to_numpy(),
                    mean_crn_uv=means[1].to_numpy(),
                    ACC_diff=(means[0] - means[1]).to_numpy()
                ))
            md[[
                "participant",
                "Trial",
                "Congruency",
                "ACC",
                "RT",
                "amplitude"
            ]].to_csv(
                trial_output,
                sep="\t",
                index=False,
                header=False,
                mode="a",
                na_rep="n/a"
            )
            cells.append(condition_df[columns])
            compact_erps.update(participant_erps)
            times_ms = current_times.copy()
            record.update(included=True, n_epochs=len(epochs))
        except Exception as exc:
            record["reason"] = str(exc)
            print(f"{participant_id}: group input excluded: {exc}")
        finally:
            del epochs, data, roi_waveforms
        file_checks.append(record)

    manifest = pd.DataFrame(file_checks)
    manifest.to_csv(
        path_result / "group_input_manifest.tsv",
        sep="\t",
        index=False,
        na_rep="n/a"
    )
    cell_df = pd.concat(
        cells,
        ignore_index=True
    ) if cells else pd.DataFrame(columns=columns)
    cell_df.to_csv(
        path_result / "ROI_ERN_condition_amplitudes.tsv",
        sep="\t",
        index=False,
        na_rep="n/a"
    )
    index = [
        dict(wave_key=f"erp_{i}", participant=pid, Congruency=level, ACC=acc) for i,
        (pid, level, acc) in enumerate(compact_erps)
    ]
    pd.DataFrame(
        index,
        columns=["wave_key", "participant", "Congruency", "ACC"]
    ).to_csv(path_result / "group_subject_condition_erps.tsv", sep="\t", index=False)
    np.savez_compressed(
        path_result / "group_subject_condition_erps.npz",
        times_ms=times_ms if times_ms is not None else np.array([]),
        **{f"erp_{i}": wave for i, wave in enumerate(compact_erps.values())}
    )

    return cell_df, compact_erps, times_ms if times_ms is not None else np.array([])


def roi_based_analysis(
    participant_info: pd.DataFrame,
    data_roi_df: pd.DataFrame,
    path_result: Path | str,
    group_config: GroupParameters,
) -> tuple[MixedLMResultsWrapper | None, pd.DataFrame]:
    """Fit the unchanged model using validated participant-condition scalar values."""

    path_result = Path(path_result)
    path_result.mkdir(parents=True, exist_ok=True)
    checks, diff_df, result = [], pd.DataFrame(), None

    try:
        data_roi_df = validate_ern_summary(data_roi_df, participant_id=None)
        if (
            not {"participant_id", "PSWQ", "Sex"}.issubset(participant_info.columns)
            or not participant_info["participant_id"].is_unique
        ):
            raise ValueError("Participant metadata requires unique IDs, PSWQ and Sex.")
        participant_info = participant_info.rename(columns={"participant_id": "participant"}).copy()
        participant_info["PSWQ"] = pd.to_numeric(
            participant_info["PSWQ"],
            errors="coerce"
        )
        participant_info["Sex"] = participant_info["Sex"].astype("string").str.strip().replace(
            "",
            pd.NA
        )
        diff_df = data_roi_df.copy()
        diff_df = diff_df.merge(
            participant_info[["participant", "PSWQ", "Sex"]],
            on="participant",
            how="left",
            validate="many_to_one"
        )
        diff_df["exclusion_reason"] = ""
        diff_df.loc[diff_df.n_error.eq(0), "exclusion_reason"] += "no error trials; "
        diff_df.loc[
            diff_df.n_correct.eq(0),
            "exclusion_reason"
        ] += "no correct trials; "
        diff_df.loc[
            ~np.isfinite(diff_df["PSWQ"]),
            "exclusion_reason"
        ] += "missing/invalid PSWQ; "
        diff_df.loc[diff_df["Sex"].isna(), "exclusion_reason"] += "missing Sex; "
        diff_df["included"] = diff_df["exclusion_reason"].eq("") & np.isfinite(diff_df["ACC_diff"]) & diff_df["Congruency"].notna()
        diff_df.to_csv(
            path_result / "group_cell_checks.tsv",
            sep="\t",
            index=False,
            na_rep="n/a"
        )
        checks.append(dict(
            check="trial_and_cell_counts",
            passed=True,
            detail=f"trials={int(data_roi_df[['n_error', 'n_correct']].sum().sum())}, cells={len(diff_df)}, paired_complete_cells={int(diff_df.included.sum())}"
        ))

        if data_roi_df["n_error"].sum() == 0 or data_roi_df["n_correct"].sum() == 0:
            raise ValueError("Missing correct or error response level; Delta ERN model cannot be fitted.")
        diff_df = diff_df.loc[diff_df["included"]].copy()

        levels = sorted(diff_df["Congruency"].unique())
        checks.append(dict(
            check="observed_BIDS_congruency",
            passed=len(levels) >= 2,
            detail=str(levels)
        ))
        checks.append(dict(
            check="independent_participants",
            passed=diff_df["participant"].nunique() >= 2,
            detail=str(diff_df["participant"].nunique())
        ))
        checks.append(dict(
            check="PSWQ_variation",
            passed=diff_df["PSWQ"].nunique() >= 2,
            detail=str(diff_df["PSWQ"].nunique())
        ))
        checks.append(dict(
            check="Sex_levels",
            passed=diff_df["Sex"].nunique() >= 2,
            detail=str(diff_df["Sex"].value_counts().to_dict())
        ))
        checks.append(dict(
            check="repeated_measures",
            passed=bool((diff_df.groupby("participant").size() > 1).any()),
            detail=str(diff_df.groupby("participant").size().to_dict())
        ))

        if not all(row["passed"] for row in checks):
            raise ValueError("Pre-fit checks failed; see group_fit_checks.tsv.")

        diff_df["Congruency"] = pd.Categorical(diff_df["Congruency"], categories=levels)
        diff_df["Sex"] = diff_df["Sex"].astype("category").cat.remove_unused_categories()
        diff_df.to_csv(
            path_result / "LMM_input.tsv",
            sep="\t",
            index=False,
            na_rep="n/a"
        )

        model = sm.mixedlm(
            "ACC_diff ~ C(Congruency) + PSWQ + C(Sex)",
            data=diff_df,
            groups=diff_df["participant"],
            re_formula="1",
            missing="raise"
        )
        rank = np.linalg.matrix_rank(model.exog)
        checks.append(dict(
            check="design_matrix_rank",
            passed=rank == model.exog.shape[1] and len(diff_df) > rank,
            detail=f"rows={len(diff_df)}, columns={model.exog.shape[1]}, rank={rank}"
        ))
        if not checks[-1]["passed"]:
            raise ValueError("Rank-deficient design matrix or insufficient residual degrees of freedom.")
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            result = model.fit(reml=True)
        for warning in fit_warnings:
            checks.append(dict(
                check="fit_warning",
                passed=False,
                detail=str(warning.message)
            ))
        if (
            not result.converged
            or not np.isfinite(result.fe_params).all()
            or not np.isfinite(result.bse_fe).all()
        ):
            raise ValueError("Model did not converge or fixed-effect estimates/SEs are invalid.")
        print(result.summary())

        fe_names = result.fe_params.index
        confidence = result.conf_int().loc[fe_names]
        result_df = pd.DataFrame({
            "Effect": fe_names,
            "Beta": result.fe_params.values,
            "SE": result.bse_fe.values,
            "z": result.tvalues.loc[fe_names].values,
            "p": result.pvalues.loc[fe_names].values,
            "CI_lower": confidence[0].values,
            "CI_upper": confidence[1].values,
        })

        result.save(path_result / "LMM_model.pickle")
        result_df.to_csv(path_result / "LMM_results.tsv", sep="\t", index=False)

        significant = result_df.loc[result_df["Effect"].ne("Intercept") & result_df["p"].lt(group_config.alpha) & np.isfinite(result_df["p"])]
        for row in significant.itertuples(index=False):
            print(f"Significant effect: {row.Effect}\nBeta = {row.Beta:.4f}\np = {row.p:.4g}")
        if significant.empty:
            print(f"No significant fixed effects (p < {group_config.alpha:g}).")
        checks.append(dict(
            check="fit_status",
            passed=True,
            detail="completed; inspect any recorded fit warnings"
        ))
    except Exception as exc:
        checks.append(dict(
            check="fit_status",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}"
        ))
        print(f"Group analysis stopped: {exc}")
        result = None
    finally:
        pd.DataFrame(checks, columns=["check", "passed", "detail"]).to_csv(
            path_result / "group_fit_checks.tsv",
            sep="\t",
            index=False,
            na_rep="n/a"
        )

    return result, diff_df


def export_pswq_correlation(
    participant_info: pd.DataFrame,
    condition_df: pd.DataFrame,
    path_result: Path | str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Export unadjusted Pearson r using one pooled error-minus-correct value per person."""
    path_result = Path(path_result)
    path_result.mkdir(parents=True, exist_ok=True)
    summary = dict(
        method="Pearson",
        n=0,
        r=np.nan,
        p=np.nan,
        ci_lower=np.nan,
        ci_upper=np.nan,
        confidence_level=1-alpha,
        status="failed",
        reason="",
        aggregation="pooled retained trials; error minus correct; one row per participant"
    )
    pooled = pd.DataFrame(columns=[
        "participant",
        "n_error",
        "n_correct",
        "delta_ern_uv",
        "PSWQ",
        "included",
        "reason"
    ])

    try:
        if (
            not {"participant_id", "PSWQ"}.issubset(participant_info)
            or not participant_info.participant_id.is_unique
        ):
            raise ValueError("Correlation requires unique participant IDs and PSWQ.")
        table = validate_ern_summary(condition_df, None)
        for count, mean, total in (
            ("n_error", "mean_ern_uv", "error_sum"),
            ("n_correct", "mean_crn_uv", "correct_sum")
        ):
            table[mean] = pd.to_numeric(table[mean], errors="raise")
            if not np.isfinite(table.loc[table[count].gt(0), mean]).all():
                raise ValueError("Missing amplitude for observed correlation trials.")
            table[total] = table[count] * table[mean].fillna(0)
        pooled = table.groupby("participant")[[
            "n_error",
            "n_correct",
            "error_sum",
            "correct_sum"
        ]].sum().reset_index()
        pooled["delta_ern_uv"] = (pooled.error_sum / pooled.n_error.replace(
            0,
            np.nan
        ) - pooled.correct_sum / pooled.n_correct.replace( 0, np.nan ))
        metadata = participant_info[[
            "participant_id",
            "PSWQ"
        ]].rename(columns={"participant_id": "participant"}).copy()
        metadata["PSWQ"] = pd.to_numeric(metadata["PSWQ"], errors="coerce")
        pooled = pooled.merge(
            metadata,
            on="participant",
            how="left",
            validate="one_to_one"
        )
        pooled["reason"] = ""
        pooled.loc[pooled.n_error.eq(0), "reason"] += "no error trials; "
        pooled.loc[pooled.n_correct.eq(0), "reason"] += "no correct trials; "
        pooled.loc[~np.isfinite(pooled.PSWQ), "reason"] += "missing/invalid PSWQ; "
        pooled["included"] = pooled.reason.eq("") & np.isfinite(pooled.delta_ern_uv)
        selected = pooled.loc[pooled.included]
        summary["n"] = len(selected)
        if (
            len(selected) < 3
            or selected.PSWQ.nunique() < 2
            or selected.delta_ern_uv.nunique() < 2
        ):
            summary.update(
                status="insufficient_data",
                reason="Need at least three participants and variation in PSWQ/Delta ERN."
            )
        else:
            result = pearsonr(selected.PSWQ, selected.delta_ern_uv)
            interval = result.confidence_interval(confidence_level=1-alpha)
            if not np.isfinite([result.statistic, result.pvalue]).all():
                raise ValueError("Correlation estimate is not finite.")
            summary.update(
                status="completed",
                r=float(result.statistic),
                p=float(result.pvalue),
                ci_lower=float(interval.low),
                ci_upper=float(interval.high)
            )
    except Exception as exc:
        summary.update(status="failed", reason=f"{type(exc).__name__}: {exc}")

    pooled.to_csv(
        path_result / "PSWQ_correlation_input.tsv",
        sep="\t",
        index=False,
        na_rep="n/a"
    )
    pd.DataFrame([summary]).to_csv(
        path_result / "PSWQ_correlation.tsv",
        sep="\t",
        index=False,
        na_rep="n/a"
    )

    return summary


def main(
    participant_info: pd.DataFrame,
    group_config: GroupParameters,
    folder_data: Path | str,
    path_result: Path | str,
) -> dict[str, Any]:
    """Run group data loading, correlation, plotting and LMM with stage status tracking."""
    path_result = Path(path_result)
    path_result.mkdir(parents=True, exist_ok=True)
    state_path = path_result / "group_analysis_state.json"
    state = dict(
        status="running",
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_status="not_run",
        correlation_status="not_run",
        plot_status="not_run",
        outputs=[],
        error=""
    )
    write_run_state(state_path, state)

    try:
        cells, erps, times = read_roi_based(
            group_config,
            participant_info,
            folder_data,
            path_result
        )
        state["outputs"].extend([
            "group_input_manifest.tsv",
            "ROI_based_trial_amplitudes.tsv",
            "ROI_ERN_condition_amplitudes.tsv",
            "group_subject_condition_erps.tsv",
            "group_subject_condition_erps.npz",
            "group_analysis_config.json"
        ])
        correlation = export_pswq_correlation(
            participant_info,
            cells,
            path_result,
            group_config.alpha
        )
        state["correlation_status"] = correlation["status"]
        state["outputs"].extend(["PSWQ_correlation.tsv", "PSWQ_correlation_input.tsv"])
        if erps:
            try:
                plot_roi_erps(compute_roi_erps(erps), times, group_config, path_result)
                state["plot_status"] = "completed"
                state["outputs"].append("ROI_ERP_comparison.png")
            except Exception as exc:
                state.update(plot_status="failed", plot_error=str(exc))
                print(f"ERP plotting failed: {exc}; continuing with model analysis.")
        model, _ = roi_based_analysis(
            participant_info,
            cells,
            path_result,
            group_config
        )
        state["model_status"] = "completed" if model is not None else "failed"
        state["outputs"].append("group_fit_checks.tsv")
        if model is not None:
            state["outputs"].extend([
                "LMM_model.pickle",
                "LMM_results.tsv",
                "LMM_input.tsv",
                "group_cell_checks.tsv"
            ])
        successful = model is not None and correlation["status"] == "completed"
        state["status"] = (
            "no_inputs"
            if cells.empty else "completed"
            if successful
            and state["plot_status"] == "completed" else "partial"
        )
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_run_state(state_path, state)

    return state
