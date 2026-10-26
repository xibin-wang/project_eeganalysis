# ManyAnalyst ERN Analysis

Python workflow for the ManyAnalyst Flanker investigation described in [Information_Subsample.docx](Information_Subsample.docx). It validates dataset-specific BIDS metadata, preprocesses each participant independently, performs ICA and response epoching, exports individual ERN measures, and runs a group mixed-effects analysis.

The workflow supports automatic participant processing with a bounded thread pool and an optional manual-review mode. Completed individual outputs are validated before reuse. Group analysis prefers individual ERN CSV files and retains participant-condition ERP means rather than concatenating all trial waveforms.

This README describes the current scripts. Implemented exports are distinguished from completed scientific deliverables: full real-data validation and the three-participant reproducibility submission remain necessary.

## 1. Project requirements and analysis definitions

The project provides 20 harmonized datasets from 10 laboratories to develop an analysis intended for approximately 700 participants. The supplied EEG is sampled at 250 Hz, referenced to FCZ, and contains the relevant task segments; removed tasks and breaks are marked by `boundary` events. The shared Flanker data represent the experimenter-absent condition.

The hypotheses are:

1. Error-minus-correct ERN differences are associated with trait worry measured by PSWQ.
2. The difference is stronger for females than males.
3. The difference is stronger for 100% than 0% congruency.

For each retained trial, the pipeline averages the configured ROI channels and samples in the ERN window. Defaults are FCz, 0-100 ms after the response, in microvolts.

```text
ERN       = mean amplitude on incorrect trials (ACC = 0)
CRN       = mean amplitude on correct trials   (ACC = 1)
Delta ERN = ERN - CRN
```

The overall value pools trials across conditions separately for errors and correct responses. It is not the unweighted average of the four condition differences. ERN = -3 uV and CRN = 1 uV gives Delta ERN = -4 uV. A stronger negative response is numerically more negative.

| Project instruction | Implementation or remaining delivery |
| --- | --- |
| One file per participant with overall and four condition-specific error-minus-correct values | `<ID>_ern_summary.csv`, five rows. |
| Independent participant processing and avoidance of redundant processing | Separate outputs, completion validation, and automatic threaded batches. |
| Continue when individual records fail | BIDS eligibility reporting and participant-level exception isolation in both modes. |
| Suggested participant preprocessing summary | `<ID>_analysis_summary.txt`, log, timings, and detailed QC tables. |
| Clear paths, dependencies, custom functions | Configuration/install instructions below; custom helpers are in `scripts/`. |
| Final preprocessed EEG for three participants | Export implemented; select and verify three actual completed datasets for submission. |
| Correlation estimate requested after submission | `PSWQ_correlation.tsv`: unadjusted Pearson r using one pooled Delta ERN per participant, separate from the LMM coefficient. |

The document permits analysts to choose preprocessing methods. The current ROI, filters, baseline, thresholds and ERN window are analysis choices, not parameters imposed by the project.

## 2. Paths to configure before running

| Setting | Location | Default |
| --- | --- | --- |
| Project root | `PathConstants.FOLDER_ROOT` in `scripts/constants.py` | Parent of `scripts/`, inferred from `__file__`. |
| Original dataset | `source` in `bids_correction.py:main()` | `<root>/eeglabset/` |
| Corrected working copy | `target` in `bids_correction.py:main()` | `<root>/eeglabset_fix/` |
| Active dataset | `prepared_data()` / first `bids_correction()` | Pinned by `bids_preparation_state.json` after the first completed preparation. |
| Reports and group outputs | `PathConstants.FOLDER_RESULT` | `<project_root>/results/`; configurable independently of the dataset. |
| Generated EEG/ICA files | Participant `eeg/` directory | `<active_dataset>/<ID>/eeg/` |

On the first preparation, any required swap/duration repair causes a copy to `eeglabset_fix`; otherwise the original directory becomes active. Reports always go to `PathConstants.FOLDER_RESULT`. Every report-producing stage receives the same explicit `path_result`, including period QC, artifact review, ERN, logs and group exports. EEG/ICA/FIF files remain in the active participant EEG directory. Summary links are calculated relative to the actual report location, including custom output roots.

Later launches call `prepared_data()` and bypass `bids_correction()` when the persisted preparation is complete. The check does not rescan EEG/events or recopy data. A complete earlier BIDS report in a known dataset directory can initialize the marker without correction. Interrupted preparations have no completed marker and may run again. If a completed marker points to a missing directory/report, restore it or explicitly set `ModeConfiguration.rebuild_bids=True`.

`rebuild_bids=True` deliberately reruns preparation from `eeglabset`, potentially copying source files over the working directory. After fixing a BIDS-ineligible participant, fix the source files and rebuild to refresh eligibility, then return the flag to `False`. Normal failures in individual preprocessing do not require rebuilding BIDS. Use individual force-reprocessing or automatic source-change detection for those participants.

The supported production layout is one Flanker recording per participant without session/run entities:

```text
eeglabset/
  participants.tsv
  participants.json
  dataset_description.json
  channels.tsv
  electrodes.tsv
  electrodes.json
  sub-<ID>/
    eeg/
      sub-<ID>_task-Flanker_eeg.set
      sub-<ID>_task-Flanker_eeg.json
      sub-<ID>_task-Flanker_events.tsv
      sub-<ID>_task-Flanker_events.json
```

Keep external data files referenced by `.set` files alongside them. The correction code assumes dataset-level channel/electrode tables. General BIDS session/run discovery is not implemented.

Required metadata:

- `participants.tsv`: unique `participant_id`, `PSWQ`, and `Sex`; IDs include `sub-` and match directories.
- Event tables: `onset`, `value`, and `sample` when onset recovery is required; `duration` is checked/repaired.
- Response metadata: `Event`, `Trial`, `ACC`, `Congruency`; missing RT is recorded as unavailable.
- EEG: correct channel types and usable head-coordinate electrode locations for spatial QC.

`ACC` is `0 = incorrect`, `1 = correct`. Described congruency codes are `0`, `0.33`, `0.66`, `1`. Response Trial IDs must be unique within a recording for group matching. The epoch builder, artifact protection and shared BIDS matcher all recognize Response labels after trimming whitespace and ignoring case.

## 3. Environment and installation

Use Python 3.12 or newer; existing f-strings require its syntax. Example PowerShell setup from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install mne mne-bids mne-icalabel onnxruntime numpy pandas scipy matplotlib statsmodels
```

For manual review, also install:

```powershell
.\.venv\Scripts\python.exe -m pip install mne-qt-browser PyQt6
```

MNE must support `Annotations.extras`. ICLabel uses the ONNX backend. A desktop/Qt environment is needed for `semi_auto=True`; automatic mode does not request review windows. Group figures are saved without `plt.show()`.

These versions were observed locally when this README was updated. This is an environment record, not a supplied lockfile or a claim that a fresh installation has been validated.

| Component | Observed version |
| --- | --- |
| Python | 3.12.3 |
| MNE / MNE-BIDS | 1.13.2 / 0.20.0 |
| mne-icalabel / onnxruntime | 0.10.0 / 1.30.0 |
| NumPy / pandas | 2.5.4 / 3.0.6 |
| SciPy / Matplotlib / statsmodels | 1.18.1 / 3.11.2 / 0.15.0 |
| mne-qt-browser / PyQt6 | 0.7.5 / 6.11.0 |

No dependency lockfile or automated regression suite is supplied. `scripts/test.py` is a guarded exploratory entry point using production epoching; it is not a test runner.

## 4. Configuration

Scientific parameters are primarily defined in `scripts/constants.py`.

| Dataclass / setting | Default and purpose |
| --- | --- |
| `ModeConfiguration.semi_auto` | `False`: automatic; `True`: manual review and optional ICA-rerun interaction. |
| `ModeConfiguration.max_workers` | `2`: concurrent participants in automatic all-participant mode; positive integer required. |
| `ModeConfiguration.rebuild_bids` | `False`: reuse prepared data; `True`: explicitly rebuild preparation from source. |
| `ModeConfiguration.force_reprocess` | `False`: validate/reuse completed individual results; `True`: rerun selected/all individuals. |
| `FilterParameters` | 0.1-80 Hz FIR band-pass; 50 Hz screening at 8 dB, fraction 0.20, 4 s windows, 2 s steps. |
| `PeriodQCConfig` zero detection | Tolerance 0 V, minimum period 2 s; tolerance is converted to uV for comparison. |
| `PeriodQCConfig` PTP | 200 uV; 0.2 s windows, 0.1 s steps, minimum sustained support 3 s. |
| `PeriodQCConfig` spatial criteria | 8 nearest neighbors; absolute correlation 0.8; simultaneous abnormality threshold 0.5. |
| `PeriodQCConfig` duration criteria | Shared intervals at least 1% of recording become BAD; channel assessment above 30%; predominantly shared threshold 0.5. |
| `ICAPrep` candidate/severe thresholds | Candidate PTP 200 uV or power score 4; severe PTP 250 uV and power score 5 in the same channel/window; at least 1 severe channel. |
| `ICAPrep` timing/bands | 1 s windows, 0.1 s steps, persistence 0.20 s; bands 30-45 and 55-75 Hz. |
| `ICAPrep` repetition | Similarity 0.90; 3 matches including self; duration ratio at most 1.5; up to 8 comparison channels. |
| `ICAPrep` alignment | Intervals above 2 s resampled to 200 points; maximum shift 0.08 s; overlap at least 0.80. |
| `ICAPrep` response protection | -0.5 to +0.5 s; enabled. Automatic rejection of unique non-severe candidates disabled. |
| `ICAParameters` | 1 Hz training high-pass; ICLabel threshold 0.90; rerun suggestion above 50% `other`. |
| `GroupParameters` | ROI `("FCz",)`, ERN window `(0, 0.1)` s, alpha 0.05. |

ROI names are case-sensitive. Mastoids are excluded from period/candidate screening by default. Average reference also excludes `VOGbelow` and bad channels. No bad-channel interpolation is implemented.

Preprocessing constructs period/detection configurations from these defaults. Group configuration checks reconstruct the current defaults and compare them with each participant's recorded configuration.

## 5. Running the workflow

```powershell
.\.venv\Scripts\python.exe scripts/main.py
```

BIDS is prepared once and its persisted directory/report is reused on subsequent launches. Only participants marked `status == "ok"` and `analysis_eligible == True` enter the menu.

| Option | Behavior |
| --- | --- |
| `1` | Complete one eligible participant, including epochs and ERN. Select the participant ID or displayed zero-based index. |
| `2` | Complete all eligible participants: threaded in automatic mode, serial in manual mode. |
| `3` | Leave the individual menu and attempt group analysis using completed available results. This is not an exit-without-analysis option. |
| `4` | Force rerun one selected participant even when existing outputs validate. |

After option 1 or 2, the menu reappears. Select 3 to proceed to group analysis. The menu remains interactive with `semi_auto=False`; there is no command-line argument interface for unattended scheduling.

Invalid menu choices, nonnumeric indices, negative indices and out-of-range indices are rejected without terminating analysis. `preprocessing_batch_status.tsv` records eligible/pending, completed, failed and BIDS-ineligible participants; it is refreshed after individual outcomes.

For direct Python use, add `scripts/` to `sys.path` and import its modules. Supply `epoch_config=ICAPrep()` when calling `analysis_individual()` to complete epochs and ERN. Without it, the function returns pre-ICA data, training data and ICA, with summary status `completed_before_epochs`; that is not a completed final analysis. Production `main.py` supplies the epoch configuration in every individual mode.

## 6. BIDS correction and eligibility

`scripts/bids_correction.py` inspects source files before choosing whether to update a working copy.

1. Identify TSV schemas: channels require `name/type/units`; electrodes require `name/x/y/z`. Swap contents only when the schemas are reversed. Unidentifiable global schemas stop correction.
2. Inspect event tables for a missing duration column or missing values. Record source-table errors per participant.
3. If any swap/repair is needed, copy all source data to `eeglabset_fix` and repair the copy.
4. Read SET sampling frequency and original annotations, then check event counts, normalized codes and one-to-one onset matching. Tolerance is `max(2e-6, 1e-6 / sfreq)` seconds.
5. Keep matching onsets; during required duration repair, try `sample / sfreq` only if existing onsets do not match. No universal one-sample shift is applied.
6. Restore only missing durations; existing values must agree with SET. Unknown/negative SET durations fail the record. Unmatched values are not inferred to be zero.
7. Allow exactly one additional first SET `boundary` with negative onset as the sole count-mismatch exception.
8. Re-read using `read_raw_bids()` and compare Raw codes, counts, onsets and durations after MNE's range handling. Update onset/duration JSON units after repair when that sidecar exists.

Comparison normalizes marker case, whitespace and the leading `x`. Original SET annotations are read separately because Raw loading can clip out-of-range boundaries.

`results/inspect/bids_correction_summary.tsv` records swaps, duration/onset repairs, boundary exceptions, status, `analysis_eligible` and errors. A record becomes eligible only after final validation. Failed event edits are restored from backups. Restoration failures are recorded without cancelling other participants. The report is checkpointed after each participant; preparation is marked completed only after the entire pass finishes. Global configuration/copy/output failures may stop preparation.

## 7. Individual preprocessing and artifact preparation

Processing order:

```text
Read BIDS and normalize boundary labels
-> period/bad-channel QC on the original reference
-> restore FCz
-> band-pass and conditional line-noise removal
-> average reference excluding bad channels
-> local artifact detection and classification
-> optional manual review
-> ICA fitting and ICLabel evaluation
-> response epochs, ICA application and baseline
-> individual ERN CSV and analysis reports
```

### 7.1 Input, boundaries and early QC

EEG is loaded through `read_raw_bids()`. Boundary/edge descriptions become `BAD_boundary` on the shared input. Filtering explicitly skips `edge`, `bad` and `boundary`; detectors exclude crossing windows and relevant positive-duration intervals.

Preprocessing does not reinterpret boundary onset/duration values. Positive durations are treated as excluded intervals, so the meaning of EEGLAB boundary durations remains a dataset-specific point to verify. Renaming a boundary does not establish that its duration represents missing samples within the retained recording.

Period QC runs before filtering/reference to preserve zero-dropout patterns and identify bad channels before reference. It uses eligible EEG channels and head-coordinate electrode positions. Existing channel-specific BAD intervals are masked only for the affected channels during this initial QC.

### 7.2 Zero and sustained PTP periods

Zero detection requires at least 2 s satisfying the absolute tolerance. `zero_tolerance_v` is in volts and multiplied by `1e6` for comparison with uV data. The default detects exact zeros; this is not a general detector for every nonzero flat signal.

PTP screening uses 0.2 s windows every 0.1 s. A channel/window is abnormal if PTP exceeds 200 uV **or** exceeds neighboring-channel mean PTP by three population standard deviations. Each flagged window contributes one step near its center; contiguous support must last 3 s. Zero and PTP intervals are merged per channel.

Eight nearest available EEG channels are chosen by Euclidean distance. A neighboring abnormality is shared when it overlaps at least 50% of the interval and satisfies zero co-occurrence or absolute waveform correlation of at least 0.8. Both positive and negative correlations qualify.

| Decision | Current rule |
| --- | --- |
| Shared BAD period | Shared interval occupying at least 1% of full recording duration. |
| REVIEW period | Shorter shared intervals and all independent intervals; independent labels retain channel scope. |
| Bad channel | More than 30% total abnormal duration, predominantly independent. |
| Unusable record | More than 30% total abnormal duration, predominantly shared. Save period report, then stop the participant before ICA. |

Fractions use the full recording sample count, not just unannotated time. Local annotations attached solely to already excluded bad channels are removed from the downstream input; detections remain in the period report.

### 7.3 FCz, filters and line noise

FCz is restored as a zero-valued original reference channel before re-reference, at `[0, 0.0388, 0.0922]` m in head coordinates. Existing FCz is not added twice; adding it requires a suitable montage.

A zero-phase FIR band-pass applies 0.1-80 Hz by default, with `0 < highpass < lowpass < Nyquist` required. It respects the explicit skip annotations.

When 50 Hz lies between passband limits, screening uses eligible 4 s windows with a 2 s step. Welch PSD segments are at most 2 s. Peak PSD within 49.5-50.5 Hz is compared with median background PSD in 45-48 and 52-55 Hz:

```text
ratio_db = 10 * log10((line_peak + tiny) / (background_median + tiny))
```

Line noise is detected if any included EEG channel exceeds 8 dB in at least 20% of valid windows. BAD/REVIEW periods, boundaries and nonfinite windows are excluded. Invalid sample lengths, unresolved frequency bands or no usable windows raise errors. Detection triggers a 50 Hz `spectrum_fit` notch, also respecting skip annotations.

Average reference uses valid EEG channels excluding `MASTl`, `MASTr`, `VOGbelow` and marked bad channels. An empty reference set raises an error.

### 7.4 Local candidate and severe detection

After reference, candidate screening uses 1 s windows every 0.1 s. BAD and REVIEW periods are withheld from this detector; boundary-crossing and nonfinite windows are skipped. High-frequency power integrates Welch PSD separately over 30-45 and 55-75 Hz.

The robust baseline is participant-, channel- and band-specific:

```text
log_power = 10 * log10(max(power, 1e-20))
robust_sigma = max(1.4826 * MAD(log_power), 1.0)
z_power = (log_power - median(log_power)) / robust_sigma
```

| Detection | Default rule |
| --- | --- |
| Candidate | PTP > 200 uV OR power score > 4 in an eligible channel. |
| Severe | PTP > 250 uV AND power score > 5 in the same channel/window; at least one such channel. |

Window-center support must persist for 0.20 s. Accepted intervals expand toward their contributing windows and merge without extra positive gaps by default. Boundaries describe window support rather than exact sample-level artifact transitions.

### 7.5 Repetition and response protection

Waveforms are demeaned per channel. Short intervals retain their samples; intervals longer than 2 s are resampled to 200 points. Only intervals in the same short/long category and within the duration-ratio limit are compared.

Similarity combines spatial amplitude-distribution agreement with signed temporal correlation over up to eight strong shared channels. Shifts are limited to 0.08 s and overlap must cover 80% of the longer waveform. Similarity at least 0.90 counts as a match. Three matches include self, so two other matching candidates meet the default repetition criterion.

Response protection recognizes trimmed, case-insensitive `Event` metadata. Candidates overlapping a -0.5 to +0.5 s response window are ERN-related. Protection does not undo BAD labels from earlier period QC.

| Candidate | Label with current defaults |
| --- | --- |
| Severe, nonrepetitive, not ERN-related | `BAD_severe` |
| Severe and repetitive and/or ERN-related | Corresponding `REVIEW_severe_*` label. |
| Non-severe and repetitive | Corresponding `REVIEW_repetitive*` label. |
| Non-severe and unique | `REVIEW_unique*`; non-ERN unique candidates become BAD only with `auto_reject_unique=True`. |

New labels are appended directly to the Raw copy, preserving timing for nonzero `first_samp`. The artifact-stage main function copies the annotated Raw instead of setting its annotations again.

### 7.6 Automatic versus manual decisions

| Mode | Behavior |
| --- | --- |
| `semi_auto=False` | No review GUI or ICA-rerun question. BAD remains excluded; REVIEW and its data remain retained, explicitly marked as not manually reviewed. |
| `semi_auto=True` | Blocking artifact browser. Resolve REVIEW to KEEP/global BAD, or mark the affected channel bad. Pending REVIEW/channel-scoped BAD decisions reopen the browser. |

A newly bad channel's local BAD/REVIEW annotations are removed. Channel-specific BAD on an otherwise retained channel is not silently accepted as global exclusion. Removing a bad-channel decision made before reference requires restarting preprocessing. Adding bad channels in the initial manual review causes reference to be recalculated before ICA.

`artifact_results_reviewed.tsv` is generated in both modes. In automatic mode `manual_review_performed=False`; its filename alone does not imply human approval.

## 8. ICA, epochs and individual ERN

ICA uses extended Infomax, `n_components=None`, `random_state=42`, and automatic maximum iterations on non-bad EEG channels. It defaults to a separate 1 Hz high-pass training copy; final epochs use the original 0.1 Hz analysis data. Training filters respect skip annotations.

ICLabel uses training samples remaining after BAD segments are omitted and concatenated. A component is excluded when its highest class probability is at least 0.90 and its predicted class is neither `brain` nor `other`. Probabilities and exclusion flags are saved per participant.

Above 50% `other`, manual mode offers review/rerun; automatic mode keeps the current solution without prompting. The final ICA file is saved after exclusion decisions. Initial and repeated ICA review both use `review_for_ica()`: unresolved annotations are checked, new bad channels trigger re-reference, and reviewed QC tables are refreshed. Invalid y/n input repeats the question rather than unnecessarily rerunning ICA.

Epochs span -0.5 to +0.5 s, retain metadata, and reject BAD annotations. ICA is applied before baseline correction from epoch start through 0 s. No additional amplitude-based epoch rejection threshold is configured.

`make_epochs()` matches response metadata to BIDS events by unique `Trial`, verifies ACC, and preserves exact numeric Congruency codes in saved epochs and individual ERN summaries, using the same matcher as group analysis. Condition rows cover all response levels present in the source BIDS events, including levels with no retained epochs. Existing summaries are checked against BIDS-matched epoch metadata before skipping a participant.

| CSV field | Meaning |
| --- | --- |
| `participant_id` | Full participant ID. |
| `condition` | `all` and `congruency_<value>` for each source BIDS response level. |
| `Congruency` | Numeric condition; `n/a` for pooled row. |
| `n_error`, `n_correct` | Retained error/correct trial counts. |
| `mean_ern_uv`, `mean_crn_uv` | Error/correct mean amplitudes. |
| `delta_ern_uv` | Error minus correct. |
| `roi` | Comma-separated channel names. |
| `tmin_s`, `tmax_s` | Measurement window. |

Missing response types produce `n/a` amplitudes/differences with zero counts, not fabricated zeros. There is no minimum error-trial threshold beyond having observations; numerical availability does not establish reliability.

## 9. Completion, reruns, logs and timing

Before processing, `participant_log()` uses the shared `load_completed_epochs()` helper to check `<ID>_analysis_state.json`, preprocessing settings, final epochs and BIDS response metadata. Group input loading uses the same helper; `read_analysis_state()` supplies the basic completion-state check. The text summary is a human-readable report and is not parsed for new runs. Legacy text summaries are not supported: without a valid new state JSON, individual analysis runs again.

Skipping requires:

- matching participant, completed status and no recorded error;
- valid configuration and positive final epoch count;
- matching filter, period QC, artifact-detection, ICA and epoch settings;
- unchanged source SET/FDT sizes/modification times and matching SHA-256 content fingerprints for preprocessing metadata;
- nonempty pre-ICA EEG, ICA, final epochs, ERN CSV and compact text summary;
- valid CSV participant/conditions/counts/amplitudes, current ERN ROI/window and BIDS-matched metadata;
- agreement of saved epochs, state epoch count and ERN trial counts.

Missing/invalid outputs, failed/running state or changed scientific settings trigger the complete individual workflow from BIDS. Zero-trial `n/a` values are valid and do not by themselves cause repeated processing. Use menu option 4, pass `force=True` to `analysis_individual()`, or set `force_reprocess=True` to rerun despite valid outputs. Switching previously automatic results to manual review also triggers reprocessing.

Input comparison tracks participant EEG-directory TSV/JSON files and dataset-level channel, electrode, coordinate, inherited EEG and event metadata. Small metadata files are compared by SHA-256 content; changing content is detected even if size and modification time are preserved. Touching an unchanged metadata file does not require reprocessing. SET/FDT files use size and nanosecond modification time without reading their full contents; an EEG replacement preserving both is outside this lightweight check and can be handled with force reprocessing. Added/deleted tracked files also invalidate reuse, and messages identify changed files. Group and individual reuse use the same check before opening saved epochs.

`participants.tsv` PSWQ/Sex values are read afresh for group statistics and do not invalidate EEG preprocessing. Comparisons refer to the pinned active dataset, not a source directory that is no longer being read. The signature format has changed; results carrying older stat-only records need one new preprocessing run. Inputs are checked again before marking a new individual run completed; edits during processing leave a failed state requiring a rerun.

State is written as `running` before analysis, and replaced with `completed` only after outputs and the text report are written. Ordinary failures record the stage and error; a hard interruption leaves `running`. JSON writes use a temporary file and replacement. Full traceback and stage timing remain in their own files.

The compact text summary directly reports channel removal, bad-channel exclusions, interpolation (not performed), FCz restoration, ICA component removal, artifact review status/BAD duration, epoch counts and retained incorrect/correct counts for each condition and overall. It includes key preprocessing/ERN settings, software versions and relative output paths. Full configuration, per-stage snapshots, QC tables, ICLabel probabilities and ERN amplitudes are not embedded; ERN amplitudes remain in the separate per-participant CSV.

Batch failures do not cancel other participants in either mode. Current-session failures are excluded from group analysis. There is no indefinite automatic retry within one batch.

| File in `results/<ID>/` | Contents |
| --- | --- |
| `<ID>_analysis.log` | Script messages with UTC time and participant ID, stage start/end messages and exception traceback. |
| `<ID>_step_timings.tsv` | Stage, start/end UTC, elapsed seconds, status. |
| `<ID>_analysis_summary.txt` | Compact standalone preprocessing report, channel/ICA decisions and retained trial counts by condition. |
| `<ID>_analysis_state.json` | Machine status, error/stage, configuration, source EEG sizes/timestamps, metadata content fingerprints, software versions and final epoch information. |

Elapsed durations use `time.perf_counter()`; wall-clock labels use `time` in UTC. Nested durations overlap and must not be summed as independent elapsed time. Context-isolated logs support threaded participants. Each run overwrites its previous log/timings/summary/state rather than maintaining versioned history. Reports use snapshots from the current attempt; an epoch-count table is included only after the epoch/ERN stage succeeds.

Project `print()` calls are timestamped. MNE's own logging and warnings keep their library formatting and are not globally redirected into participant logs.

## 10. Group analysis and reuse of individual results

After menu option 3, `main.py` reads basic completed machine state and notes CSV availability; shared epoch/source checks run when the group reader loads each participant. Missing CSV alone can use group fallback; failed/running state or missing required EEG/events cannot. Editing the new text report does not change eligibility.

### 10.1 Configuration and input validation

`read_roi_based()` processes one participant at a time. Recorded filter, period QC, artifact-detection, ICA and epoch configurations must match the current defaults. Incompatible preprocessing is excluded and requires a new individual analysis; differently processed participants are not silently pooled.

The shared `load_completed_epochs()` checks preprocessing configuration and final epoch count, sampling rate and baseline against saved state, then matches metadata to BIDS response events. `participant_paths()` supplies the same file names to individual and group entry points. ROI availability, full ERN-window coverage, finite ROI samples and the common epoch time axis are checked. `match_bids_responses()` matches retained metadata to BIDS Response rows by unique `Trial`, verifies ACC, and keeps exact source Congruency values without nearest-level mapping.

Configuration and input checks are saved. Events, state, summary and ERN CSV receive SHA-256 fingerprints when present. Epoch files are recorded by absolute path, size and nanosecond modification time instead of a full hash, avoiding an extra full EEG read solely for hashing.

### 10.2 CSV-first inputs and bounded waveform storage

Individual CSV values are used directly when participant, ROI, window, source condition codes, trial counts and amplitude validity checks pass. No duplicate ERN table is embedded in or compared with the text summary. Counts are checked against retained BIDS-matched trials. The manifest records `model_source=personal_ern_csv`.

A missing, malformed or ROI/window/condition-incompatible CSV triggers recalculation for that participant only from compatible epochs: `model_source=epochs_recomputed` plus the cache reason. This fallback writes group outputs; it does not repair the individual CSV required for submission.

Epochs open with `preload=False`. ROI samples are processed one participant at a time. Only condition/ACC mean waveforms survive the iteration: no all-participant trial-waveform concatenation is retained. The model uses a small participant-condition scalar table. Trial amplitudes are appended to TSV per participant, not assembled globally in memory.

Plots still require reading included participants' ROI samples; CSV reuse does not eliminate waveform I/O. Subject-condition means are saved in compressed NPZ with a TSV index.

### 10.3 Model and diagnostics

The model remains a random-intercept LMM using REML:

```text
ACC_diff ~ C(Congruency) + PSWQ + C(Sex)
random intercept grouped by participant
```

Cells without errors, correct responses, finite PSWQ, sex or a usable difference are excluded and recorded. Pre-fit checks require at least two participants, two observed congruency levels, PSWQ variation, two sex levels, repeated measurements, full fixed-effects design rank and more observations than its rank.

Only observed BIDS levels become model categories. The lowest observed congruency is the reference; 0 and 1 must both be available to interpret the 100%-minus-0% hypothesis. With F/M labels, F is normally the sex reference: inspect effect names and signs.

Warnings, convergence and finite fixed-effect estimates/SEs are checked. Failures are caught and written to `group_fit_checks.tsv`; warnings remain recorded even if a converged model is exported. Consult the final `fit_status`. Older model/figure files are not automatically deleted when a later attempt fails.

ERP plots are scheduled by the group entry point separately from model fitting, independent of significance, and saved and closed without a GUI. A plotting failure is reported without preventing the model attempt. `roi_based_analysis()` accepts participant metadata, condition amplitudes, the output directory and group settings; it does not take waveforms or a time axis. Descriptive curves use eligible EEG data; model cells undergo additional pairing/covariate filtering, so their sample sizes can differ. Correct/error curves can also have different contributing participant counts, displayed in labels.

Exports include fixed-effect estimates, SEs, z statistics, p values and confidence intervals. The code does not currently export a random-slope model, residual/influence diagnostic report, precision weighting by trial counts or a multiple-testing adjustment. These require explicit analytical decisions. The project does not mandate a PSWQ-by-congruency interaction.

### 10.4 Pooled Delta ERN and PSWQ correlation

`PSWQ_correlation.tsv` reports unadjusted Pearson r, two-sided p, participant N, confidence interval (`1 - alpha`), status and the aggregation definition. It uses exactly one observation per participant. Correct and incorrect condition means are weighted separately by their retained trial counts, then pooled error minus pooled correct is calculated. The four condition differences are not averaged equally and are not treated as independent participants.

`PSWQ_correlation_input.tsv` records pooled amplitudes/counts, PSWQ, inclusion and exclusion reasons. Missing Sex does not exclude a participant from this unadjusted correlation. Participants without both response types or finite PSWQ are excluded. At least three usable participants and variation in both variables are required; otherwise the summary records `insufficient_data`. Correlation and LMM can therefore use different samples.

Each group attempt starts `group_analysis_state.json` as `running` and ends as `completed`, `partial`, `failed` or `no_inputs`. Model, correlation and plot statuses are separate. The `outputs` list identifies files generated/accepted in this attempt; older files may remain on disk and must not be treated as current if absent from this list. Even an empty group attempt refreshes state and diagnostic outputs. A hard interruption leaves `running`.

## 11. Output reference

`<reports>` is `PathConstants.FOLDER_RESULT` (default `<project_root>/results`); `<active>` is the pinned EEG dataset directory. All report paths below use these definitions.

`<active>` is the pinned EEG dataset directory. `<reports>` is `PathConstants.FOLDER_RESULT`, default `<project_root>/results`. `<ID>` includes `sub-`.

### Individual and inspection files

| Location | Files and purpose |
| --- | --- |
| `<reports>/inspect/` | `bids_correction_summary.tsv`: repairs, eligibility, errors; `bids_preparation_state.json`: completed preparation and pinned dataset. |
| `<reports>/<ID>/` | `bad_period_results.tsv`: early channel/period QC. |
| `<reports>/<ID>/` | `artifact_results_automatic.tsv`, `artifact_summary_automatic.json`: candidate classifications and totals. |
| `<reports>/<ID>/` | `artifact_results_reviewed.tsv`, `artifact_summary_reviewed.json`: initial review/pass-through decisions, mode and pending counts. |
| `<reports>/<ID>/` | `<ID>_iclabel.tsv`: labels, probabilities and exclusion flags. |
| `<reports>/<ID>/` | `<ID>_ern_summary.csv`: five-row ERN/CRN/Delta ERN output. |
| `<reports>/<ID>/` | `<ID>_analysis.log`, `<ID>_step_timings.tsv`, `<ID>_analysis_summary.txt`, `<ID>_analysis_state.json`: run records. |
| `<active>/<ID>/eeg/` | `<ID>_task_Flanker_eeg_before_ica.fif`: continuous preprocessed EEG before component removal. |
| `<active>/<ID>/eeg/` | `<ID>_ica.fif`: final solution and exclusions. |
| `<active>/<ID>/eeg/` | `<ID>_task_Flanker_eeg_epochs.fif`: ICA-cleaned, baseline-corrected final response epochs. |

Generated FIF names are pipeline-specific, not a BIDS Derivatives scheme. Pre-ICA FIF is not the final ICA-cleaned deliverable.

### Group files in `<reports>/`

| File | Contents |
| --- | --- |
| `group_entry_checks.tsv` | Main-menu entry checks and CSV availability. |
| `preprocessing_batch_status.tsv` | Per-participant processing/BIDS eligibility status and latest error. |
| `group_analysis_state.json` | Current attempt status, stage statuses and current output list. |
| `PSWQ_correlation.tsv`, `PSWQ_correlation_input.tsv` | Pooled participant-level Pearson correlation and input/exclusion table. |
| `group_analysis_config.json` | Settings, expected preprocessing, versions, policies and configuration hash. |
| `group_input_manifest.tsv` | Input fingerprints, model source/cache reason, inclusion and exclusion; one combined input/check report. |
| `ROI_based_trial_amplitudes.tsv` | Streamed trial amplitudes matched to BIDS conditions. |
| `ROI_ERN_condition_amplitudes.tsv` | Participant-condition scalar values. |
| `group_subject_condition_erps.tsv` | NPZ wave-key index: participant, Congruency, ACC. |
| `group_subject_condition_erps.npz` | `times_ms` and `erp_<index>` ROI means in uV. |
| `group_cell_checks.tsv` | Counts, covariate availability and exclusion reasons. |
| `group_fit_checks.tsv` | Pre-fit checks, design rank, fit warnings and final status. |
| `LMM_input.tsv` | Model input after exclusions. |
| `LMM_model.pickle`, `LMM_results.tsv` | Model and fixed-effect table when fitting succeeds. |
| `ROI_ERP_comparison.png` | Saved descriptive ERP curves. |

Outputs are overwritten for the current attempt, not versioned by run. Downstream files are created only when their stage succeeds: consult current checks before using older files. Trial amplitudes read from FIF can differ slightly from cached pre-serialization scalars because of floating-point storage precision.

## 12. Script reference

| Script | Responsibility |
| --- | --- |
| `main.py` | BIDS eligibility, menu, serial/threaded orchestration, exception isolation and group entry checks. |
| `constants.py` | Paths, modes and scientific dataclasses. |
| `bids_correction.py` | Conditional copy/swap, event repair/validation and eligibility report. |
| `preprocess.py` | BIDS import, early QC, FCz, filters and reference. |
| `ica_preparation.py` | Period/candidate classification, repetition, ERP protection, review and QC exports. |
| `ica.py` | ICA/ICLabel, individual workflow, epochs and ERN. |
| `analysis_validation.py` | `participant_paths`, `read_analysis_state`, `load_completed_epochs`, `match_bids_responses`, `validate_ern_summary`, `input_signature`, `write_run_state`; shared paths, source/epoch validation and atomic run-state writing. |
| `pipeline_logging.py` | Isolated timestamped output, timing, snapshots, completion/retry checks and summaries. |
| `group_analysis.py` | CSV-first data, compact ERPs, diagnostics, LMM and saved figures. |
| `test.py` | Guarded exploratory entry point using production epoching and the configured result root; not a regression suite. |

## 13. Troubleshooting and final delivery

| Symptom | Check or action |
| --- | --- |
| Participant absent from menu | Check `results/inspect/bids_correction_summary.tsv`; only eligible IDs are offered. |
| Unknown/negative SET duration | Resolve the source issue or document exclusion; the code does not fabricate zero durations. |
| No usable detection windows | Inspect BAD/REVIEW coverage, boundary durations, record length and window settings. |
| Unusable period QC | Inspect `bad_period_results.tsv`; processing stops before ICA. |
| Browser repeatedly opens | Resolve REVIEW and unsupported channel-scoped BAD labels, or mark affected channels bad. |
| Existing participant reruns | Inspect completion-validation message and state JSON: missing files, incompatible settings, invalid CSV or failed/running status trigger full processing. |
| Need rerun after settings change | Changed scientific settings/source files trigger reprocessing. Use option 4 or `force_reprocess=True` to force an unchanged participant to rerun. |
| Group preprocessing mismatch | Regenerate the individual using the intended common settings. ROI/window-only changes can use compatible-epoch group fallback. |
| Group model unavailable | Read entry/file/cell checks and final `fit_status` in `group_fit_checks.tsv`. |
| Cached thirds differ from BIDS | Individual and group analysis preserve source codes; incompatible old personal CSV causes individual rerun or group recalculation. |


Development checks have exercised synthetic ERN extraction, cache reuse/fallback, summary validation, exception isolation, compact ERP storage and reference-model agreement. They do not establish full 700-participant performance, complete clean-environment reproducibility or a finished real-data statistical result. No formal automated regression suite is currently supplied.

## 14. License and authorship

Author: Xibin WANG  
Affiliation: MoDyCo Lab - UMR 7114, Universite Paris Nanterre

Original project code is licensed under the MIT License; see [LICENSE](LICENSE).
Copyright (c) 2026 Xibin WANG.

This license does not grant rights to the EEG datasets, supplied project documents
(including `Information_Subsample.docx`), or third-party software. Those materials
remain subject to their respective licenses and permissions.
