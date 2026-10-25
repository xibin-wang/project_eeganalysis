# ManyAnalyst ERN Analysis

Python workflow for response-locked EEG analysis of the Flanker task in the ManyAnalyst ERN investigation. The scripts correct dataset-specific BIDS metadata, preprocess EEG, support manual artifact review, fit ICA, create response epochs, and perform ROI-based group analysis.

This first version documents the current implementation. The workflow is interactive and does not yet produce all participant-level deliverables requested in `Information_Subsample.docx`.

## Research objectives

The primary question is whether the error-related negativity difference (incorrect minus correct responses) is associated with trait worry measured by the Penn State Worry Questionnaire (PSWQ). Secondary questions concern differences by sex and congruency.

The study description specifies harmonized EEG sampled at 250 Hz and originally referenced to FCz, with task discontinuities marked by `boundary` events. Analysis uses congruency values `0`, `0.33`, `0.66`, and `1`.

## Project layout

| Path | Purpose |
| --- | --- |
| `eeglabset/` | Original BIDS-organized EEGLAB data. |
| `eeglabset_fix/` | Corrected working copy and generated FIF files. |
| `scripts/` | Processing and analysis scripts. |
| `results/` | Inspection reports, participant QC tables, and group results. |

Expected input files include:

```text
eeglabset/participants.tsv
eeglabset/participants.json
eeglabset/dataset_description.json
eeglabset/channels.tsv
eeglabset/electrodes.tsv
eeglabset/electrodes.json
eeglabset/sub-<ID>/eeg/sub-<ID>_task-Flanker_eeg.set
eeglabset/sub-<ID>/eeg/sub-<ID>_task-Flanker_eeg.json
eeglabset/sub-<ID>/eeg/sub-<ID>_task-Flanker_events.tsv
eeglabset/sub-<ID>/eeg/sub-<ID>_task-Flanker_events.json
```

Keep external data files referenced by `.set` files alongside them. The scripts assume one Flanker recording per participant without session or run entities.

### Required metadata

`participants.tsv` must contain unique `participant_id` values and the columns `PSWQ` and `Sex`. Participant IDs must match folder names, including the `sub-` prefix.

| Event field | Use |
| --- | --- |
| `onset`, `sample` | Timing correction using sample index and sampling frequency. |
| `duration`, `value`, `trial_type` | Event correction and BIDS import. |
| `Event` | Response selection; epoch creation requires the exact value `Response`. |
| `Trial`, `RT`, `ACC`, `Congruency` | Epoch metadata and condition summaries. |

`ACC = 0` means incorrect and `ACC = 1` means correct. Congruency must use `0`, `0.33`, `0.66`, or `1`, rather than percentages. Each EEG JSON sidecar must provide `SamplingFrequency`. Valid electrode coordinates and channel types are required for spatial artifact assessment.

## Environment and installation

Use Python 3.12 or newer because the source includes f-string syntax requiring Python 3.12. The local virtual environment identifies Python 3.12.3. A graphical desktop is required for Qt-based manual review.

Example setup from the project root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install mne mne-bids mne-icalabel onnxruntime numpy pandas scipy matplotlib statsmodels mne-qt-browser PyQt6
```

The project currently has no pinned dependency manifest. Use an MNE version supporting `Annotations.extras`, which carries the event metadata used by these scripts. ICLabel explicitly uses the ONNX backend. The installation example is setup guidance, not a verified environment lockfile.

## Configuration

Most parameters are defined in `scripts/constants.py`.

| Configuration | Current defaults |
| --- | --- |
| `PathConstants` | Project root inferred from the location of `scripts/constants.py`. |
| `FilterParameters` | 0.1-80 Hz band-pass; nominal line frequency 50 Hz. |
| `PeriodQCConfig` detection | Zero periods at least 2 s; sustained PTP threshold 200 uV; 0.2 s windows, 0.1 s steps, minimum sustained duration 3 s. |
| `PeriodQCConfig` spatial assessment | Eight nearest neighbors; correlation threshold 0.8; neighbor overlap threshold 0.5. |
| `PeriodQCConfig` classification | Individual periods at least 1% of recording duration become BAD; channel assessment above 30% affected duration; predominantly shared threshold 50%. |
| `ICAPrep` detection | 1 s windows and 0.1 s steps; candidate amplitude 200 uV or power z-score 4; severe amplitude 250 uV and power z-score 5 in the same channel/window. |
| `ICAPrep` frequency bands | 30-45 Hz and 55-75 Hz. |
| `ICAPrep` waveform comparison | Similarity 0.90; repetition count at least 3, including the candidate itself; minimum interval duration 0.20 s. |
| `ICAPrep` response protection | -0.5 to +0.5 s; protection enabled; automatic rejection of unique non-severe candidates disabled. |
| `ICAParameters` | Train on a 1 Hz high-pass copy; rejection confidence at least 0.90; review prompt above 50% components labeled other. |
| `GroupParameters` | ROI FCz; amplitude window 0-0.1 s; alpha 0.05. |

Data and result folder names are configured in `scripts/main.py`. Correction source and destination are also configured in `scripts/bids_correction.py`; update both scripts when relocating data. Artifact preparation constructs its own `PeriodQCConfig` and `ICAPrep` instances, so edit their defaults in `constants.py` to configure those stages consistently.

## Running the workflow

Run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/main.py
```

The entry point first corrects BIDS metadata for the entire dataset and then presents three options:

1. **Single participant ICA:** select the zero-based participant row index in `participants.tsv`, then complete preprocessing, artifact review, ICA, and epoch creation.
2. **All participants ICA:** process participants sequentially, with manual review for each participant.
3. **Skip to group analysis:** leave the menu and analyze saved epochs for every participant listed in `participants.tsv`.

The menu reappears after options 1 and 2. Choose option 3 to proceed to group analysis. Every listed participant must already have an epochs file. Enter a valid nonnegative participant index; input validation is currently limited.

**Rerun behavior:** every launch of `main.py`, including one intended only for group analysis, copies source data into `eeglabset_fix/` again and repeats correction. FIF outputs use `overwrite=True`. Completed participants are not automatically skipped.

### Manual artifact review

The MNE Raw browser opens during artifact preparation. Press `a` to enter annotation mode, review and edit suggested intervals, then close the window to continue.

- `BAD_*`: excluded during ICA fitting and can cause overlapping epochs to be rejected.
- `REVIEW_*`: retained until manually changed.
- `KEEP_*`: manually accepted intervals, retained during processing.

Severe, nonrepetitive candidates outside protected response windows can become `BAD_severe`. Repetitive candidates and candidates overlapping protected response windows are flagged for review. Period-level QC runs separately and can already create `BAD_period*` annotations. Channels with extensive independent abnormalities can be marked bad; extensively shared abnormalities trigger an additional review window. Channels are not automatically interpolated.

If more than half of ICA components are labeled `other`, answer `y` at the console prompt to review artifacts and refit, or `n` to continue with the current solution. Changes in this later review affect subsequent processing, but the earlier reviewed-artifact TSV is not regenerated by that rerun loop.

## Processing stages

### 1. Dataset-specific BIDS correction

`scripts/bids_correction.py` copies the source dataset, swaps root-level `channels.tsv` and `electrodes.tsv`, recomputes onset as `sample / SamplingFrequency`, compares imported event onsets with EEGLAB annotations, removes a leading `x` from event values, and restores matched durations from `.set` annotations.

The swap assumes the supplied source stores channel types and coordinates in the opposite files. Review this assumption before using another dataset. Inspection reports precede duration correction and do not certify full BIDS validity.

### 2. Preprocessing

`scripts/preprocess.py` imports data through MNE-BIDS, removes leading `/` characters from annotation descriptions, restores FCz as a reference channel at head-space position `(0.0, 0.0388, 0.0922)` meters, and applies a zero-phase FIR band-pass. Average referencing excludes `MASTl`, `MASTr`, and `VOGbelow` from the reference calculation.

A line-noise detector and notch-filter branch exist, but their guard is currently `highpass_freq > line_freq + 5`. The default 0.1 Hz high-pass does not satisfy this condition, so the default workflow does not apply a 50 Hz notch.

### 3. Artifact preparation

`scripts/ica_preparation.py` prepares the preprocessed continuous EEG for ICA through two detection stages followed by manual review. The entry function runs `detect_period_artifacts()`, then `semi_auto_ica_cleaning()`, then `manual_review_artifacts()`. This stage marks time intervals and can mark channels bad; it does not remove ICA components, interpolate channels, or delete samples. Original task annotations and the sample timeline are preserved.

Parameters come from `PeriodQCConfig` for period/channel QC and `ICAPrep` for candidate detection, waveform comparison, and response protection. Both configurations exclude `MASTl` and `MASTr` from screening. EEG channel types determine the remaining eligible channels, and channels already marked bad are excluded through MNE picking and an explicit check in candidate detection.

#### 3.1 Sustained abnormal periods

`detect_period_artifacts()` works channel by channel, using EEG amplitudes converted to microvolts. Existing BAD-annotated samples are requested as NaN. Electrode coordinates must be finite, nonzero, and in the head coordinate frame; otherwise processing raises an error. The eight nearest available EEG neighbors are selected using Euclidean distance between electrode positions.

Two mechanisms identify abnormal periods:

| Mechanism | Implemented rule with default settings |
| --- | --- |
| Zero periods | Consecutive samples satisfying `abs(signal) <= zero_tolerance_v` for at least 2 s. With the default tolerance of zero, this means exact zeros. |
| Sustained extreme peak-to-peak activity | Evaluate 0.2 s windows every 0.1 s. Flag a window when its PTP exceeds 200 uV **or** exceeds the mean neighbor PTP plus three population standard deviations in that window. |

For PTP detection, each flagged window contributes only one 0.1 s step near its center to the detection mask. A continuous run of this support must last at least 3 s to become a PTP interval. Zero and PTP intervals are then combined per channel; overlapping or touching intervals merge with the default gap of zero.

The window exclusion helper recognizes annotation descriptions containing `boundary` or `edge` and skips windows crossing those discontinuities, including zero-duration boundaries. Positive-duration boundary intervals are also masked for zero detection. No additional discontinuity annotation is inserted by this QC stage.

The parameter name `zero_tolerance_v` suggests volts, but the current comparison occurs after conversion to microvolts. The default zero is unaffected; if changing this parameter, account for the implementation's actual units.

#### 3.2 Shared versus independent abnormalities

Each detected channel interval is compared with its nearest neighbors. A neighbor must first be abnormal for at least 50% of that interval. The interval is then considered shared if at least one qualifying neighbor meets either applicable similarity criterion:

- For zero samples, at least 50% of the channel's detected zero samples are also zero in that neighbor.
- For finite waveforms with nonzero variance, the absolute Pearson correlation is at least 0.8. Positive and negative correlations both qualify.

If no neighbor qualifies, the interval is independent. A shared interval receives a global annotation; an independent interval receives a channel-specific annotation with the affected channel attached.

Two duration decisions are made separately:

| Decision | Rule | Result |
| --- | --- | --- |
| Individual interval label | Interval duration is at least 1% of total recording duration. | `BAD_period` for shared intervals or `BAD_period_independent_<channel>` for independent intervals. |
| Shorter interval label | Interval duration is below 1% of total recording duration. | Corresponding `REVIEW_period` or `REVIEW_period_independent_<channel>` label. |
| Channel usability | Combined abnormal duration exceeds 30% of total recording duration and less than 50% of abnormal duration is classified as shared. | Channel is added to `raw.info['bads']`. |
| Extensive shared abnormality | Combined abnormal duration exceeds 30% and at least 50% of abnormal duration is classified as shared. | Channel is reported as `data_unusable` and an extra browser opens for review; the code does not automatically abort or discard the participant. |

These fractions use the entire recording sample count as the denominator for recording duration, rather than only unannotated data. For example, in a 600 s recording, one 6 s interval meets the 1% BAD-period threshold; several shorter separate intervals do not each become BAD merely because their combined duration exceeds 6 s.

#### 3.3 Amplitude and high-frequency candidate detection

`detect_artifacts()` scans the remaining eligible data in 1 s windows stepped every 0.1 s. Positive-duration annotations starting with `BAD` or `REVIEW` are masked across all selected channels for this detection stage, including channel-specific period annotations. Windows containing nonfinite samples or crossing recognized boundary/edge annotations are skipped. Thus, an earlier `REVIEW_period` is withheld from further candidate screening even though it remains retained for ICA unless manually relabeled.

For every valid window and channel, the script calculates PTP amplitude and a Welch power spectral density using a Hann window, the full detection window as `nperseg`, no internal overlap, and constant detrending. Power is integrated separately over 30-45 Hz and 55-75 Hz, leaving the region around 50 Hz out of these detection features. This frequency selection does not itself filter line noise from the EEG.

Power is transformed to decibels. A robust baseline is calculated independently for each channel and frequency band across all valid windows in that participant:

```text
log_power = 10 * log10(max(band_power, 1e-20))
robust_sigma = max(1.4826 * median_absolute_deviation(log_power), 1.0)
z_power = (log_power - median(log_power)) / robust_sigma
```

The scale floor is 1 dB. These are participant-specific screening scores, not inferential z tests.

| Detection | Default rule |
| --- | --- |
| Candidate channel/window | PTP greater than 200 uV **or** power score greater than 4 in either frequency band. |
| Candidate window | At least one selected channel meets the candidate rule. |
| Severe channel/window | PTP greater than 250 uV **and** power score greater than 5 in either band, in the same channel and window. |
| Severe window | At least `min_severe_channels` channels meet the severe rule; the default is one. |

A high amplitude in one channel and high-frequency power in another cannot jointly satisfy the severe rule. All configured frequency bands must lie below Nyquist. Processing raises an error if the recording is shorter than one detection window, no valid channels remain, or no valid detection windows remain.

#### 3.4 Persistence and interval construction

Candidate and severe detections are converted to separate time masks. Each detected window contributes one step centered on the window. Contiguous center support must last at least `min_duration = 0.20 s` before it is accepted; a single isolated 1 s detection window therefore does not automatically pass the duration requirement.

Accepted support intervals are expanded back toward their contributing 1 s analysis windows and clipped to recording limits. Overlapping or touching expanded intervals merge. Severe intervals are included in the final candidate union, so they are not lost during consolidation. `merge_gap = 0` allows no extra positive gap between final intervals.

Consequently, the reported interval boundaries reflect detection-window support and expansion, rather than an exact sample-by-sample estimate of artifact onset and offset. Detection windows and minimum-duration settings are rounded to samples at the recording's sampling frequency.

#### 3.5 Waveform repetition

`find_repetitive_intervals()` compares final candidate intervals within the participant. `extract_waveform()` subtracts each channel's mean and rejects waveforms with fewer than five samples or nonfinite data. Intervals up to 2 s retain their original sampling points; longer intervals are resampled to 200 points for comparison. Short and long intervals are compared only within their respective groups, and paired durations must differ by a ratio no greater than 1.5.

For each eligible pair, `waveform_similarity()` combines temporal and spatial agreement:

1. Calculate each channel's standard deviation in both intervals. Select up to eight channels with the strongest shared activity, ranked by the smaller of their two standard deviations, and exclude channels with negligible variance.
2. Calculate spatial similarity as the cosine similarity between the channel standard-deviation vectors across all selected EEG channels.
3. Search temporal shifts up to approximately 0.08 s in either direction, requiring overlap of at least 80% of the longer waveform's sample count. For resampled long intervals, the shift conversion uses an effective sampling rate based on their mean duration.
4. At each valid shift, recenter the overlapping signals and average their signed channel correlations. Multiply the best temporal correlation by spatial similarity and clip the result to [-1, 1].

A score of at least 0.90 counts as a match. A valid candidate counts as matching itself; the default repetition threshold of three therefore requires at least two other matching intervals. Matching is evaluated for each candidate independently, rather than by requiring every member of a group to match every other member. Repetition flags an interval for review and does not automatically establish that it is safe or artifactual.

#### 3.6 Response protection and automatic labels

With `protect_erp = True`, response times are read from annotation extras using a whitespace-trimmed, case-insensitive `Event` comparison to `response`. Onsets are converted to time relative to the current Raw start. A candidate is ERP-related if it overlaps any response window from -0.5 to +0.5 s; merely touching a window endpoint does not count. Flanker- and target-locked windows are not separately protected.

A candidate is severe if it overlaps any severe interval. The following table gives the complete default classification:

| Severe | Repetitive | Overlaps response window | Label |
| --- | --- | --- | --- |
| Yes | Yes | Yes | `REVIEW_severe_repetitive_ERN` |
| Yes | Yes | No | `REVIEW_severe_repetitive` |
| Yes | No | Yes | `REVIEW_severe_ERN` |
| Yes | No | No | `BAD_severe` |
| No | Yes | Yes | `REVIEW_repetitive_ERN` |
| No | Yes | No | `REVIEW_repetitive` |
| No | No | Yes | `REVIEW_unique_ERN` |
| No | No | No | `REVIEW_unique` |

Setting `auto_reject_unique = True` changes the last row to `BAD_unique`. Disabling `protect_erp` removes response-overlap protection for this classification. Protection does not reverse BAD-period labels from the earlier QC stage, and manual review can still reject an interval overlapping a response.

For a nonempty annotation collection, missing usable response metadata raises an error when response protection is enabled. A completely empty annotation collection returns no response times. The later epoch-creation function uses the stricter exact value `Response`, so consistent event capitalization remains necessary.

New candidate labels are appended as global annotations while preserving existing task and period annotations. Interval calculations use seconds relative to the recording start; annotation onsets include `raw.first_time` when stored in MNE.

#### 3.7 Manual decisions and saved records

`manual_review_artifacts()` opens a blocking browser showing 10 s at a time, initially with 30 channels and automatic scaling. Press `a` to edit annotations. Inspect the waveform and spatial distribution, change rejected intervals to a `BAD_*` label, and use a non-BAD label such as `KEEP_*` for accepted intervals. Close the browser to return the edited Raw object and continue processing. Retaining `REVIEW_*` leaves the interval available for subsequent ICA and epoching; the program does not require every REVIEW label to be resolved.

The artifact stage creates these records:

| Record | Contents and timing |
| --- | --- |
| `results/<ID>/bad_period_results.tsv` | Automatic period QC before the main manual review: channel, start, duration, shared status, individual-period fraction, total affected fraction, shared fraction, unusable flag, and label. |
| `results/<ID>/artifact_results_reviewed.tsv` | BAD/REVIEW/KEEP annotations after the main review: annotation index, relative start/end, duration, label, and `exclude_from_ica`. Original task event annotations are omitted. |
| `eeglabset_fix/<ID>/eeg/<ID>_task_Flanker_eeg_before_ica.fif` | Continuous preprocessed EEG with reviewed annotations and bad-channel state. |
| Console summary | Reviewed annotation counts, label counts, and merged BAD duration and percentage. Overlapping BAD intervals are merged to avoid double-counting time. |

The automatic candidate report additionally contains severity, repetition count, and ERP-overlap flags in memory, but `main()` does not export that full report or the automatic summary. The reviewed TSV does not include annotation channel scopes or the bad-channel list; those remain in the FIF recording. Its duration summary combines BAD intervals across channels, rather than reporting separate channel-specific data loss. The label counts printed after the reviewed summary come from the automatic candidate report and can differ from the final manual labels.

MNE's annotation rejection and channel-picking behavior determine how stored BAD intervals and bad channels affect ICA fitting and epoch creation. This stage preserves data samples for later processing; annotations identify which portions should be omitted. Subsequent ICA rerun review can update the saved continuous data again, but does not refresh the original reviewed-artifact TSV.

### 4. ICA and ICLabel

`scripts/ica.py` fits extended Infomax ICA on non-bad EEG channels using `random_state=42`. Fitting defaults to a copy high-pass filtered at 1 Hz. ICLabel is evaluated on training data with bad annotation segments omitted.

A component is selected for removal when its highest class probability is at least 0.90 and its predicted class is neither `brain` nor `other`. The solution is applied to epochs from the original preprocessed recording, retaining the 0.1 Hz analysis high-pass.

The ICA training copy renames `boundary` to `BAD_boundary`. That rename is not propagated to the original recording used for epoching, so epoch-stage discontinuity handling needs verification.

### 5. Response-locked epochs

Responses are identified through `raw.annotations.extras` using `Event == "Response"`. Epochs span -0.5 to +0.5 s and retain trial metadata. ICA is applied before baseline correction over epoch start through time zero, approximately -0.5 to 0 s. Epoch rejection uses bad annotations; `make_epochs()` does not configure an additional amplitude rejection threshold.

### 6. Group analysis

`scripts/group_analysis.py` averages selected ROI channels and samples in the configured 0-100 ms window to obtain one amplitude per trial in microvolts. It then computes a difference for each participant and congruency condition:

```text
ACC_diff = mean(incorrect response amplitude) - mean(correct response amplitude)
```

More negative values indicate a more negative error-minus-correct response in this window. Cells lacking either response type, PSWQ, or sex are omitted from model fitting. There is no minimum retained-trial requirement beyond having both response types.

The random-intercept linear mixed model uses REML:

```text
ACC_diff ~ C(Congruency) + PSWQ + C(Sex)
groups = participant
random effects = intercept
```

Congruency is categorical with zero congruency as the reference. Sex is categorical; PSWQ is continuous and uncentered. Fixed-effect output includes estimates, standard errors, z statistics, p-values, and confidence intervals. Significance screening uses the configured alpha without multiple-comparison adjustment.

If any non-intercept fixed effect is significant, correct/incorrect ERP plots are saved and displayed for all four congruency levels. These waveforms are averaged within participant first, then across participants included in the model for each condition. A separate correlation estimate is not exported.

## Generated outputs

`<ID>` below means the full participant ID, including `sub-`.

| Location | File | Contents |
| --- | --- | --- |
| `results/inspect/` | `inspect_results.txt` | Event onset comparison messages. |
| `results/inspect/` | `<ID>_set_annotations.tsv`, `<ID>_bids_annotations.tsv` | Annotation exports for timing inspection. |
| `results/<ID>/` | `bad_period_results.tsv` | Period-level artifact assessment by channel. |
| `results/<ID>/` | `artifact_results_reviewed.tsv` | Artifact annotations after initial manual review. |
| `results/<ID>/` | `<ID>_iclabel.tsv` | Component labels, confidence, rejection flags, and class probabilities. |
| `eeglabset_fix/<ID>/eeg/` | `<ID>_task_Flanker_eeg_before_ica.fif` | Continuous preprocessed data and reviewed annotations before component removal. |
| `eeglabset_fix/<ID>/eeg/` | `<ID>_ica.fif` | ICA solution and selected exclusions. |
| `eeglabset_fix/<ID>/eeg/` | `<ID>_task_Flanker_eeg_epochs.fif` | ICA-cleaned, baseline-corrected response epochs and metadata. |
| `eeglabset_fix/` | `ROI_based_trial_amplitudes.tsv` | Combined participant trial amplitudes and behavioral metadata. |
| `results/` | `LMM_model.pickle` | Serialized fitted model. |
| `results/` | `LMM_results.tsv` | Fixed-effect estimates and inferential statistics. |
| `results/` | `ROI_ERP_comparison.png` | Generated only when a non-intercept fixed effect is significant. |

Generated FIF files reside inside the corrected data tree using the exact filenames above. A separate BIDS Derivatives layout is not implemented. Artifact summaries are printed to the console; a consolidated participant preprocessing summary is not saved.

## Script reference

| Script | Role |
| --- | --- |
| `scripts/main.py` | Interactive entry point and orchestration. |
| `scripts/constants.py` | Shared paths and processing parameters. |
| `scripts/bids_correction.py` | Working-copy creation and metadata correction; also executable independently. |
| `scripts/preprocess.py` | BIDS import, FCz restoration, filtering, and referencing. Its standalone block processes the first participant without saving the returned Raw object. |
| `scripts/ica_preparation.py` | Period QC, candidate detection, classification, and review. |
| `scripts/ica.py` | ICA fitting, ICLabel evaluation, participant processing, and epochs. |
| `scripts/group_analysis.py` | ROI extraction, condition differences, modeling, and plots. |
| `scripts/test.py` | Local exploratory review using existing FIF files; not an automated test suite. |

## Current limitations and remaining deliverables

To satisfy the study's requested submission workflow, the following work remains:

- Export one file per participant with incorrect-minus-correct ERN across all conditions and separately for all four congruency conditions. Condition differences are currently calculated in memory during group analysis; the saved trial table combines participants.
- Export one preprocessing summary per participant with bad/interpolated channel counts, removed ICA component counts, and retained epochs by accuracy and congruency.
- Add completed-output checks and participant-level error handling so a failed participant does not stop the batch.
- Add unattended execution if required for the larger dataset; current processing requires graphical review and console input.
- Verify event sample conventions, epoch-stage boundary handling, and the line-noise filtering guard before finalizing the workflow.
- Prepare the requested preprocessed EEG examples for three participants and the requested correlation estimate.

Group analysis requires saved epochs for every listed participant, an available non-bad ROI, and consistent time axes; failures stop processing. The pipeline has not been run end to end as part of preparing this README.

The root `.gitignore` excludes new root files by default, including this README. Tracking it requires an explicit allow rule or forced add; creating this document does not modify `.gitignore`.
