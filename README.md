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

`scripts/ica_preparation.py` detects zero periods and sustained extreme peak-to-peak activity, assesses neighboring channels, detects amplitude and high-frequency candidates, compares waveform repetition, assigns response-aware labels, and opens manual review. The reviewed continuous recording is saved before ICA.

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
