"""
Semi-automatic EEG artifact cleaning before ICA
================================================

Pipeline:
1. Detect candidate artifacts
2. Detect high-frequency + high-amplitude artifacts
3. Merge intervals and apply minimum duration
4. Compare waveform similarity among remaining intervals
5. Identify repetitive waveforms
6. Protect stimulus/response-related intervals
7. Generate MNE annotations
8. Save ICA training dataset and detection report

Dependencies:
    mne
    numpy
    scipy
    pandas

Input:
    MNE Raw object

Output:
    raw_ica: ICA training copy with annotations
    report: pandas DataFrame
    summary: dict

All interval times are relative to the beginning of Raw.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch, resample

from constants import PeriodQCConfig, ICAPrep

# ============================================================
# 1. BASIC UTILITIES
# ============================================================
def get_eeg_picks(raw: mne.io.BaseRaw, cfg: ICAPrep | PeriodQCConfig) -> np.ndarray:
    picks = mne.pick_types(
        raw.info,
        eeg=True
    )
    return np.array([
        p for p in picks
        if raw.ch_names[p] not in cfg.exclude_channels
    ], dtype=int)

def merge_intervals(
    intervals: list,
    gap: float=0.0,
    min_duration: float=0.20,
):
    """Merge overlapping/nearby intervals.
    """
    intervals = sorted(
        (float(a), float(b))
        for a, b in intervals
        if b > a
    )
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(
                merged[-1][1], end
            )
        else:
            merged.append([start, end])
    return [
        (a, b)
        for a, b in merged
        if b - a >= min_duration
    ]

def subtract_intervals(intervals, excluded):
    """Subtract severe intervals from ordinary candidates.

    """
    result = []
    for start, end in intervals:
        pieces = [(start, end)]
        for a, b in excluded:
            next_pieces = []
            for x, y in pieces:
                if b <= x or a >= y:
                    next_pieces.append((x, y))
                else:
                    if x < a:
                        next_pieces.append((x, a))
                    if b < y:
                        next_pieces.append((b, y))
            pieces = next_pieces
        result.extend(pieces)
    return result

def mask_to_intervals(mask, fs, min_duration=0.0):
    """Convert sample mask into time intervals."""
    edges = np.diff(
        np.r_[False, mask, False].astype(int)
    )
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [
        (a / fs, b / fs)
        for a, b in zip(starts, stops)
        if (b - a) / fs >= min_duration
    ]

def intervals_to_mask(intervals, n_samples, fs):
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in intervals:
        a = max(0, int(np.floor(start * fs)))
        b = min(
            n_samples,
            int(np.ceil(end * fs))
        )
        mask[a:b] = True

    return mask

def get_excluded_start_ranges(raw, win, include_periods=False):

    fs = raw.info["sfreq"]
    ranges = []

    for ann in raw.annotations:
        desc = str(ann["description"]).strip().lower()

        is_boundary = "boundary" in desc or "edge" in desc
        is_period = (
            include_periods
            and desc.startswith(("bad_", "review_"))
        )

        if not (is_boundary or is_period):
            continue

        duration = float(ann["duration"])
        if duration < 0:
            continue

        # Zero-duration boundaries still block crossing windows.
        if duration == 0 and not is_boundary:
            continue

        period_start = (
            ann["onset"] - raw.first_time
        ) * fs
        period_stop = period_start + duration * fs

        ranges.append((period_start - win, period_stop))

    return ranges


# ============================================================
# 2. BAD CHANNEL DETECTION
# ============================================================

def detect_period_artifacts(raw, cfg: PeriodQCConfig):
    """Return a Raw copy with QC annotations and a period-level report."""
    raw = raw.copy()
    fs = raw.info["sfreq"]

    win = max(2, round(cfg.ptp_window_sec * fs))
    step = max(1, round(cfg.ptp_step_sec * fs))
    picks = get_eeg_picks(raw, cfg)
    if not picks.size:
        raise ValueError("No EEG channels available for screening.")
    data = raw.get_data(
        picks=picks,
        reject_by_annotation="NaN",
    ) * 1e6

    n_channels, n_samples = data.shape

    boundary_ranges = get_excluded_start_ranges(
        raw, win, include_periods=False
    )
    sample_positions = np.arange(n_samples)
    boundary_mask = np.zeros(n_samples, dtype=bool)

    for left, right in boundary_ranges:
        # Recover the actual boundary interval, without window padding.
        boundary_mask |= (
                (sample_positions >= left + win)
                & (sample_positions < right)
        )

    if not (
            0 < cfg.bad_period_fraction <= 1
            and 0 < cfg.bad_channel_fraction <= 1
            and 0 <= cfg.correlation_threshold <= 1
            and 0 < cfg.neighbor_overlap_threshold <= 1
    ):
        raise ValueError("Invalid classification or correlation thresholds.")

    if (
        cfg.zero_tolerance_v < 0
        or cfg.zero_min_sec <= 0
        or cfg.ptp_threshold_uv <= 0
        or cfg.ptp_min_sec <= 0
        or cfg.ptp_step_sec <= 0
        or cfg.ptp_window_sec <= 0
        or cfg.merge_gap_sec < 0
        or cfg.n_neighbors < 1
    ):
        raise ValueError("Invalid detection parameters.")

    names = [raw.ch_names[p] for p in picks]
    # Select neighboring EEG channels by 3D distance in head space.
    montage = raw.get_montage()
    if montage is None:
        raise ValueError(
            "A montage with head-space electrode coordinates is required."
        )

    positions = montage.get_positions()
    if positions["coord_frame"] != "head":
        raise ValueError(
            "Electrode coordinates must be in head space; "
            f"got {positions['coord_frame']!r}."
        )
    ch_pos = positions["ch_pos"]

    # Keep coordinate order identical to the rows of data.
    missing = [name for name in names if name not in ch_pos]
    if missing:
        raise ValueError(
            f"Missing electrode coordinates for: {missing}"
        )

    xyz = np.asarray(
        [ch_pos[name] for name in names],
        dtype=float,
    )

    valid_positions = (
        np.isfinite(xyz).all(axis=1)
        & (np.linalg.norm(xyz, axis=1) > 0)
    )
    if not valid_positions.all():
        invalid = [
            names[i]
            for i in np.flatnonzero(~valid_positions)
        ]
        raise ValueError(
            f"Invalid head-space electrode coordinates for: {invalid}"
        )

    # Pairwise Euclidean distance in meters.
    distances = np.linalg.norm(
        xyz[:, None, :] - xyz[None, :, :],
        axis=2,
    )

    np.fill_diagonal(distances, np.inf)
    n_nearest = min(cfg.n_neighbors, n_channels - 1)
    channel_neighbors = [
        np.argsort(
            distances[i], kind="stable"
        )[:n_nearest].tolist()
        for i in range(n_channels)
    ]

    if step > win:
        raise ValueError("ptp_step_sec must not exceed ptp_window_sec.")

    zero = np.zeros(data.shape, dtype=bool)
    abnormal = np.zeros(data.shape, dtype=bool)
    channel_intervals = []

    # Detect zero periods and sustained extreme PTP.
    for i, x in enumerate(data):
        zero_intervals = mask_to_intervals(
            (np.abs(x) <= cfg.zero_tolerance_v) & ~boundary_mask,
            fs,
            min_duration=cfg.zero_min_sec,
        )
        zero[i] = intervals_to_mask(zero_intervals, n_samples, fs)

        ptp_mask = np.zeros(n_samples, dtype=bool)

        for a in range(0, n_samples - win + 1, step):
            if any(
                    left < a < right
                    for left, right in boundary_ranges
            ):
                continue

            segment = x[a:a + win]

            if not np.isfinite(segment).all():
                continue

            amplitude_uv = np.ptp(segment)
            locally_high = False
            neighbors = channel_neighbors[i]

            if neighbors:
                neighbor_segments = data[
                    neighbors, a:a + win
                ]
                finite_neighbors = np.isfinite(
                    neighbor_segments
                ).all(axis=1)
                if finite_neighbors.any():
                    neighbor_ptp_uv = np.ptp(
                        neighbor_segments[finite_neighbors],
                        axis=1,
                    )
                    locally_high = amplitude_uv > (
                        neighbor_ptp_uv.mean()
                        + 3.0 * neighbor_ptp_uv.std(ddof=0)
                    )

            if (
                amplitude_uv > cfg.ptp_threshold_uv
                or locally_high
            ):
                # Assign one step around each window center.
                left = a + (win - step) // 2
                ptp_mask[left:left + step] = True

        ptp_intervals = mask_to_intervals(
            ptp_mask,
            fs,
            min_duration=cfg.ptp_min_sec,
        )

        intervals = merge_intervals(
            zero_intervals + ptp_intervals,
            gap=cfg.merge_gap_sec,
            min_duration=0.0,
        )
        channel_intervals.append(intervals)
        abnormal[i] = intervals_to_mask(intervals, n_samples, fs)

    records = []
    unusable = []

    # Assess neighboring channels and classify abnormal periods.
    for i, name in enumerate(names):
        if not channel_intervals[i]:
            continue

        affected_samples = int(abnormal[i].sum())
        fraction = affected_samples / n_samples
        periods = []

        neighbors = channel_neighbors[i]

        for start, end in channel_intervals[i]:
            a, b = round(start * fs), round(end * fs)
            shared = False
            for j in neighbors:
                # Require simultaneous abnormality in the neighbor.
                if (
                    abnormal[j, a:b].mean()
                    < cfg.neighbor_overlap_threshold
                ):
                    continue
                # Compare zero co-occurrence over the detected zero samples.
                zero_samples = zero[i, a:b]
                if zero_samples.any():
                    shared = (
                        zero[j, a:b][zero_samples].mean()
                        >= cfg.neighbor_overlap_threshold
                    )
                # Also assess waveform correlation when variance is nonzero.
                if not shared:
                    x, y = data[i, a:b], data[j, a:b]
                    if (
                        np.isfinite(x).all()
                        and np.isfinite(y).all()
                        and np.std(x) > 0
                        and np.std(y) > 0
                    ):
                        shared = (
                            abs(np.corrcoef(x, y)[0, 1])
                            >= cfg.correlation_threshold
                        )
                if shared:
                    break

            periods.append((start, end, shared))

        # Calculate the proportion of abnormal duration classified as shared.
        shared_samples = sum(
            round(end * fs) - round(start * fs)
            for start, end, shared in periods
            if shared
        )
        shared_fraction = shared_samples / affected_samples
        mostly_shared = (
            shared_fraction >= cfg.shared_duration_threshold
        )

        # Assess channel usability when total abnormal duration exceeds 30%.
        if fraction > cfg.bad_channel_fraction:
            if mostly_shared:
                unusable.append(name)
            elif name not in raw.info["bads"]:
                raw.info["bads"].append(name)

        # Classify each interval independently by its own duration.
        for start, end, shared in periods:
            a, b = round(start * fs), round(end * fs)
            period_fraction = (b - a) / n_samples

            prefix = (
                "BAD"
                if period_fraction >= cfg.bad_period_fraction
                else "REVIEW"
            )
            label = f"{prefix}_period"

            if not shared:
                label += f"_independent_{name}"

            raw.annotations.append(
                onset=raw.first_time + start,
                duration=end - start,
                description=label,
                ch_names=[[] if shared else [name]],
            )
            records.append({
                "channel": name,
                "start": start,
                "duration": end - start,
                "shared": shared,
                "period_fraction": period_fraction,
                "affected_fraction": fraction,
                "shared_fraction": shared_fraction,
                "data_unusable": (
                    fraction > cfg.bad_channel_fraction
                    and mostly_shared
                ),
                "label": label,
            })

    report = pd.DataFrame(
        records,
        columns=[
            "channel",
            "start",
            "duration",
            "shared",
            "period_fraction",
            "affected_fraction",
            "shared_fraction",
            "data_unusable",
            "label",
        ],
    )

    if unusable:
        print(
            f"Data marked unusable pending manual review: {unusable}. "
            "Total abnormal duration exceeds 30% and is predominantly shared."
        )
        raw.plot(block=True, scalings="auto")

    return raw, report

# ============================================================
# 3. AUTOMATIC CANDIDATE DETECTION
# ============================================================
def detect_artifacts(raw: mne.io.BaseRaw, cfg:ICAPrep):
    """
    Detect all candidate and severe artifact intervals.

    Candidate:
        high amplitude OR high-frequency power anomaly

    Severe:
        high amplitude AND high-frequency power anomaly
        in the SAME channel and SAME window

    Returns:
        all_intervals, severe_intervals
    """
    fs = raw.info["sfreq"]
    picks = np.array([
        p for p in get_eeg_picks(raw, cfg)
        if raw.ch_names[p] not in raw.info["bads"]
    ], dtype=int)

    if len(picks) == 0:
        raise ValueError("No valid EEG channels")
    for low, high in cfg.high_freq_band:
        if not (0 < low < high < fs / 2):
            raise ValueError(
                f"Invalid high-frequency band ({low}, {high}); "
                f"require 0 < low < high < Nyquist ({fs / 2} Hz)"
            )
    # Convert from V to µV
    data = raw.get_data(
        picks=picks,
        reject_by_annotation="NaN",
    ) * 1e6

    n_ch, n_samples = data.shape

    # Exclude all BAD and REVIEW periods across all selected channels.
    excluded = np.zeros(n_samples, dtype=bool)

    for ann in raw.annotations:
        description = str(ann["description"]).strip().lower()

        if not description.startswith(("bad", "review")):
            continue
        if ann["duration"] <= 0:
            continue

        start = ann["onset"] - raw.first_time
        end = start + ann["duration"]
        a = max(0, int(np.floor(start * fs)))
        b = min(n_samples, int(np.ceil(end * fs)))

        if b > a:
            excluded[a:b] = True

    # Preserve the original sample positions.
    data[:, excluded] = np.nan

    win = int(round(cfg.window_sec * fs))
    step = max(1, int(round(cfg.step_sec * fs)))

    if win > n_samples:
        raise ValueError("Recording is too short")
    starts = np.arange(
        0,
        n_samples - win + 1,
        step,
    )

    valid_starts = []
    powers = []
    amplitudes = []

    # Detect annotation discontinuities / boundaries
    excluded_start_ranges = get_excluded_start_ranges(
        raw, win, include_periods=True
    )

    # --------------------------------------------------------
    # First pass: calculate PSD + amplitude
    # --------------------------------------------------------
    for start in starts:
        stop = start + win
        if any(
                left < start < right
                for left, right in excluded_start_ranges
        ):
            continue
        segment = data[:, start:stop]
        if not np.isfinite(segment).all():
            continue

        amplitude = np.ptp(
            segment,
            axis=1,
        ) # Calculate peak to peak amplitude of each segment

        freqs, psd = welch(
            segment,
            fs=fs,
            window="hann",
            nperseg=win,
            noverlap=0,
            detrend="constant",
            scaling="density",
            axis=-1,
        )   # Calculate band power of each segment
        band_powers = []

        for low, high in cfg.high_freq_band:
            band_mask = (
                (freqs >= low) &
                (freqs <= high)
            )
            band_power = np.trapezoid(
                psd[:, band_mask],
                freqs[band_mask],
                axis=-1,
            )

            band_powers.append(band_power)

        band_powers = np.stack(band_powers, axis=-1)
        valid_starts.append(start)
        amplitudes.append(amplitude)
        powers.append(band_powers)

    if not valid_starts:
        raise ValueError(
            "No valid windows for detection"
        )

    valid_starts = np.asarray(valid_starts)
    amplitudes = np.stack(amplitudes)
    powers = np.stack(powers)

    # --------------------------------------------------------
    # Second pass: robust per-channel power baseline
    # --------------------------------------------------------
    log_power = 10 * np.log10(
        np.maximum(powers, 1e-20)
    )
    median_power = np.median(
        log_power,
        axis=0,
    )
    mad = np.median(
        np.abs(log_power - median_power),
        axis=0,
    )
    robust_sigma = np.maximum(
        1.4826 * mad,
        1.0,
    )

    z_power = (
        log_power - median_power
    ) / robust_sigma

    # --------------------------------------------------------
    # Third pass: candidate / severe classification
    # --------------------------------------------------------
    candidate_amplitude = (
        amplitudes > cfg.candidate_amp_uv
    )
    candidate_frequency = np.any(
        z_power > cfg.candidate_power_z,
        axis=-1,
    )

    # Broad candidate screening
    candidate_channels = (
        candidate_amplitude |
        candidate_frequency
    )
    candidate_windows = np.any(
        candidate_channels,
        axis=1,
    )

    severe_frequency = np.any(
        z_power > cfg.severe_power_z,
        axis=-1,
    )

    # Severe: BOTH conditions in SAME channel
    severe_channels = (
        (amplitudes > cfg.severe_amp_uv) &
        severe_frequency
    )
    severe_windows = (
        np.sum(severe_channels, axis=1)
        >= cfg.min_severe_channels
    )

    # --------------------------------------------------------
    # Fourth pass: convert window detections to intervals
    # --------------------------------------------------------
    candidate_mask = np.zeros(
        n_samples, dtype=bool
    )
    severe_mask = np.zeros(
        n_samples, dtype=bool
    )

    for start, candidate, severe in zip(
        valid_starts,
        candidate_windows,
        severe_windows,
    ):
        stop = start + win
        if candidate:
            candidate_mask[start:stop] = True
        if severe:
            severe_mask[start:stop] = True

    # Important:
    # The min-duration requirement applies to contiguous
    # detected-window support, not a single expanded window.
    # Otherwise one 1-second window always passes a 0.2s cutoff.
    #
    # Use window centers to estimate persistence.
    center_mask = np.zeros(
        n_samples, dtype=bool
    )
    severe_center_mask = center_mask.copy()

    for start, candidate, severe in zip(
        valid_starts,
        candidate_windows,
        severe_windows,
    ):
        center = start + win // 2
        if candidate:
            a = max(0, center - step // 2)
            b = min(n_samples, a + step)
            center_mask[a:b] = True
        if severe:
            a = max(0, center - step // 2)
            b = min(n_samples, a + step)
            severe_center_mask[a:b] = True
    persistent_candidate = mask_to_intervals(
        center_mask,
        fs,
        cfg.min_duration,
    )
    persistent_severe = mask_to_intervals(
        severe_center_mask,
        fs,
        cfg.min_duration,
    )

    # Expand accepted detections to their contributing
    # analysis windows; merge only afterward.
    left_pad = (win // 2 - step // 2) / fs
    right_pad = (win - (win // 2 - step // 2) - step) / fs

    def expand_intervals(intervals):
        expanded = [
            (
                max(0.0, a - left_pad),
                min(n_samples / fs, b + right_pad),
            )
            for a, b in intervals
        ]

        return merge_intervals(
            expanded,
            gap=0.0,
            min_duration=0.0,
        )

    all_intervals = expand_intervals(persistent_candidate)
    severe_intervals = expand_intervals(persistent_severe)

    # Never lose severe intervals when building union
    all_intervals = merge_intervals(
        all_intervals + severe_intervals,
        gap=cfg.merge_gap,
        min_duration=cfg.min_duration,
    )

    return all_intervals, severe_intervals


# ============================================================
# 4. WAVEFORM EXTRACTION
# ============================================================
def extract_waveform(
    raw,
    interval,
    picks,
    cfg,
):
    """
    Extract waveform for similarity comparison.

    Short candidates:
        Preserve original sampling points.

    Long candidates:
        Normalize to cfg.waveform_points.

    Returns
    -------
    dict or None
        {
            "data": ndarray (channels, samples),
            "duration": float,
            "normalized": bool,
        }
    """
    start, end = interval
    fs = raw.info["sfreq"]
    a = max(0, int(round(start * fs)))
    b = min(
        raw.n_times,
        int(round(end * fs))
    )
    if b - a < 5:
        return None
    x = raw.get_data(
        picks=picks,
        start=a,
        stop=b,
        reject_by_annotation="NaN",
    )
    if not np.isfinite(x).all():
        return None

    duration = (b - a) / fs

    # Remove DC from each channel
    x = x - np.mean(
        x, axis=1, keepdims=True
    )
    # Classify according to duration
    is_long = duration > cfg.long_candidate_sec

    if is_long:
        x = resample(
            x,
            cfg.waveform_points,
            axis=1,
        )

    return {
        "data": x,
        "duration": duration,
        "normalized": is_long,
    }

# ============================================================
# 5. WAVEFORM SIMILARITY
# ============================================================

def waveform_similarity(
    x_info,
    y_info,
    cfg,
):
    """
    Compare temporal waveform shape and spatial distribution.

    Returns similarity in [-1, 1].
    """

    if x_info is None or y_info is None:
        return 0.0
    # Never compare short with long candidates
    if x_info["normalized"] != y_info["normalized"]:
        return 0.0
    x = x_info["data"]
    y = y_info["data"]
    if x.shape[0] != y.shape[0]:
        return 0.0
    duration_x = x_info["duration"]
    duration_y = y_info["duration"]

    # -----------------------------------------
    # 1. Determine allowable time lag
    # -----------------------------------------

    if x_info["normalized"]:
        # Normalized candidates have different effective
        # time resolutions depending on their duration.
        mean_duration = (duration_x + duration_y) / 2

        effective_fs = (
            cfg.waveform_points / mean_duration
        )
    else:
        # Short candidates retain original sample spacing.
        # Length / duration gives original sampling rate.
        effective_fs = x.shape[1] / duration_x

    max_lag = int(round(
        cfg.max_time_shift_sec * effective_fs
    ))

    # -----------------------------------------
    # 2. Channel amplitude distributions
    # -----------------------------------------

    sx = np.std(x, axis=1)
    sy = np.std(y, axis=1)
    strength = np.minimum(sx, sy)

    idx = np.argsort(strength)[
        -cfg.max_similarity_channels:
    ]
    valid = (
        (sx[idx] > 1e-12) &
        (sy[idx] > 1e-12)
    )

    idx = idx[valid]
    if len(idx) == 0:
        return 0.0

    # Spatial similarity from all EEG channels
    spatial = np.dot(sx, sy) / max(
        np.linalg.norm(sx) *
        np.linalg.norm(sy),
        1e-20,
    )

    # -----------------------------------------
    # 3. Find best temporal alignment
    # -----------------------------------------

    xx = x[idx]
    yy = y[idx]

    nx = xx.shape[1]
    ny = yy.shape[1]

    best_temporal = -1.0

    min_overlap = int(np.ceil(
        cfg.min_overlap_fraction * max(nx, ny)
    ))

    for lag in range(-max_lag, max_lag + 1):
        # Positive lag shifts the starting point of x.
        # Negative lag shifts the starting point of y.
        x_start = max(0, lag)
        y_start = max(0, -lag)
        overlap = min(
            nx - x_start,
            ny - y_start,
        )
        if overlap < min_overlap:
            continue
        x_part = xx[
            :,
            x_start:x_start + overlap
        ]
        y_part = yy[
            :,
            y_start:y_start + overlap
        ]

        # Re-center after cropping overlapping segments
        x_part = (
            x_part -
            x_part.mean(axis=1, keepdims=True)
        )
        y_part = (
            y_part -
            y_part.mean(axis=1, keepdims=True)
        )
        numerator = np.sum(
            x_part * y_part,
            axis=1,
        )
        denominator = np.sqrt(
            np.sum(x_part ** 2, axis=1) *
            np.sum(y_part ** 2, axis=1)
        )
        correlations = numerator / np.maximum(
            denominator,
            1e-20,
        )
        temporal = float(np.mean(correlations))
        best_temporal = max(
            best_temporal,
            temporal,
        )

    if best_temporal == -1.0:
        return 0.0

    return float(np.clip(
        best_temporal * spatial,
        -1.0,
        1.0,
    ))

# ============================================================
# 6. FIND REPETITIVE WAVEFORMS
# ============================================================

def find_repetitive_intervals(
    raw,
    intervals,
    cfg,
):
    """
    Identify repetitive artifact waveforms.

    Short candidates are compared only with short ones.
    Long normalized candidates are compared only with
    other long normalized candidates.

    Returns
    -------
    repetitive : ndarray of bool
    counts : ndarray of int
    """

    n = len(intervals)
    if n == 0:
        return (
            np.array([], dtype=bool),
            np.array([], dtype=int),
        )
    picks = get_eeg_picks(raw, cfg)
    waveforms = [
        extract_waveform(
            raw,
            interval,
            picks,
            cfg,
        )
        for interval in intervals
    ]
    durations = np.array([
        end - start
        for start, end in intervals
    ])

    # Invalid waveforms do not count as repetitions
    similarity = np.zeros((n, n), dtype=float)
    for i in range(n):
        if waveforms[i] is not None:
            similarity[i, i] = 1.0
    for i in range(n):
        if waveforms[i] is None:
            continue
        for j in range(i + 1, n):
            if waveforms[j] is None:
                continue
            # Short and long groups are independent
            if (
                waveforms[i]["normalized"] !=
                waveforms[j]["normalized"]
            ):
                continue
            # Original duration-ratio restriction
            duration_ratio = max(
                durations[i],
                durations[j],
            ) / max(
                min(durations[i], durations[j]),
                1e-10,
            )
            if duration_ratio > cfg.max_duration_ratio:
                continue
            score = waveform_similarity(
                waveforms[i],
                waveforms[j],
                cfg,
            )
            similarity[i, j] = score
            similarity[j, i] = score
    matches = (
        similarity >= cfg.similarity_threshold
    )

    counts = matches.sum(axis=1)
    repetitive = (
        counts >= cfg.min_repetitions
    )

    return repetitive, counts

# ============================================================
# 7. EVENT-AWARE CLASSIFICATION
# ============================================================
def get_response_times(raw):
    """
    Extract response events from raw.annotations.extras.

    Response events are identified by:
        extras["Event"] == "Response"

    Returns
    -------
    response_times : np.ndarray
        Response onset times in seconds, relative to
        the beginning of the current Raw data.
    """

    annotations = raw.annotations

    if len(annotations) == 0:
        return np.array([], dtype=float)

    if annotations.extras is None:
        raise ValueError(
            "raw.annotations.extras is missing."
        )

    # Boolean mask aligned with annotations
    response_mask = np.array([
        isinstance(extra, dict)
        and str(extra.get("Event", "")).strip().lower() == "response"
        for extra in annotations.extras
    ], dtype=bool)

    # Annotation onsets are stored in the Raw annotation
    # time coordinate; convert to current Raw-relative time.
    response_times = (
        annotations.onset[response_mask]
        - raw.first_time
    )

    return np.asarray(response_times, dtype=float)

def overlaps_events(
    start,
    end,
    event_times,
    event_window,
):
    """
    Check whether an artifact interval overlaps
    any response-locked ERN epoch.
    """

    if len(event_times) == 0:
        return False

    return bool(np.any(
        (start < event_times + event_window[1]) &
        (end > event_times + event_window[0])
    ))

def is_erp_related(
    interval,
    response_times,
    cfg,
):
    """
    Check whether an artifact overlaps an ERN epoch.

    Only response-locked ERN intervals are protected.
    Flanker-locked and target-locked intervals are ignored.
    """

    start, end = interval

    return overlaps_events(
        start,
        end,
        response_times,
        cfg.response_window,
    )

# ============================================================
# 8. FULL ARTIFACT CLASSIFICATION
# ============================================================

def classify_all_artifacts(
    raw,
    all_intervals,
    severe_intervals,
    cfg,
):
    """
    Classify detected EEG artifact intervals.

    Classification criteria
    -----------------------
    1. Severe artifact overlap
    2. Waveform repetitiveness
    3. Response-locked ERN epoch overlap

    ERN events are identified using:
        raw.annotations.extras["Event"] == "Response"

    Returns
    -------
    report : pandas.DataFrame
        Classification results for all candidate intervals.
    """

    response_times = (
        get_response_times(raw)
        if cfg.protect_erp
        else np.array([], dtype=float)
    )

    repetitive, counts = find_repetitive_intervals(
        raw,
        all_intervals,
        cfg,
    )

    records = []

    for i, interval in enumerate(all_intervals):

        start, end = interval

        # Check whether the candidate overlaps
        # any severe artifact interval.
        is_severe = any(
            start < severe_end
            and end > severe_start
            for severe_start, severe_end in severe_intervals
        )

        # Check whether the candidate overlaps
        # a response-locked ERN epoch.
        ern_related = (
            cfg.protect_erp
            and is_erp_related(
                interval,
                response_times,
                cfg,
            )
        )

        # Waveform repetition classification
        is_repetitive = bool(repetitive[i])
        repetition_count = int(counts[i])

        if is_severe:
            if is_repetitive:
                if ern_related:
                    label = "REVIEW_severe_repetitive_ERN"
                else:
                    label = "REVIEW_severe_repetitive"
            elif ern_related:
                label = "REVIEW_severe_ERN"
            else:
                label = "BAD_severe"
        elif is_repetitive:
            if ern_related:
                label = "REVIEW_repetitive_ERN"
            else:
                label = "REVIEW_repetitive"
        else:
            if ern_related:
                label = "REVIEW_unique_ERN"
            elif cfg.auto_reject_unique:
                label = "BAD_unique"
            else:
                label = "REVIEW_unique"

        records.append({
            "start": float(start),
            "end": float(end),
            "duration": float(end - start),
            "severe": bool(is_severe),
            "repetitive": is_repetitive,
            "repetition_count": repetition_count,
            "ern_related": bool(ern_related),
            "label": label,
        })

    return pd.DataFrame(
        records,
        columns=[
            "start",
            "end",
            "duration",
            "severe",
            "repetitive",
            "repetition_count",
            "ern_related",
            "label",
        ],
    )

# ============================================================
# 9. MNE ANNOTATIONS
# ============================================================

def add_artifact_annotations(raw, report):

    cleaned = raw.copy()
    annotations = cleaned.annotations.copy()

    for row in report.itertuples():
        # MNE annotations use absolute raw timeline offset
        annotations.append(
            onset=cleaned.first_time + row.start,
            duration=row.duration,
            description=row.label,
        )
    cleaned.set_annotations(annotations)

    return cleaned

# ============================================================
# 10. SUMMARY
# ============================================================

def summarize_cleaning(raw, report):

    duration = raw.n_times / raw.info["sfreq"]
    bad_intervals = [
        (row.start, row.end)
        for row in report.itertuples()
        if row.label.startswith("BAD")
    ]
    bad_mask = intervals_to_mask(
        bad_intervals,
        raw.n_times,
        raw.info["sfreq"],
    )
    bad_duration = (
        bad_mask.sum() / raw.info["sfreq"]
    )

    summary = {
        "duration_sec": duration,
        "candidate_intervals": len(report),
        "bad_intervals": sum(
            report["label"].str.startswith("BAD")
        ),
        "review_intervals": sum(
            report["label"].str.startswith("REVIEW")
        ),
        "new_bad_duration_sec": bad_duration,
        "new_bad_percent": (
            100 * bad_duration / duration
        ),
    }

    return summary

# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def semi_auto_ica_cleaning(
    raw,
    cfg,
):
    """
    Input:
        Raw with EEG channel types correctly defined.

    Returns:
        raw_ica: Raw with artifact annotations
        report: classification DataFrame
        summary: quality summary dict
    """

    # 1. Candidate / severe detection
    all_intervals, severe_intervals = (
        detect_artifacts(raw, cfg)
    )

    # 2. Waveform analysis and classification
    report = classify_all_artifacts(
        raw,
        all_intervals,
        severe_intervals,
        cfg,
    )

    # 3. Add annotations
    raw_ica = add_artifact_annotations(
        raw,
        report,
    )

    # 4. Summary
    summary = summarize_cleaning(
        raw,
        report,
    )

    return raw_ica, report, summary

def manual_review_artifacts(
    annotated_raw,
) -> tuple[mne.io.BaseRaw, pd.DataFrame, dict]:
    """
    Interactively review artifact annotations.

    BAD_*:
        Excluded from ICA fitting.

    REVIEW_* / KEEP_*:
        Retained during ICA fitting.

    Changes made in the MNE Raw browser are saved
    after the browser window is closed.
    """

    # Work on a copy to preserve automatic detection results
    reviewed_raw = annotated_raw.copy()

    print("\n=== Manual artifact review ===")
    print("BAD_*    : Exclude from ICA")
    print("REVIEW_* : Retain for ICA, pending review")
    print("KEEP_*   : Manually accepted for ICA")
    print("\nPress 'a' to enter annotation mode.")
    print("Edit annotations and close the browser when finished.")

    # Open interactive Raw browser
    reviewed_raw.plot(
        duration=10.0,
        start=0.0,
        n_channels=30,
        scalings="auto",
        show=True,
        block=True,
        title="Manual artifact review for ICA",
    )

    # Browser has closed: annotations now contain manual edits
    annotations = reviewed_raw.annotations

    # Save a detailed annotation table
    records = []

    for i, (onset, duration, description) in enumerate(
        zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
        )
    ):
        # Exclude original task event annotations
        if not str(description).upper().startswith(
            ("BAD", "REVIEW", "KEEP")
        ):
            continue
        records.append({
            "annotation_index": i,
            "start": float(onset - reviewed_raw.first_time),
            "end": float(
                onset - reviewed_raw.first_time + duration
            ),
            "duration": float(duration),
            "label": str(description),
            "exclude_from_ica": str(description).lower().startswith("bad"),
        })
    review_report = pd.DataFrame(
        records,
        columns=[
            "annotation_index",
            "start",
            "end",
            "duration",
            "label",
            "exclude_from_ica",
        ],
    )

    label_counts = Counter(review_report["label"])

    bad_intervals = [
        (row.start, row.end)
        for row in review_report.itertuples()
        if row.exclude_from_ica
    ]

    # Merge overlapping BAD intervals to avoid double counting
    merged_bad = merge_intervals(
        bad_intervals,
        gap=0.0,
        min_duration=0.0,
    )

    bad_duration = sum(
        end - start
        for start, end in merged_bad
    )

    summary = {
        "total_annotations": len(review_report),
        "label_counts": dict(label_counts),
        "bad_annotations": int(
            review_report["exclude_from_ica"].sum()
        ),
        "bad_intervals": len(merged_bad),
        "bad_duration_sec": float(bad_duration),
        "bad_duration_percent": float(
            bad_duration / (reviewed_raw.n_times / reviewed_raw.info["sfreq"]) * 100
        ),
    }

    return reviewed_raw, review_report, summary

def main(
        raw: mne.io.BaseRaw,
        p_id: str,
        path_data: Path,
        path_result: Path,
) -> mne.io.BaseRaw:

    cfg_channel = PeriodQCConfig()
    cfg_detection = ICAPrep()

    raw_period, report_period = detect_period_artifacts(raw, cfg_channel)
    report_period.to_csv(
        path_result/p_id/"bad_period_results.tsv",
        sep="\t",
        index=False,
    )

    annotated_detection, report, summary = (
        semi_auto_ica_cleaning(
            raw_period,
            cfg_detection,
        )
    )
    raw_ica = raw_period.copy()
    # Transfer annotations (same sample timeline)
    raw_ica.set_annotations(
        annotated_detection.annotations.copy()
    )

    raw_reviewed, review_report, review_summary = manual_review_artifacts(raw_ica)
    review_report.to_csv(
        path_result/p_id/"artifact_results_reviewed.tsv",
        sep="\t",
        index=False,
    )
    # Save EEG with reviewed annotations
    raw_reviewed.save(
        path_data/f"{p_id}_task_Flanker_eeg_before_ica.fif",
        overwrite=True,
    )

    print("\nArtifact summary")
    for key, value in review_summary.items():
        print(f"{key}: {value}")
    print("\nLabel counts:")
    print(Counter(report["label"]))

    return raw_reviewed
