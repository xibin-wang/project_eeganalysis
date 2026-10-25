"""
Constant variables for all other scripts

"""

from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class PathConstants:
    SCRIPT_DIR: Path = Path(__file__).resolve().parent
    FOLDER_ROOT: Path = SCRIPT_DIR.parent

@dataclass(frozen=True)
class FilterParameters:
    # --------- Band pass parameters --------------
    line_freq: float=50.0
    lowpass_freq: float=0.1
    highpass_freq: float=80.0
    # --------- Power line noise detection parameters --------
    threshold_db: int=8
    min_fraction: float=0.2
    window_sec: float=4
    step_sec: float=2

@dataclass
class PeriodQCConfig:
    # Flat-zero detection
    zero_tolerance_v: float = 0.0
    zero_min_sec: float = 2.0

    # Sustained extreme PTP detection
    ptp_threshold_uv: float = 200
    ptp_window_sec: float = 0.2
    ptp_step_sec: float = 0.1
    ptp_min_sec: float = 3.0

    # Spatial assessment
    n_neighbors: int = 8
    correlation_threshold: float = 0.8
    neighbor_overlap_threshold: float = 0.5

    # Classification
    bad_period_fraction: float = 0.01
    bad_channel_fraction: float = 0.30
    shared_duration_threshold: float = 0.50

    # Intervals
    merge_gap_sec: float = 0.0

    # Excluded from screening
    exclude_channels: tuple = ("MASTl", "MASTr")

@dataclass(frozen=True)
class ICAPrep:
    # --------- Exclude channel ----------
    exclude_channels: tuple=("MASTl", "MASTr")

    # ---------- Detection windows ----------
    window_sec: float = 1.0
    step_sec: float = 0.10

    # ---------- Candidate detection ----------
    candidate_amp_uv: float = 200.0
    candidate_power_z: float = 4.0

    # ---------- Severe artifact detection ----------
    severe_amp_uv: float = 250.0
    severe_power_z: float = 5.0
    # At least this many channels must independently
    # satisfy BOTH amplitude and power conditions
    min_severe_channels: int = 1

    # ---------- High frequency ----------
    high_freq_band: tuple = ((30.0, 45.0), (55.0, 75.0))

    # ---------- Temporal filtering ----------
    min_duration: float = 0.20
    merge_gap: float = 0.0

    # ---------- Waveform similarity ----------
    similarity_threshold: float = 0.90
    min_repetitions: int = 3
    max_duration_ratio: float = 1.5
    max_similarity_channels: int = 8
    waveform_points: int = 200
    long_candidate_sec: float = 2.0
    max_time_shift_sec: float = 0.08
    min_overlap_fraction: float = 0.80

    # ---------- Event protection ----------
    response_window: tuple[float, float] = (-0.5, 0.5)

    # ---------- Automatic labeling ----------
    # Safer default: only severe candidates become BAD
    auto_reject_unique: bool = False
    # Protect ERP-related candidates from automatic
    protect_erp: bool = True

@dataclass(frozen=True)
class ICAParameters:
    # ---------- ICA results transfer ----------
    if_transfer: bool = True
    highpass_train: float = 1

    # ---------- ICALabel parameters ----------
    label_threshold: float = 0.9
    n_other_threshold: float = 0.5

@dataclass(frozen=True)
class GroupParameters:
    roi_channels: tuple = ("FCz",) # multiple channels are allowed
    win_analysis_s: tuple = (0, 0.1)
    alpha : float = 0.05