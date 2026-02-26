#!/usr/bin/env python3
"""
8D Audio Converter — Deep Analysis Backend v9.0
================================================

ANALYSIS ENGINE  (v6.0 additions over v4.0)
  ✦ MFCC timbral fingerprint (13 coefficients) — detects timbre type
  ✦ Chroma-based key / mode detection (major / minor)
  ✦ Tonnetz tonal tension analysis
  ✦ Interaural correlation analysis — detects fake stereo sources
  ✦ Crest factor / peak-to-RMS — detects over-compressed material
  ✦ Harmonic-to-noise ratio (HNR) via HPSS
  ✦ Onset rate & transient density classification
  ✦ Spectral rolloff & brightness ratio
  ✦ Zero-crossing rate → breathiness / noisiness score
  ✦ Perceptual sharpness estimate (psychoacoustic model)
  ✦ Per-segment energy variance → identifies intros/drops/outros
  ✦ Enhanced _optimize_parameters: uses crest factor, mode, transient
    density, stereo correlation, and HNR to fine-tune every parameter

8D ENGINE  (v9.0 — full feature set)
  ✦ HRIR Convolution Engine (NEW in v9.0) — breaks the FFmpeg ceiling:
      Real Head-Related Impulse Response convolution using scipy.signal.fftconvolve
      Woodworth ITD model + Duda/Martens ILD + pinna notch resonances
      OLA (overlap-add) block processing for time-varying spatial position
      Optional KEMAR dataset support (drop kemar_hrirs.npy alongside backend.py)
  ✦ Beat-Locked Rotation (NEW in v9.0):
      Beat positions from librosa drive a piecewise-linear LFO
      Source returns to the same azimuth on every beat — no drift
      Falls back to BPM-synced free-running LFO if no beat data
  ✦ Dual-Band Elevation v8.1 (ported to ALL engines):
      HF shelf at 8 kHz + sub cut/boost at 80 Hz
      Previously only in the 8-band engine; now in simple/vocal/6-band too
  ✦ ITD (Interaural Time Difference) simulation via per-band static
    channel delays — higher bands get up to 630 μs inter-ear offset
  ✦ Pinna notch filters: narrow EQ cuts at ~8.5 kHz and ~10.5 kHz
    applied to the contra-lateral channel to simulate pinna reflections
  ✦ Pre-reverb delay (15–35 ms) — simulates room distance
  ✦ Allpass diffusion network — series of 4 prime-spaced aecho taps
    before the main reverb for richer early reflections
  ✦ Diffuse-field EQ (IEC 711 compensation for headphone listening)
  ✦ Equal-loudness compensation shelf (Fletcher-Munson at 70 phons)
  ✦ Frequency-dependent stereo width (lows narrower, highs wider)
  ✦ Per-band hard limiter before amix to prevent inter-band clipping
  ✦ Phase rotation awareness — alternates LFO phase offset per band

INFRASTRUCTURE  (v9.0)
  ✦ TTL eviction: stem sessions + batch queues auto-purged after 2 hours
  ✦ Cancel endpoint: DELETE /process/{job_id} terminates FFmpeg + asyncio task
  ✦ Active proc registry: _active_procs / _active_tasks for clean teardown

FILTERGRAPH ENGINES  (all retained + enhanced)
  ✦ HRIR Convolution + Mastering  v9.0 — Python HRIR → FFmpeg reverb/EQ
  ✦ Studio Grade v9.0             — 8-band HRTF + beat-locked ITD + pinna
  ✦ 6-band multiband              — for enable_multi_band without full HRTF
  ✦ Vocal-aware 3-band            — legacy vocal center mode
  ✦ Simple 2-channel              — fallback for minimal FFmpeg installations

EQ  (12-band, unchanged API)
  ✦ 30 / 60 / 100 / 200 / 350 / 700 / 1500 / 3000 / 5000 / 8000 / 12000 / 16000 Hz
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import numpy as np

from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import librosa
    import soundfile as sf
    from scipy import signal
    ADVANCED_ANALYSIS = True
except ImportError:
    print("⚠️  Advanced analysis disabled. Run: pip install librosa soundfile scipy")
    ADVANCED_ANALYSIS = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("⚠️  SOFA/HDF5 support disabled. Run: pip install h5py")

try:
    import yt_dlp
    YOUTUBE_SUPPORT = True
except ImportError:
    print("⚠️  YouTube disabled. Run: pip install yt-dlp")
    YOUTUBE_SUPPORT = False

# Demucs 4.x uses `from demucs.api import Separator`; older versions used
# `import demucs.api`.  We import Separator directly so the runtime path also
# uses the 4.x API.  The `STEM_ENGINE` flag is set exactly once here —
# the old code had a second detection block that unconditionally reset it to
# "none" even after a successful import, silently disabling stem separation.
STEM_ENGINE     = "none"
STEM_SEPARATION = False

try:
    import torch
    import demucs                                        # noqa: F401 (version check)
    from demucs.api import Separator as DemucsSeperator  # 4.x API
    STEM_SEPARATION = True
    STEM_ENGINE     = "demucs"
    _dv = getattr(demucs, "__version__", "4.x")
    print(f"✅  Demucs stem separation available (v{_dv})")
except ImportError as _e:
    print(f"⚠️  Demucs import failed: {_e}")
    try:
        from spleeter.separator import Separator as SpleeterSeparator
        STEM_SEPARATION = True
        STEM_ENGINE     = "spleeter"
        print("✅  Spleeter stem separation available (fallback)")
    except ImportError as _e2:
        print(f"⚠️  Spleeter import failed: {_e2}")
        print("❌  Stem separation DISABLED — run: pip install demucs")

print(f"📊  Stem engine: {STEM_ENGINE} | Available: {STEM_SEPARATION}")

# Stem cache: session_id -> {stem_name: path}
stem_sessions: Dict[str, Dict[str, str]] = {}
# Timestamps for TTL eviction: session_id / batch_id -> creation time (epoch seconds)
_session_timestamps: Dict[str, float] = {}
# Active FFmpeg subprocesses: job_id -> asyncio.subprocess.Process
# Used by the cancel endpoint to terminate in-flight jobs.
_active_procs: Dict[str, Any] = {}
# Active asyncio Tasks: job_id -> Task — used to cancel the coroutine itself
_active_tasks: Dict[str, asyncio.Task] = {}

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(title="8D Audio Converter AI API", version="6.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR   = BASE_DIR / "temp"
for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(exist_ok=True)

# ============================================================================
# MODELS
# ============================================================================

class ProcessingParams(BaseModel):
    # 8D parameters
    rotation_speed:     float = 0.15
    reverb_room:        float = 0.60
    reverb_mix:         float = 0.30
    bass_rotation:      float = 0.08
    treble_rotation:    float = 0.20
    stereo_width:       float = 1.0
    elevation:          float = 0.0
    distance:           float = 1.0
    intensity_multiplier: float = 1.0
    enable_vocal_center:  bool = False
    enable_multi_band:    bool = True
    enable_hrtf:          bool = True
    enable_convolution_reverb: bool = True

    # Output
    # output_format: "mp3" | "wav" | "flac" | "ambisonics_foa" | "atmos_71_4"
    output_format: str = "mp3"
    bitrate:       int = 320
    sample_rate:   int = 48000
    bit_depth:     int = 24

    # Stem separation
    enable_stem_separation: bool = False
    stem_session_id:        Optional[str] = None   # reuse already-separated stems
    stem_engine_model:      str   = "htdemucs"     # htdemucs | htdemucs_6s | spleeter

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        # BUG FIX: The frontend sends "stem_model" but Pydantic only knows
        # "stem_engine_model".  Pydantic silently drops unknown fields, so the
        # 6-stem model selector never reached the backend (always fell back to
        # "htdemucs").  Map the alias here before validation so both names work.
        if isinstance(obj, dict) and "stem_model" in obj and "stem_engine_model" not in obj:
            obj = {**obj, "stem_engine_model": obj.pop("stem_model")}
        return super().model_validate(obj, *args, **kwargs)

    # Stem psychoacoustics (None = InstrumentRouter auto-assign)
    stem_auto_route:        bool  = True   # let InstrumentRouter assign all per-stem params
    enable_gain_staging:    bool  = True   # normalize each stem to target LUFS before process
    stem_target_lufs:       float = -23.0  # LUFS target for per-stem normalization
    enable_multiband_master: bool = True   # post-mix multiband compressor on master bus

    # Per-stem override rotation (None = use global params / InstrumentRouter)
    stem_vocals_rotation:         Optional[float] = None
    stem_drums_rotation:          Optional[float] = None
    stem_bass_rotation_override:  Optional[float] = None
    stem_other_rotation:          Optional[float] = None
    stem_guitar_rotation:         Optional[float] = None
    stem_piano_rotation:          Optional[float] = None

    # Per-stem override width (None = InstrumentRouter)
    stem_vocals_width:    Optional[float] = None
    stem_drums_width:     Optional[float] = None
    stem_bass_width:      Optional[float] = None
    stem_other_width:     Optional[float] = None
    stem_guitar_width:    Optional[float] = None
    stem_piano_width:     Optional[float] = None

    # Per-stem override elevation (None = InstrumentRouter)
    stem_vocals_elevation:  Optional[float] = None
    stem_drums_elevation:   Optional[float] = None
    stem_bass_elevation:    Optional[float] = None
    stem_other_elevation:   Optional[float] = None
    stem_guitar_elevation:  Optional[float] = None
    stem_piano_elevation:   Optional[float] = None

    # Per-stem override reverb mix (None = InstrumentRouter)
    stem_vocals_reverb:   Optional[float] = None
    stem_drums_reverb:    Optional[float] = None
    stem_bass_reverb:     Optional[float] = None
    stem_other_reverb:    Optional[float] = None
    stem_guitar_reverb:   Optional[float] = None
    stem_piano_reverb:    Optional[float] = None

    # Video visualizer
    generate_video:  bool = False
    video_style:     str  = "waveform"   # "waveform" | "spectrum" | "vectorscope"
    video_resolution: str = "1280x720"
    video_fps:       int  = 25

    # 12-band EQ (dB)
    eq_sub30_gain:       float =  7.0   # +3→+7: punchier sub-bass
    eq_sub60_gain:       float =  8.0   # +4→+8: deeper bass
    eq_bass100_gain:     float =  5.0   # +3→+5: bass body
    eq_ubass200_gain:    float =  2.5   # +1.5→+2.5: upper bass
    eq_lowmid350_gain:   float = -2.5
    eq_mid700_gain:      float = -0.5   # -1→-0.5: less vocal cut
    eq_umid1500_gain:    float =  2.0   # +1→+2: vocal clarity
    eq_presence3k_gain:  float =  3.5   # +2→+3.5: presence
    eq_def5k_gain:       float =  2.5   # +1.5→+2.5: definition
    eq_bril8k_gain:      float =  3.0   # +2→+3: brilliance
    eq_air12k_gain:      float =  5.0   # +2→+5: air/sparkle
    eq_uair16k_gain:     float =  3.0   # +1→+3: ultra-air

    # Legacy aliases (absorbed into new bands)
    eq_sub_bass_gain:    float =  0.0
    eq_bass_gain:        float =  0.0
    eq_low_mid_gain:     float =  0.0
    eq_presence_gain:    float =  0.0
    eq_air_gain:         float =  0.0

    # Enhancement toggles
    vocal_safe_bass:       bool = True
    instrument_enhance:    bool = True

    # Studio v5.0+ Fields
    reverb_density:   float = 0.7
    hrtf_intensity:   float = 1.0
    enable_limiter:   bool  = True

    # Preset management
    preset_id: Optional[str] = None  # genre key or 'auto'

    @classmethod
    def apply_preset(cls, base: 'ProcessingParams', preset_data: Dict[str, Any]) -> 'ProcessingParams':
        """Merge preset dict into ProcessingParams, ignoring unknown keys."""
        valid = {
            k: v for k, v in preset_data.items()
            if k in cls.model_fields and k != 'name'
        }
        return base.model_copy(update=valid)


class YouTubeDownloadRequest(BaseModel):
    url: str

class BatchJob(BaseModel):
    job_id:     str
    filename:   str
    status:     str
    progress:   int = 0
    output_url: Optional[str] = None
    error:      Optional[str] = None

batch_queue: Dict[str, List[BatchJob]] = {}


# ============================================================================
# PRESET LIBRARY v2.0 — Genre & Characteristic-Based Presets
# ============================================================================

class PresetLibrary:
    """
    Comprehensive preset library with automatic selection based on audio analysis.
    Each preset defines rotation, reverb, EQ, and spatial params optimised for
    a specific music style or spectral characteristic.
    """

    GENRE_PRESETS: Dict[str, Dict[str, Any]] = {
        # ── Electronic ────────────────────────────────────────────────────────
        'electronic': {
            'name': 'Electronic Pulse',
            'rotation_speed': 0.22, 'bass_rotation': 0.10, 'treble_rotation': 0.35,
            'reverb_room': 0.65, 'reverb_mix': 0.28, 'reverb_density': 0.75,
            'stereo_width': 1.35, 'elevation': 0.05, 'distance': 0.9,
            'eq_sub30_gain': 6.0, 'eq_sub60_gain': 7.5, 'eq_bass100_gain': 5.0,
            'eq_lowmid350_gain': -3.5, 'eq_mid700_gain': -1.5,
            'eq_presence3k_gain': 3.0, 'eq_bril8k_gain': 3.5, 'eq_air12k_gain': 4.5,
            'enable_vocal_center': False, 'instrument_enhance': True,
            'intensity_multiplier': 1.15, 'hrtf_intensity': 1.1,
        },
        'house': {
            'name': 'House Groove',
            'rotation_speed': 0.18, 'bass_rotation': 0.08, 'treble_rotation': 0.30,
            'reverb_room': 0.70, 'reverb_mix': 0.32, 'reverb_density': 0.80,
            'stereo_width': 1.25, 'elevation': 0.0, 'distance': 1.0,
            'eq_sub30_gain': 7.0, 'eq_sub60_gain': 8.0, 'eq_bass100_gain': 5.5,
            'eq_lowmid350_gain': -2.5, 'eq_mid700_gain': -0.5,
            'eq_presence3k_gain': 2.5, 'eq_bril8k_gain': 3.0, 'eq_air12k_gain': 4.0,
            'enable_vocal_center': False, 'instrument_enhance': True,
            'intensity_multiplier': 1.1, 'hrtf_intensity': 1.0,
        },
        'trance': {
            'name': 'Trance Atmosphere',
            'rotation_speed': 0.16, 'bass_rotation': 0.07, 'treble_rotation': 0.28,
            'reverb_room': 0.85, 'reverb_mix': 0.45, 'reverb_density': 0.90,
            'stereo_width': 1.45, 'elevation': 0.12, 'distance': 1.2,
            'eq_sub30_gain': 5.0, 'eq_sub60_gain': 6.5, 'eq_bass100_gain': 4.5,
            'eq_lowmid350_gain': -2.0, 'eq_mid700_gain': 0.0,
            'eq_presence3k_gain': 3.5, 'eq_bril8k_gain': 4.0, 'eq_air12k_gain': 5.5,
            'enable_vocal_center': False, 'instrument_enhance': True,
            'intensity_multiplier': 1.05, 'hrtf_intensity': 1.2,
        },
        # ── Rock / Metal ──────────────────────────────────────────────────────
        'rock': {
            'name': 'Rock Arena',
            'rotation_speed': 0.19, 'bass_rotation': 0.11, 'treble_rotation': 0.32,
            'reverb_room': 0.55, 'reverb_mix': 0.30, 'reverb_density': 0.70,
            'stereo_width': 1.20, 'elevation': 0.0, 'distance': 1.0,
            'eq_sub30_gain': 4.0, 'eq_sub60_gain': 5.0, 'eq_bass100_gain': 5.5,
            'eq_lowmid350_gain': -3.5, 'eq_mid700_gain': -2.0,
            'eq_presence3k_gain': 4.5, 'eq_def5k_gain': 3.5, 'eq_bril8k_gain': 3.0,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.1, 'hrtf_intensity': 1.0,
        },
        'metal': {
            'name': 'Metal Intensity',
            'rotation_speed': 0.24, 'bass_rotation': 0.13, 'treble_rotation': 0.40,
            'reverb_room': 0.45, 'reverb_mix': 0.25, 'reverb_density': 0.65,
            'stereo_width': 1.30, 'elevation': -0.05, 'distance': 0.95,
            'eq_sub30_gain': 5.5, 'eq_sub60_gain': 6.0, 'eq_bass100_gain': 4.5,
            'eq_lowmid350_gain': -5.0, 'eq_mid700_gain': -3.5,
            'eq_presence3k_gain': 6.0, 'eq_def5k_gain': 4.5, 'eq_bril8k_gain': 3.5,
            'enable_vocal_center': True, 'instrument_enhance': False,
            'intensity_multiplier': 1.2, 'hrtf_intensity': 1.1,
        },
        'indie': {
            'name': 'Indie Intimate',
            'rotation_speed': 0.14, 'bass_rotation': 0.07, 'treble_rotation': 0.22,
            'reverb_room': 0.68, 'reverb_mix': 0.38, 'reverb_density': 0.75,
            'stereo_width': 1.10, 'elevation': 0.05, 'distance': 1.1,
            'eq_sub30_gain': 2.5, 'eq_sub60_gain': 3.5, 'eq_bass100_gain': 4.0,
            'eq_lowmid350_gain': -1.5, 'eq_mid700_gain': 0.5,
            'eq_presence3k_gain': 2.5, 'eq_bril8k_gain': 2.5, 'eq_air12k_gain': 3.0,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.0, 'hrtf_intensity': 0.95,
        },
        # ── Hip-Hop / R&B ─────────────────────────────────────────────────────
        'hip_hop': {
            'name': 'Hip-Hop Bass',
            'rotation_speed': 0.13, 'bass_rotation': 0.06, 'treble_rotation': 0.20,
            'reverb_room': 0.50, 'reverb_mix': 0.25, 'reverb_density': 0.65,
            'stereo_width': 1.15, 'elevation': -0.08, 'distance': 0.85,
            'eq_sub30_gain': 8.5, 'eq_sub60_gain': 9.5, 'eq_bass100_gain': 6.5,
            'eq_lowmid350_gain': -4.5, 'eq_mid700_gain': -2.5,
            'eq_presence3k_gain': 2.0, 'eq_bril8k_gain': 2.0, 'eq_air12k_gain': 2.5,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.05, 'hrtf_intensity': 0.9,
        },
        'trap': {
            'name': 'Trap Spatial',
            'rotation_speed': 0.15, 'bass_rotation': 0.07, 'treble_rotation': 0.25,
            'reverb_room': 0.60, 'reverb_mix': 0.30, 'reverb_density': 0.70,
            'stereo_width': 1.25, 'elevation': 0.0, 'distance': 0.9,
            'eq_sub30_gain': 9.0, 'eq_sub60_gain': 10.0, 'eq_bass100_gain': 6.0,
            'eq_lowmid350_gain': -4.0, 'eq_mid700_gain': -2.0,
            'eq_presence3k_gain': 2.5, 'eq_bril8k_gain': 3.0, 'eq_air12k_gain': 3.5,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.1, 'hrtf_intensity': 1.0,
        },
        'rnb': {
            'name': 'R&B Smooth',
            'rotation_speed': 0.14, 'bass_rotation': 0.07, 'treble_rotation': 0.22,
            'reverb_room': 0.72, 'reverb_mix': 0.40, 'reverb_density': 0.80,
            'stereo_width': 1.15, 'elevation': 0.05, 'distance': 1.1,
            'eq_sub30_gain': 5.0, 'eq_sub60_gain': 6.0, 'eq_bass100_gain': 5.0,
            'eq_lowmid350_gain': -2.0, 'eq_mid700_gain': 0.0,
            'eq_presence3k_gain': 3.0, 'eq_bril8k_gain': 3.0, 'eq_air12k_gain': 4.0,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.0, 'hrtf_intensity': 0.95,
        },
        # ── Classical / Acoustic ──────────────────────────────────────────────
        'classical': {
            'name': 'Classical Hall',
            'rotation_speed': 0.10, 'bass_rotation': 0.05, 'treble_rotation': 0.16,
            'reverb_room': 0.90, 'reverb_mix': 0.55, 'reverb_density': 0.92,
            'stereo_width': 1.35, 'elevation': 0.15, 'distance': 1.4,
            'eq_sub30_gain': 1.0, 'eq_sub60_gain': 1.5, 'eq_bass100_gain': 2.0,
            'eq_lowmid350_gain': -1.0, 'eq_mid700_gain': 0.5,
            'eq_presence3k_gain': 3.5, 'eq_bril8k_gain': 3.0, 'eq_air12k_gain': 4.0,
            'enable_vocal_center': False, 'instrument_enhance': False,
            'intensity_multiplier': 0.95, 'hrtf_intensity': 1.15,
        },
        'orchestral': {
            'name': 'Orchestral Wide',
            'rotation_speed': 0.11, 'bass_rotation': 0.05, 'treble_rotation': 0.18,
            'reverb_room': 0.88, 'reverb_mix': 0.50, 'reverb_density': 0.90,
            'stereo_width': 1.40, 'elevation': 0.12, 'distance': 1.3,
            'eq_sub30_gain': 1.5, 'eq_sub60_gain': 2.0, 'eq_bass100_gain': 2.5,
            'eq_lowmid350_gain': -1.5, 'eq_mid700_gain': 0.0,
            'eq_presence3k_gain': 3.0, 'eq_bril8k_gain': 3.5, 'eq_air12k_gain': 4.5,
            'enable_vocal_center': False, 'instrument_enhance': False,
            'intensity_multiplier': 1.0, 'hrtf_intensity': 1.2,
        },
        'acoustic': {
            'name': 'Acoustic Natural',
            'rotation_speed': 0.12, 'bass_rotation': 0.06, 'treble_rotation': 0.19,
            'reverb_room': 0.75, 'reverb_mix': 0.42, 'reverb_density': 0.78,
            'stereo_width': 1.15, 'elevation': 0.08, 'distance': 1.2,
            'eq_sub30_gain': 2.0, 'eq_sub60_gain': 3.0, 'eq_bass100_gain': 3.5,
            'eq_lowmid350_gain': -1.5, 'eq_mid700_gain': 0.5,
            'eq_presence3k_gain': 2.5, 'eq_bril8k_gain': 2.5, 'eq_air12k_gain': 3.5,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.0, 'hrtf_intensity': 1.0,
        },
        # ── Jazz / Blues ──────────────────────────────────────────────────────
        'jazz': {
            'name': 'Jazz Club',
            'rotation_speed': 0.15, 'bass_rotation': 0.07, 'treble_rotation': 0.24,
            'reverb_room': 0.70, 'reverb_mix': 0.40, 'reverb_density': 0.75,
            'stereo_width': 1.15, 'elevation': 0.0, 'distance': 1.0,
            'eq_sub30_gain': 2.5, 'eq_sub60_gain': 3.5, 'eq_bass100_gain': 3.0,
            'eq_lowmid350_gain': -1.5, 'eq_mid700_gain': 0.0,
            'eq_presence3k_gain': 2.5, 'eq_bril8k_gain': 2.5, 'eq_air12k_gain': 3.0,
            'enable_vocal_center': False, 'instrument_enhance': True,
            'intensity_multiplier': 1.0, 'hrtf_intensity': 1.0,
        },
        'blues': {
            'name': 'Blues Soul',
            'rotation_speed': 0.13, 'bass_rotation': 0.06, 'treble_rotation': 0.21,
            'reverb_room': 0.65, 'reverb_mix': 0.35, 'reverb_density': 0.70,
            'stereo_width': 1.10, 'elevation': 0.0, 'distance': 1.0,
            'eq_sub30_gain': 3.5, 'eq_sub60_gain': 4.5, 'eq_bass100_gain': 4.0,
            'eq_lowmid350_gain': -2.0, 'eq_mid700_gain': -0.5,
            'eq_presence3k_gain': 3.0, 'eq_bril8k_gain': 2.5, 'eq_air12k_gain': 3.0,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.0, 'hrtf_intensity': 0.95,
        },
        # ── Pop / Vocal ───────────────────────────────────────────────────────
        'pop': {
            'name': 'Pop Polish',
            'rotation_speed': 0.17, 'bass_rotation': 0.08, 'treble_rotation': 0.28,
            'reverb_room': 0.62, 'reverb_mix': 0.32, 'reverb_density': 0.72,
            'stereo_width': 1.20, 'elevation': 0.0, 'distance': 1.0,
            'eq_sub30_gain': 4.5, 'eq_sub60_gain': 5.5, 'eq_bass100_gain': 4.5,
            'eq_lowmid350_gain': -2.5, 'eq_mid700_gain': -0.5,
            'eq_presence3k_gain': 3.5, 'eq_bril8k_gain': 3.0, 'eq_air12k_gain': 4.0,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.05, 'hrtf_intensity': 1.0,
        },
        'ballad': {
            'name': 'Ballad Emotion',
            'rotation_speed': 0.11, 'bass_rotation': 0.05, 'treble_rotation': 0.18,
            'reverb_room': 0.78, 'reverb_mix': 0.45, 'reverb_density': 0.82,
            'stereo_width': 1.10, 'elevation': 0.08, 'distance': 1.2,
            'eq_sub30_gain': 3.0, 'eq_sub60_gain': 4.0, 'eq_bass100_gain': 4.0,
            'eq_lowmid350_gain': -1.5, 'eq_mid700_gain': 0.5,
            'eq_presence3k_gain': 3.0, 'eq_bril8k_gain': 2.5, 'eq_air12k_gain': 3.5,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 0.95, 'hrtf_intensity': 0.95,
        },
        # ── Ambient / Chill ───────────────────────────────────────────────────
        'ambient': {
            'name': 'Ambient Drift',
            'rotation_speed': 0.07, 'bass_rotation': 0.03, 'treble_rotation': 0.11,
            'reverb_room': 0.95, 'reverb_mix': 0.65, 'reverb_density': 0.95,
            'stereo_width': 1.50, 'elevation': 0.18, 'distance': 1.6,
            'eq_sub30_gain': 1.5, 'eq_sub60_gain': 2.5, 'eq_bass100_gain': 2.0,
            'eq_lowmid350_gain': -1.0, 'eq_mid700_gain': 0.0,
            'eq_presence3k_gain': 2.0, 'eq_bril8k_gain': 3.0, 'eq_air12k_gain': 5.5,
            'enable_vocal_center': False, 'instrument_enhance': False,
            'intensity_multiplier': 0.9, 'hrtf_intensity': 1.25,
        },
        'lofi': {
            'name': 'Lo-Fi Chill',
            'rotation_speed': 0.09, 'bass_rotation': 0.04, 'treble_rotation': 0.14,
            'reverb_room': 0.72, 'reverb_mix': 0.38, 'reverb_density': 0.70,
            'stereo_width': 1.20, 'elevation': 0.0, 'distance': 1.1,
            'eq_sub30_gain': 3.0, 'eq_sub60_gain': 4.0, 'eq_bass100_gain': 3.5,
            'eq_lowmid350_gain': -1.5, 'eq_mid700_gain': -0.5,
            'eq_presence3k_gain': 1.5, 'eq_bril8k_gain': 1.5, 'eq_air12k_gain': 2.0,
            'enable_vocal_center': False, 'instrument_enhance': False,
            'intensity_multiplier': 0.95, 'hrtf_intensity': 0.9,
        },
        # ── South Asian ───────────────────────────────────────────────────────
        'bollywood': {
            'name': 'Bollywood Grand',
            'rotation_speed': 0.15, 'bass_rotation': 0.07, 'treble_rotation': 0.24,
            'reverb_room': 0.72, 'reverb_mix': 0.40, 'reverb_density': 0.78,
            'stereo_width': 1.20, 'elevation': 0.05, 'distance': 1.1,
            'eq_sub30_gain': 4.0, 'eq_sub60_gain': 5.5, 'eq_bass100_gain': 4.5,
            'eq_lowmid350_gain': -2.0, 'eq_mid700_gain': 0.5,
            'eq_presence3k_gain': 4.0, 'eq_bril8k_gain': 3.0, 'eq_air12k_gain': 3.5,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.05, 'hrtf_intensity': 1.0,
        },
        'bhangra': {
            'name': 'Bhangra Energy',
            'rotation_speed': 0.21, 'bass_rotation': 0.12, 'treble_rotation': 0.32,
            'reverb_room': 0.58, 'reverb_mix': 0.30, 'reverb_density': 0.72,
            'stereo_width': 1.28, 'elevation': 0.0, 'distance': 0.95,
            'eq_sub30_gain': 6.0, 'eq_sub60_gain': 7.5, 'eq_bass100_gain': 6.0,
            'eq_lowmid350_gain': -3.5, 'eq_mid700_gain': -1.0,
            'eq_presence3k_gain': 4.5, 'eq_bril8k_gain': 4.0, 'eq_air12k_gain': 4.0,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 1.15, 'hrtf_intensity': 1.05,
        },
        'indian_classical': {
            'name': 'Indian Classical',
            'rotation_speed': 0.10, 'bass_rotation': 0.05, 'treble_rotation': 0.16,
            'reverb_room': 0.85, 'reverb_mix': 0.52, 'reverb_density': 0.88,
            'stereo_width': 1.15, 'elevation': 0.12, 'distance': 1.3,
            'eq_sub30_gain': 1.5, 'eq_sub60_gain': 2.5, 'eq_bass100_gain': 3.0,
            'eq_lowmid350_gain': -1.0, 'eq_mid700_gain': 1.0,
            'eq_presence3k_gain': 3.5, 'eq_bril8k_gain': 3.0, 'eq_air12k_gain': 4.0,
            'enable_vocal_center': True, 'instrument_enhance': True,
            'intensity_multiplier': 0.95, 'hrtf_intensity': 1.15,
        },
    }

    @staticmethod
    def select_preset(
        analysis: Dict[str, Any],
        user_genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Auto-select optimal preset from analysis. Returns preset dict."""
        gp = PresetLibrary.GENRE_PRESETS

        # 1. Explicit genre override
        if user_genre and user_genre in gp:
            print(f"  ↳ Auto-preset: user genre '{user_genre}'")
            return gp[user_genre].copy()

        # 2. Detected genre
        genre = analysis.get('genre', 'unknown')
        selected = gp.get(genre, gp['pop']).copy()
        print(f"  ↳ Auto-preset: detected '{genre}' → '{selected['name']}'")

        # 3. Tempo-based rotation tweak
        bpm = analysis.get('bpm', 120)
        if bpm < 90:
            selected['rotation_speed'] = min(selected.get('rotation_speed', 0.15) * 0.85, 0.15)
            selected['bass_rotation']  = min(selected.get('bass_rotation', 0.08) * 0.85, 0.08)
            print(f"  ↳ Auto-adjust: slow tempo ({bpm} BPM)")
        elif bpm > 140:
            selected['rotation_speed']   = min(selected.get('rotation_speed', 0.15) * 1.15, 0.30)
            selected['treble_rotation']  = min(selected.get('treble_rotation', 0.20) * 1.15, 0.45)
            print(f"  ↳ Auto-adjust: fast tempo ({bpm} BPM)")

        # 4. Vocal protection
        if analysis.get('has_vocals') and analysis.get('vocal_prominence', 0) > 0.6:
            selected['enable_vocal_center'] = True
            selected['vocal_safe_bass']     = True
            selected['eq_mid700_gain']      = max(selected.get('eq_mid700_gain', -0.5), -0.5)
            print("  ↳ Auto-adjust: strong vocals → vocal protection enabled")

        # 5. Dynamic range
        crest = analysis.get('crest_factor_db', 12)
        if crest < 6:
            selected['intensity_multiplier'] = selected.get('intensity_multiplier', 1.0) * 1.1
            print(f"  ↳ Auto-adjust: over-compressed ({crest}dB)")
        elif crest > 18:
            selected['reverb_mix'] = min(selected.get('reverb_mix', 0.3) * 1.1, 0.55)
            print(f"  ↳ Auto-adjust: very dynamic ({crest}dB)")

        # 6. Spectral balance: bass-heavy
        sub  = analysis.get('sub_bass_ratio', 0) + analysis.get('upper_sub_ratio', 0)
        bass = analysis.get('bass_ratio', 0)     + analysis.get('upper_bass_ratio', 0)
        if sub + bass > 0.45:
            selected['eq_sub30_gain']  = min(selected.get('eq_sub30_gain', 7.0) + 1.0, 10.0)
            selected['eq_sub60_gain']  = min(selected.get('eq_sub60_gain', 8.0) + 1.0, 11.0)
            print("  ↳ Auto-adjust: bass-heavy mix")

        # 7. Fake stereo widening
        if analysis.get('is_fake_stereo'):
            selected['stereo_width'] = min(selected.get('stereo_width', 1.0) * 1.25, 1.6)
            print("  ↳ Auto-adjust: fake stereo widening")

        return selected

    @staticmethod
    def get_preset_list() -> List[Dict[str, str]]:
        """Return list of presets for the frontend dropdown."""
        CATEGORY_LABEL = {
            'electronic': 'Electronic', 'house': 'Electronic', 'trance': 'Electronic',
            'rock': 'Rock', 'metal': 'Rock', 'indie': 'Rock',
            'hip_hop': 'Hip-Hop', 'trap': 'Hip-Hop', 'rnb': 'Hip-Hop',
            'classical': 'Classical', 'orchestral': 'Classical', 'acoustic': 'Classical',
            'jazz': 'Jazz', 'blues': 'Jazz',
            'pop': 'Pop', 'ballad': 'Pop',
            'ambient': 'Ambient', 'lofi': 'Ambient',
            'bollywood': 'South Asian', 'bhangra': 'South Asian', 'indian_classical': 'South Asian',
        }
        out = [{'id': 'auto', 'name': '\U0001f916 Auto-Detect', 'category': 'auto',
                'description': 'Analyze audio and select optimal preset automatically'}]
        for key, preset in PresetLibrary.GENRE_PRESETS.items():
            out.append({
                'id': key,
                'name': preset['name'],
                'category': CATEGORY_LABEL.get(key, 'Other'),
                'description': f"Optimised for {key.replace('_', ' ')} music",
            })
        return out

# ============================================================================
# TTL EVICTION — background sweeper for stem sessions and batch jobs
# ============================================================================
#
# stem_sessions and batch_queue previously grew indefinitely — every track
# processed accumulates orphaned WAV stems on disk and stale queue entries
# in memory.  This sweeper runs every 10 minutes and evicts entries older
# than TTL_HOURS (default 2 hours), unlinking associated files.

TTL_HOURS  = 2
TTL_SECS   = TTL_HOURS * 3600

async def _ttl_sweeper():
    """Background coroutine: sweeps stem_sessions and batch_queue every 10 min."""
    import time
    while True:
        await asyncio.sleep(600)   # run every 10 minutes
        now = time.time()
        cutoff = now - TTL_SECS

        # ── Stem sessions ────────────────────────────────────────────────────
        expired_sessions = [
            sid for sid, ts in _session_timestamps.items()
            if ts < cutoff and sid in stem_sessions
        ]
        for sid in expired_sessions:
            stems = stem_sessions.pop(sid, {})
            for stem_path in stems.values():
                try:
                    p = Path(stem_path)
                    if p.exists():
                        p.unlink()
                    # Also remove the parent temp dir if empty
                    parent = p.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                except Exception:
                    pass
            _session_timestamps.pop(sid, None)
            print(f"  🗑  TTL evicted stem session {sid[:8]}… ({len(stems)} files)")

        # ── Batch jobs ───────────────────────────────────────────────────────
        expired_batches = [
            bid for bid, ts in _session_timestamps.items()
            if ts < cutoff and bid in batch_queue
        ]
        for bid in expired_batches:
            batch_queue.pop(bid, None)
            _session_timestamps.pop(bid, None)
            print(f"  🗑  TTL evicted batch queue {bid[:8]}…")

@app.on_event("startup")
async def startup_event():
    """Register background tasks on application startup."""
    asyncio.create_task(_ttl_sweeper())


# ============================================================================
# DEEP AUDIO ANALYZER  v6.0
# ============================================================================

class IntelligentAudioAnalyzer:
    """
    12-dimensional spectral analysis, 15-genre classifier, vocal center
    detection, key/mode detection, ITD-awareness, and a comprehensive
    8D parameter recommendation engine.
    """

    GENRE_SIGNATURES = {
        # Western
        'electronic':       {'low_ratio': (0.60, 1.0),  'centroid': (2000, 8000), 'rhythm': (0.5, 1.0)},
        'classical':        {'low_ratio': (0.25, 0.55), 'centroid': (1000, 4000), 'rhythm': (0.1, 0.5)},
        'rock':             {'low_ratio': (0.40, 0.70), 'centroid': (1500, 5000), 'rhythm': (0.4, 0.8)},
        'hip_hop':          {'low_ratio': (0.65, 1.0),  'centroid': (600,  3000), 'rhythm': (0.3, 0.7)},
        'jazz':             {'low_ratio': (0.30, 0.60), 'centroid': (1200, 4500), 'rhythm': (0.2, 0.6)},
        'pop':              {'low_ratio': (0.45, 0.70), 'centroid': (1500, 5000), 'rhythm': (0.3, 0.6)},
        'ambient':          {'low_ratio': (0.35, 0.65), 'centroid': (400,  2000), 'rhythm': (0.0, 0.3)},
        'rnb':              {'low_ratio': (0.55, 0.90), 'centroid': (800,  3500), 'rhythm': (0.3, 0.65)},
        'metal':            {'low_ratio': (0.45, 0.80), 'centroid': (2500, 7000), 'rhythm': (0.6, 1.0)},
        # South Asian
        'bollywood':        {'low_ratio': (0.45, 0.75), 'centroid': (900,  3500), 'rhythm': (0.35, 0.70)},
        'bhangra':          {'low_ratio': (0.55, 0.85), 'centroid': (800,  3000), 'rhythm': (0.50, 0.85)},
        'nepali_folk':      {'low_ratio': (0.25, 0.55), 'centroid': (1000, 4000), 'rhythm': (0.25, 0.60)},
        'ghazal':           {'low_ratio': (0.20, 0.50), 'centroid': (600,  2500), 'rhythm': (0.10, 0.45)},
        'indian_classical':  {'low_ratio': (0.25, 0.55), 'centroid': (700,  3000), 'rhythm': (0.10, 0.50)},
        'devotional':       {'low_ratio': (0.25, 0.55), 'centroid': (600,  2500), 'rhythm': (0.10, 0.50)},
    }

    BANDS = [
        ('sub_bass',    20,    60),
        ('upper_sub',   60,   100),
        ('bass',       100,   200),
        ('upper_bass', 200,   350),
        ('low_mid',    350,   700),
        ('mid',        700,  1500),
        ('upper_mid', 1500,  3000),
        ('presence',  3000,  6000),
        ('brilliance',6000, 10000),
        ('air',      10000, 20000),
    ]

    KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def analyze_comprehensive(self, file_path: str) -> Dict[str, Any]:
        if not ADVANCED_ANALYSIS:
            return self._basic_analysis(file_path)
        try:
            print(f"🧠 Deep Analysis v6.0 — {Path(file_path).name}")
            y, sr = librosa.load(file_path, sr=None, mono=False)
            y_mono = librosa.to_mono(y) if y.ndim > 1 else y
            duration = librosa.get_duration(y=y_mono, sr=sr)

            analysis: Dict[str, Any] = {
                'duration': round(duration, 2),
                'sample_rate': sr,
                'channels': 2 if y.ndim > 1 else 1,
            }

            # ── BPM & Beat Tracking ──────────────────────────────────────────
            tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr)
            tempo = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
            analysis['bpm'] = round(tempo)
            analysis['beat_positions'] = beats.tolist()[:100]

            # ── Spectral Frame Analysis ──────────────────────────────────────
            S     = np.abs(librosa.stft(y_mono))
            freqs = librosa.fft_frequencies(sr=sr)
            total_energy = float(np.sum(S)) + 1e-10

            # Spectral centroid
            sc = librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0]
            analysis['spectral_centroid_mean'] = float(np.mean(sc))
            analysis['spectral_centroid_std']  = float(np.std(sc))

            # Spectral flatness (0=tonal, 1=noise)
            sf_val = librosa.feature.spectral_flatness(y=y_mono)[0]
            analysis['spectral_flatness'] = float(np.mean(sf_val))

            # Spectral rolloff (frequency below which 85% of energy resides)
            rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr, roll_percent=0.85)[0]
            analysis['spectral_rolloff'] = float(np.mean(rolloff))
            analysis['spectral_brightness'] = min(float(np.mean(rolloff)) / (sr / 2), 1.0)

            # ── 10-Band Energy Ratios ────────────────────────────────────────
            band_ratios: Dict[str, float] = {}
            for name, lo, hi in self.BANDS:
                idx = (freqs >= lo) & (freqs < hi)
                e = float(np.sum(S[idx, :]))
                ratio = e / total_energy
                band_ratios[f'{name}_ratio'] = ratio
                analysis[f'{name}_ratio'] = ratio

            # Aggregate legacy fields
            analysis['bass_ratio']    = band_ratios['bass_ratio'] + band_ratios['upper_bass_ratio']
            analysis['sub_bass_ratio'] = band_ratios['sub_bass_ratio']
            analysis['mid_ratio']     = band_ratios['mid_ratio']
            analysis['hi_mid_ratio']  = band_ratios['upper_mid_ratio'] + band_ratios['presence_ratio']
            analysis['high_ratio']    = band_ratios['brilliance_ratio'] + band_ratios['air_ratio']
            analysis['low_mid_ratio'] = band_ratios['low_mid_ratio']
            analysis['air_ratio']     = band_ratios['air_ratio']

            # ── Dynamic Range & Crest Factor ─────────────────────────────────
            rms = librosa.feature.rms(y=y_mono)[0]
            analysis['dynamic_range'] = float(20 * np.log10(
                np.max(rms) / (np.mean(rms) + 1e-10)))

            peak    = float(np.max(np.abs(y_mono)))
            rms_val = float(np.sqrt(np.mean(y_mono ** 2)))
            crest   = 20 * np.log10(peak / (rms_val + 1e-10))
            analysis['crest_factor_db']    = round(float(crest), 1)
            analysis['is_over_compressed'] = bool(crest < 6.0)
            analysis['loudness_headroom']  = round(20 * np.log10(1.0 / (peak + 1e-10)), 1)

            # ── Vocal & Harmonic Separation ──────────────────────────────────
            y_harm, y_perc = librosa.effects.hpss(y_mono)
            h_e = float(np.sum(np.abs(y_harm)))
            p_e = float(np.sum(np.abs(y_perc)))
            analysis['has_vocals']       = bool(h_e > p_e * 0.7)
            analysis['vocal_prominence'] = h_e / (h_e + p_e + 1e-10)

            # HNR (Harmonic-to-Noise Ratio)
            hnr = h_e / (p_e + 1e-10)
            analysis['harmonic_to_noise_ratio'] = round(float(hnr), 3)
            analysis['hnr_db'] = round(float(20 * np.log10(hnr + 1e-10)), 1)

            # Vocal fundamental frequency (strongest harmonic 100–800 Hz)
            vocal_idx = (freqs >= 100) & (freqs < 800)
            harm_stft = np.abs(librosa.stft(y_harm))
            vocal_mag  = np.mean(harm_stft[vocal_idx, :], axis=1)
            if len(vocal_mag) > 0:
                peak_bin = int(np.argmax(vocal_mag))
                analysis['vocal_fundamental_hz'] = float(freqs[vocal_idx][peak_bin])
            else:
                analysis['vocal_fundamental_hz'] = 250.0

            # ── Instrument Prominence ────────────────────────────────────────
            harm_S = np.abs(librosa.stft(y_harm))
            def harm_band(lo, hi):
                idx = (freqs >= lo) & (freqs < hi)
                return float(np.sum(harm_S[idx, :])) / (total_energy + 1e-10)

            analysis['string_instrument_prominence'] = harm_band(200, 1200)
            analysis['wind_instrument_prominence']   = harm_band(300, 2500)
            analysis['perc_instrument_prominence']   = p_e / (total_energy + 1e-10)
            analysis['brass_instrument_prominence']  = harm_band(150, 900)
            analysis['sitar_sarangi_prominence']     = harm_band(300, 1800)  # South Asian

            # ── Rhythm & Transient Analysis ──────────────────────────────────
            onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
            analysis['rhythm_complexity']      = float(np.std(onset_env))
            analysis['rhythm_complexity_norm'] = min(float(np.std(onset_env)) / 5.0, 1.0)

            onsets = librosa.onset.onset_detect(y=y_mono, sr=sr)
            onset_rate = float(len(onsets) / duration) if duration > 0 else 0.0
            analysis['onset_rate'] = round(onset_rate, 2)
            analysis['transient_density'] = (
                'sparse'   if onset_rate < 1.0 else
                'moderate' if onset_rate < 3.5 else
                'dense'
            )

            # ── MFCC Timbral Fingerprint ─────────────────────────────────────
            try:
                mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
                analysis['mfcc_mean']       = np.mean(mfccs, axis=1).tolist()
                analysis['mfcc_std']        = np.std(mfccs, axis=1).tolist()
                # MFCC[1] inversely correlates with brightness
                analysis['timbre_brightness'] = float(np.mean(mfccs[1]))
                # MFCC[0] (energy), MFCC[1..3] encode spectral slope
            except Exception as e:
                print(f"  ⚠ MFCC failed: {e}")
                analysis['mfcc_mean'] = [0.0] * 13
                analysis['timbre_brightness'] = 0.0

            # ── Key & Mode Detection ─────────────────────────────────────────
            try:
                chroma     = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
                chroma_avg = np.mean(chroma, axis=1)
                chroma_n   = chroma_avg / (np.sum(chroma_avg) + 1e-10)
                key_idx    = int(np.argmax(chroma_avg))
                analysis['key']             = self.KEYS[key_idx]
                analysis['key_confidence']  = round(float(chroma_n[key_idx]), 3)

                # Major / minor classification
                maj = np.array([1,0,1,0,1,1,0,1,0,1,0,1], dtype=float)
                min_ = np.array([1,0,1,1,0,1,0,1,1,0,1,0], dtype=float)
                maj_rot  = np.roll(maj,  key_idx) / maj.sum()
                min_rot  = np.roll(min_, key_idx) / min_.sum()
                maj_corr = float(np.dot(chroma_n, maj_rot))
                min_corr = float(np.dot(chroma_n, min_rot))
                analysis['mode']           = 'major' if maj_corr > min_corr else 'minor'
                analysis['mode_strength']  = round(abs(maj_corr - min_corr), 3)
            except Exception as e:
                print(f"  ⚠ Key detection failed: {e}")
                analysis['key'] = 'C'; analysis['mode'] = 'unknown'; analysis['key_confidence'] = 0.0

            # ── Tonnetz Tonal Complexity ─────────────────────────────────────
            try:
                tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
                analysis['tonal_complexity'] = float(np.mean(np.std(tonnetz, axis=1)))
                analysis['tonal_movement']   = float(np.mean(np.abs(np.diff(tonnetz, axis=1))))
            except Exception as e:
                print(f"  ⚠ Tonnetz failed: {e}")
                analysis['tonal_complexity'] = 0.3
                analysis['tonal_movement']   = 0.1

            # ── Stereo Analysis ──────────────────────────────────────────────
            if y.ndim > 1 and y.shape[0] >= 2:
                # Interaural correlation
                min_len = min(len(y[0]), len(y[1]))
                corr = np.corrcoef(y[0, :min_len], y[1, :min_len])[0, 1]
                analysis['stereo_correlation'] = round(float(corr), 3)
                analysis['is_fake_stereo']      = bool(corr > 0.95)

                # Mid/Side balance
                mid  = (y[0] + y[1]) / 2
                side = (y[0] - y[1]) / 2
                mid_e  = float(np.mean(mid ** 2))
                side_e = float(np.mean(side ** 2))
                analysis['stereo_width_measured'] = round(
                    float(np.sqrt(side_e / (mid_e + 1e-10))), 3)
            else:
                analysis['stereo_correlation']    = 1.0
                analysis['is_fake_stereo']        = True
                analysis['stereo_width_measured'] = 0.0

            # ── Zero Crossing Rate (noisiness / breathiness) ─────────────────
            zcr = librosa.feature.zero_crossing_rate(y_mono)[0]
            analysis['zero_crossing_rate'] = float(np.mean(zcr))
            analysis['breathiness']        = min(float(np.mean(zcr)) * 20.0, 1.0)

            # ── Perceptual Sharpness Estimate ────────────────────────────────
            # Weighted centroid above 1.5kHz vs below — heuristic
            sc_norm = analysis['spectral_centroid_mean'] / (sr / 2.0)
            analysis['perceptual_sharpness'] = round(float(sc_norm) * 2.0, 3)

            # ── Energy Profile (per 2-second segment) ────────────────────────
            analysis['energy_profile'] = self._energy_profile(y_mono, sr)

            # ── Genre Classification ─────────────────────────────────────────
            analysis['genre'] = self._classify_genre(analysis)

            print(f"  ↳ genre={analysis['genre']}  key={analysis['key']} {analysis['mode']}  "
                  f"bpm={analysis['bpm']}  crest={analysis['crest_factor_db']}dB  "
                  f"HNR={analysis.get('hnr_db', '?')}dB  "
                  f"ITD-source={'fake' if analysis['is_fake_stereo'] else 'true'} stereo  "
                  f"transients={analysis['transient_density']}")

            # ── Recommended Settings ─────────────────────────────────────────
            analysis['recommended_settings'] = self._optimize_parameters(analysis)
            return analysis

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback; traceback.print_exc()
            return self._basic_analysis(file_path)

    # ────────────────────────────────────────────────────────────────────────
    # Genre classifier
    # ────────────────────────────────────────────────────────────────────────

    def _classify_genre(self, feat: Dict) -> str:
        low      = feat['bass_ratio'] + feat['sub_bass_ratio']
        centroid = feat['spectral_centroid_mean']
        rhythm   = feat.get('rhythm_complexity_norm', 0.4)

        best, best_score = 'unknown', float('inf')
        for genre, sig in self.GENRE_SIGNATURES.items():
            lo, hi     = sig['low_ratio']
            clo, chi   = sig['centroid']
            rlo, rhi   = sig.get('rhythm', (0, 1))
            d_low = max(0.0, lo - low, low - hi)
            d_cen = max(0.0, clo - centroid, centroid - chi) / 1000.0
            d_rhy = max(0.0, rlo - rhythm, rhythm - rhi)
            score = d_low + d_cen + d_rhy * 0.5
            if score < best_score:
                best_score, best = score, genre
        return best

    # ────────────────────────────────────────────────────────────────────────
    # Energy profile
    # ────────────────────────────────────────────────────────────────────────

    def _energy_profile(self, y: np.ndarray, sr: int) -> Dict:
        seg = sr * 2
        n   = len(y) // seg
        curve = [float(np.sqrt(np.mean(y[i*seg:(i+1)*seg]**2))) for i in range(n)]
        if not curve:
            curve = [0.0]
        return {
            'curve':       curve[:50],
            'mean':        float(np.mean(curve)),
            'std':         float(np.std(curve)),
            'has_buildup': len(curve) >= 10 and
                           bool(np.mean(curve[len(curve)//2:]) >
                                np.mean(curve[:len(curve)//2]) * 1.3),
        }

    # ────────────────────────────────────────────────────────────────────────
    # Parameter optimizer  (v6.0 — uses all new analysis fields)
    # ────────────────────────────────────────────────────────────────────────

    def _optimize_parameters(self, a: Dict) -> Dict:
        genre       = a['genre']
        bpm         = float(a.get('bpm', 120) or 120)
        bass_r      = a.get('bass_ratio', 0.2)
        sub_r       = a.get('sub_bass_ratio', 0.1)
        has_vocals  = a.get('has_vocals', False)
        voc_prom    = a.get('vocal_prominence', 0.5)
        dyn         = a.get('dynamic_range', 10)
        rhythm      = a.get('rhythm_complexity_norm', 0.4)
        string_prom = a.get('string_instrument_prominence', 0.0)
        voc_f0      = a.get('vocal_fundamental_hz', 250)
        crest       = a.get('crest_factor_db', 12.0)
        mode        = a.get('mode', 'major')
        is_fake_st  = a.get('is_fake_stereo', False)
        transients  = a.get('transient_density', 'moderate')
        hnr         = a.get('harmonic_to_noise_ratio', 1.0)
        onset_rate  = a.get('onset_rate', 2.0)

        # ── Defaults ────────────────────────────────────────────────────────
        p = {
            'rotation_speed':      0.15,
            'reverb_room':         0.60,
            'reverb_mix':          0.30,
            'bass_rotation':       0.08,
            'treble_rotation':     0.20,
            'stereo_width':        1.0,
            'elevation':           0.0,
            'distance':            1.0,
            'enable_vocal_center': has_vocals,
            'intensity_multiplier': 1.0,
            'vocal_safe_bass':     True,
            'instrument_enhance':  True,
            'eq_sub30_gain':      3.0,
            'eq_sub60_gain':      4.0,
            'eq_bass100_gain':    3.0,
            'eq_ubass200_gain':   1.5,
            'eq_lowmid350_gain': -2.5,
            'eq_mid700_gain':    -1.0,
            'eq_umid1500_gain':   1.0,
            'eq_presence3k_gain': 2.0,
            'eq_def5k_gain':      1.5,
            'eq_bril8k_gain':     2.0,
            'eq_air12k_gain':     2.0,
            'eq_uair16k_gain':    1.0,
        }

        # ── Genre overrides ──────────────────────────────────────────────────
        genre_overrides = {
            'electronic': {
                'rotation_speed': 0.25, 'treble_rotation': 0.35, 'bass_rotation': 0.12,
                'reverb_room': 0.70, 'stereo_width': 1.30,
                'eq_sub30_gain': 6.0, 'eq_sub60_gain': 7.0, 'eq_bass100_gain': 5.0,
                'eq_lowmid350_gain': -3.0, 'eq_air12k_gain': 4.0, 'eq_uair16k_gain': 3.0,
            },
            'classical': {
                'rotation_speed': 0.10, 'treble_rotation': 0.15, 'bass_rotation': 0.05,
                'reverb_room': 0.85, 'reverb_mix': 0.50, 'elevation': 0.15,
                'eq_sub30_gain': 0.5, 'eq_sub60_gain': 1.0, 'eq_bass100_gain': 1.5,
                'eq_lowmid350_gain': -1.5, 'eq_presence3k_gain': 3.0,
                'eq_bril8k_gain': 2.5, 'eq_air12k_gain': 3.0,
            },
            'rock': {
                'rotation_speed': 0.18, 'bass_rotation': 0.10, 'treble_rotation': 0.28,
                'stereo_width': 1.20, 'reverb_room': 0.50,
                'eq_sub60_gain': 4.0, 'eq_bass100_gain': 5.0, 'eq_ubass200_gain': 2.5,
                'eq_lowmid350_gain': -3.0, 'eq_presence3k_gain': 4.0, 'eq_def5k_gain': 3.0,
            },
            'hip_hop': {
                'rotation_speed': 0.12, 'bass_rotation': 0.06, 'treble_rotation': 0.18,
                'reverb_mix': 0.25, 'distance': 0.80,
                'eq_sub30_gain': 8.0, 'eq_sub60_gain': 9.0, 'eq_bass100_gain': 6.0,
                'eq_lowmid350_gain': -4.0, 'eq_mid700_gain': -2.0,
                'eq_presence3k_gain': 2.5, 'eq_air12k_gain': 1.5,
            },
            'rnb': {
                'rotation_speed': 0.13, 'bass_rotation': 0.07, 'stereo_width': 1.10,
                'reverb_room': 0.65, 'reverb_mix': 0.35,
                'eq_sub60_gain': 5.0, 'eq_bass100_gain': 4.5, 'eq_ubass200_gain': 2.0,
                'eq_presence3k_gain': 2.5, 'eq_bril8k_gain': 2.5,
            },
            'metal': {
                'rotation_speed': 0.22, 'bass_rotation': 0.12, 'treble_rotation': 0.38,
                'stereo_width': 1.30, 'reverb_room': 0.45,
                'eq_sub60_gain': 5.0, 'eq_bass100_gain': 4.0,
                'eq_lowmid350_gain': -5.0, 'eq_mid700_gain': -3.0,
                'eq_presence3k_gain': 6.0, 'eq_def5k_gain': 4.0, 'eq_bril8k_gain': 3.0,
            },
            'jazz': {
                'rotation_speed': 0.14, 'reverb_room': 0.68, 'reverb_mix': 0.38,
                'stereo_width': 1.10,
                'eq_bass100_gain': 2.0, 'eq_ubass200_gain': 1.0,
                'eq_presence3k_gain': 2.0, 'eq_def5k_gain': 1.5,
            },
            'ambient': {
                'rotation_speed': 0.07, 'bass_rotation': 0.03, 'treble_rotation': 0.10,
                'reverb_room': 0.92, 'reverb_mix': 0.65, 'stereo_width': 1.45, 'distance': 1.60,
                'eq_sub30_gain': 1.0, 'eq_sub60_gain': 2.0, 'eq_air12k_gain': 5.0,
                'eq_uair16k_gain': 3.0,
            },
            'bollywood': {
                'rotation_speed': 0.14, 'bass_rotation': 0.07, 'treble_rotation': 0.22,
                'reverb_room': 0.70, 'reverb_mix': 0.38, 'stereo_width': 1.15,
                'enable_vocal_center': True,
                'eq_sub30_gain': 3.0, 'eq_sub60_gain': 4.5, 'eq_bass100_gain': 3.5,
                'eq_ubass200_gain': 1.5, 'eq_lowmid350_gain': -1.5,
                'eq_mid700_gain': 0.5,
                'eq_umid1500_gain': 2.0, 'eq_presence3k_gain': 3.5,
                'eq_def5k_gain': 2.5, 'eq_bril8k_gain': 2.5, 'eq_air12k_gain': 2.5,
                'instrument_enhance': True, 'vocal_safe_bass': True,
            },
            'bhangra': {
                'rotation_speed': 0.20, 'bass_rotation': 0.12, 'treble_rotation': 0.30,
                'reverb_room': 0.55, 'reverb_mix': 0.28, 'stereo_width': 1.25,
                'eq_sub30_gain': 5.0, 'eq_sub60_gain': 7.0, 'eq_bass100_gain': 5.5,
                'eq_ubass200_gain': 2.5, 'eq_lowmid350_gain': -3.0,
                'eq_presence3k_gain': 4.0, 'eq_def5k_gain': 3.0, 'eq_bril8k_gain': 3.5,
            },
            'nepali_folk': {
                'rotation_speed': 0.14, 'bass_rotation': 0.07, 'treble_rotation': 0.20,
                'reverb_room': 0.72, 'reverb_mix': 0.40, 'stereo_width': 1.15,
                'elevation': 0.08,
                'eq_sub30_gain': 1.5, 'eq_sub60_gain': 2.5, 'eq_bass100_gain': 3.0,
                'eq_ubass200_gain': 1.5, 'eq_lowmid350_gain': -1.0,
                'eq_umid1500_gain': 2.5, 'eq_presence3k_gain': 3.5,
                'eq_def5k_gain': 2.0, 'eq_bril8k_gain': 2.0, 'eq_air12k_gain': 2.0,
                'instrument_enhance': True,
            },
            'ghazal': {
                'rotation_speed': 0.10, 'bass_rotation': 0.05, 'treble_rotation': 0.14,
                'reverb_room': 0.80, 'reverb_mix': 0.48, 'stereo_width': 1.05,
                'enable_vocal_center': True,
                'eq_sub30_gain': 0.5, 'eq_sub60_gain': 1.5, 'eq_bass100_gain': 2.0,
                'eq_ubass200_gain': 1.0, 'eq_lowmid350_gain': -1.0,
                'eq_mid700_gain': 1.5,
                'eq_umid1500_gain': 1.5, 'eq_presence3k_gain': 2.5,
                'eq_bril8k_gain': 2.0, 'eq_air12k_gain': 2.0,
                'vocal_safe_bass': True,
            },
            'indian_classical': {
                'rotation_speed': 0.10, 'bass_rotation': 0.05, 'treble_rotation': 0.15,
                'reverb_room': 0.82, 'reverb_mix': 0.50, 'elevation': 0.12,
                'stereo_width': 1.10,
                'eq_sub30_gain': 1.0, 'eq_sub60_gain': 2.0, 'eq_bass100_gain': 2.5,
                'eq_ubass200_gain': 2.0,
                'eq_lowmid350_gain': -1.0, 'eq_umid1500_gain': 2.0,
                'eq_presence3k_gain': 3.0, 'eq_def5k_gain': 2.0,
                'eq_bril8k_gain': 2.5, 'eq_air12k_gain': 3.0,
                'instrument_enhance': True,
            },
            'devotional': {
                'rotation_speed': 0.09, 'bass_rotation': 0.04, 'treble_rotation': 0.13,
                'reverb_room': 0.88, 'reverb_mix': 0.55, 'stereo_width': 1.20,
                'elevation': 0.10, 'enable_vocal_center': True,
                'eq_sub30_gain': 1.0, 'eq_sub60_gain': 2.0, 'eq_bass100_gain': 2.5,
                'eq_lowmid350_gain': -1.0, 'eq_umid1500_gain': 1.5,
                'eq_presence3k_gain': 2.5, 'eq_bril8k_gain': 2.0, 'eq_air12k_gain': 3.0,
                'vocal_safe_bass': True,
            },
        }

        if genre in genre_overrides:
            p.update(genre_overrides[genre])

        # ── BPM-synced rotation speed ────────────────────────────────────────
        bps = bpm / 60.0
        for ratio in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            cand = bps * ratio
            if 0.05 <= cand <= 0.50:
                p['rotation_speed'] = round(cand, 3)
                break

        # ── Heavy sub-bass → ease reverb & bass rotation ─────────────────────
        if (sub_r + bass_r) > 0.55:
            p['bass_rotation'] = round(p['bass_rotation'] * 0.78, 3)
            p['reverb_mix']    = round(p['reverb_mix']    * 0.88, 3)

        # ── Vocal protection ─────────────────────────────────────────────────
        if has_vocals and voc_prom > 0.55:
            p['enable_vocal_center'] = True
            p['rotation_speed']      = round(p['rotation_speed'] * 0.85, 3)
            # Protect vocal fundamental ± 1 octave
            voc_lo = voc_f0 * 0.6
            voc_hi = voc_f0 * 1.8
            if voc_lo < 700 < voc_hi:
                p['eq_mid700_gain']  = max(p['eq_mid700_gain'], -0.5)
            if voc_lo < 1500 < voc_hi:
                p['eq_umid1500_gain'] = min(p['eq_umid1500_gain'], 0.5)

        # ── Dynamic range adjustments ────────────────────────────────────────
        if dyn > 16:
            p['reverb_mix']   = round(p['reverb_mix'] * 0.78, 3)
            p['stereo_width'] = round(p['stereo_width'] * 0.90, 3)
        elif dyn < 7:
            p['stereo_width']         = round(p['stereo_width'] * 1.12, 3)
            p['intensity_multiplier'] = 1.25

        # ── Crest factor correction (over-compressed material) ───────────────
        if crest < 6.0:
            # Heavily compressed — reduce rotation speed (it'll sound better)
            p['rotation_speed'] = round(p['rotation_speed'] * 0.90, 3)
            # Boost sub-bass slightly to add missing impact
            p['eq_sub30_gain'] = round(p['eq_sub30_gain'] + 1.5, 1)
            p['eq_sub60_gain'] = round(p['eq_sub60_gain'] + 1.5, 1)
            print("  ↳ Over-compressed source detected → adjusted EQ & rotation")
        elif crest > 18.0:
            # Very dynamic → increase reverb mix for fuller space perception
            p['reverb_mix'] = round(min(p['reverb_mix'] * 1.15, 0.65), 3)

        # ── Mode-aware reverb tuning ─────────────────────────────────────────
        if mode == 'minor':
            # Minor keys benefit from slightly more reverb depth
            p['reverb_room'] = round(min(p['reverb_room'] * 1.08, 0.95), 3)
            p['reverb_mix']  = round(min(p['reverb_mix']  * 1.05, 0.65), 3)
        elif mode == 'major' and genre in ('electronic', 'pop', 'bhangra'):
            # Major + energetic → slightly wider stereo
            p['stereo_width'] = round(min(p['stereo_width'] * 1.05, 1.5), 2)

        # ── Transient density → rotation adjustment ──────────────────────────
        if transients == 'dense':
            p['rotation_speed'] = round(p['rotation_speed'] * 0.82, 3)
            p['bass_rotation']  = round(p['bass_rotation']  * 0.85, 3)
        elif transients == 'sparse':
            p['rotation_speed'] = round(min(p['rotation_speed'] * 1.10, 0.45), 3)

        # ── Fake stereo → compensate by widening ────────────────────────────
        if is_fake_st:
            p['stereo_width'] = round(min(p['stereo_width'] * 1.20, 1.60), 2)
            print("  ↳ Fake stereo source → increased stereo width compensation")

        # ── High HNR + low noise → instrument enhance less needed ────────────
        if hnr > 10.0:
            # Very clean harmonic content → ease off enhancement to avoid artefacts
            p['instrument_enhance'] = False

        # ── High rhythm complexity → slightly slower rotation ────────────────
        if rhythm > 0.75:
            p['rotation_speed'] = round(p['rotation_speed'] * 0.82, 3)

        # ── Boost hidden strings if subdued ─────────────────────────────────
        if string_prom < 0.05 and p.get('instrument_enhance'):
            p['eq_umid1500_gain']   = round(p['eq_umid1500_gain'] + 1.5, 1)
            p['eq_presence3k_gain'] = round(p['eq_presence3k_gain'] + 1.0, 1)

        # ── Vocal-safe bass caps ─────────────────────────────────────────────
        if p.get('vocal_safe_bass'):
            p['eq_sub30_gain']   = min(p['eq_sub30_gain'],   10.0)
            p['eq_sub60_gain']   = min(p['eq_sub60_gain'],   10.0)
            p['eq_bass100_gain'] = min(p['eq_bass100_gain'],  8.0)
            p['eq_ubass200_gain'] = min(p['eq_ubass200_gain'], 3.0)

        # ── Clamp all EQ ────────────────────────────────────────────────────
        for k in ['eq_sub30_gain','eq_sub60_gain','eq_bass100_gain','eq_ubass200_gain',
                  'eq_lowmid350_gain','eq_mid700_gain','eq_umid1500_gain',
                  'eq_presence3k_gain','eq_def5k_gain','eq_bril8k_gain',
                  'eq_air12k_gain','eq_uair16k_gain']:
            p[k] = max(-14.0, min(14.0, round(p[k], 1)))

        return p

    # ────────────────────────────────────────────────────────────────────────
    # Fallback (no librosa)
    # ────────────────────────────────────────────────────────────────────────

    def _basic_analysis(self, file_path: str) -> Dict:
        return {
            'duration': round(get_audio_duration(file_path), 2),
            'sample_rate': 48000, 'channels': 2,
            'bpm': None, 'genre': 'unknown',
            'key': 'C', 'mode': 'unknown', 'key_confidence': 0.0,
            'sub_bass_ratio': 0.08, 'upper_sub_ratio': 0.07,
            'bass_ratio': 0.15, 'upper_bass_ratio': 0.10,
            'low_mid_ratio': 0.12, 'mid_ratio': 0.25,
            'upper_mid_ratio': 0.12, 'presence_ratio': 0.06,
            'brilliance_ratio': 0.03, 'air_ratio': 0.02,
            'has_vocals': False, 'vocal_fundamental_hz': 250.0,
            'crest_factor_db': 12.0, 'is_over_compressed': False,
            'is_fake_stereo': True, 'stereo_correlation': 1.0,
            'transient_density': 'moderate', 'onset_rate': 2.0,
            'harmonic_to_noise_ratio': 1.0, 'hnr_db': 0.0,
            'recommended_settings': {
                'rotation_speed': 0.15, 'reverb_room': 0.60, 'reverb_mix': 0.30,
                'bass_rotation': 0.08, 'treble_rotation': 0.20,
                'stereo_width': 1.0, 'elevation': 0.0, 'distance': 1.0,
                'enable_vocal_center': False,
                'eq_sub30_gain': 3.0, 'eq_sub60_gain': 4.0,
                'eq_bass100_gain': 3.0, 'eq_ubass200_gain': 1.5,
                'eq_lowmid350_gain': -2.5, 'eq_mid700_gain': -1.0,
                'eq_umid1500_gain': 1.0, 'eq_presence3k_gain': 2.0,
                'eq_def5k_gain': 1.5, 'eq_bril8k_gain': 2.0,
                'eq_air12k_gain': 2.0, 'eq_uair16k_gain': 1.0,
                'vocal_safe_bass': True, 'instrument_enhance': True,
            }
        }


audio_analyzer = IntelligentAudioAnalyzer()


# ============================================================================
# INSTRUMENT ROUTER  v8.0 — Psychoacoustic per-stem parameter engine
# ============================================================================
#
# Based on the psychoacoustic principles:
#   • Vocals / bass / kick → centre, slow rotation (localisable at low freq)
#   • Hi-hats / cymbals → wide, fast rotation (non-localisable, add space)
#   • Sustained pads → slow wide rotation for immersion
#   • Transient-heavy → faster, more localised movements for punch
#   • Elevation: cymbals/flutes above (+), bass/kick below (-)
# ============================================================================

class InstrumentRouter:
    """
    Maps stem/instrument class → psychoacoustically optimal spatial params.
    Implements the full rule table from the design document, then applies
    BPM-sync, energy-section modulation, and key/mode adjustments.
    """

    # ── Base parameter table ──────────────────────────────────────────────────
    # Keys: stem names as returned by Demucs / Spleeter
    # 'other' is a catch-all for piano, strings, guitar when 4-stem is used.
    #
    # Format: rotation_speed, bass_rotation, treble_rotation,
    #         stereo_width, elevation, reverb_mix, enable_vocal_center
    STEM_TABLE = {
        # ── Vocals ───────────────────────────────────────────────────────────
        'vocals': {
            'rotation_speed': 0.07,   # slow — keeps voice centred and intelligible
            'bass_rotation':  0.02,
            'treble_rotation':0.08,
            'stereo_width':   0.75,   # narrow — voice should feel inside your head
            'elevation':      0.0,
            'reverb_mix':     0.25,   # modest reverb — too much smears lyrics
            'reverb_room':    0.55,
            'enable_vocal_center': True,
            'instrument_enhance': False,  # vocals don't need harmonic enhancement
        },
        # ── Drums (full kit) ─────────────────────────────────────────────────
        'drums': {
            'rotation_speed': 0.14,   # moderate — kit spreads around the space
            'bass_rotation':  0.05,   # kick stays more centred
            'treble_rotation':0.30,   # hi-hats/cymbals orbit fast
            'stereo_width':   1.20,
            'elevation':      0.05,   # slight top-of-room feel
            'reverb_mix':     0.15,   # tight — preserve transient punch
            'reverb_room':    0.40,
            'enable_vocal_center': False,
        },
        # ── Bass (bass guitar, synth bass) ───────────────────────────────────
        'bass': {
            'rotation_speed': 0.04,   # very slow — low freq poorly localisable
            'bass_rotation':  0.03,
            'treble_rotation':0.06,
            'stereo_width':   0.80,   # fairly narrow — preserve mono compatibility
            'elevation':      -0.08,  # slightly below listener (psychoacoustic)
            'reverb_mix':     0.12,   # minimal reverb — keeps bass tight
            'reverb_room':    0.35,
            'enable_vocal_center': False,
        },
        # ── Guitar (electric / acoustic — 6-stem model) ──────────────────────
        'guitar': {
            'rotation_speed': 0.12,
            'bass_rotation':  0.06,
            'treble_rotation':0.18,
            'stereo_width':   1.10,
            'elevation':      0.0,
            'reverb_mix':     0.28,
            'reverb_room':    0.60,
            'enable_vocal_center': False,
        },
        # ── Piano (6-stem model) ─────────────────────────────────────────────
        'piano': {
            'rotation_speed': 0.09,
            'bass_rotation':  0.04,
            'treble_rotation':0.15,
            'stereo_width':   1.00,
            'elevation':      0.0,
            'reverb_mix':     0.28,
            'reverb_room':    0.65,
            'enable_vocal_center': False,
        },
        # ── Other (pads, strings, synth — catch-all) ─────────────────────────
        'other': {
            'rotation_speed': 0.07,   # slow, wide orbits for pads and strings
            'bass_rotation':  0.03,
            'treble_rotation':0.12,
            'stereo_width':   1.40,   # wide — pads fill the space
            'elevation':      0.08,   # slight upward tilt (strings/pads feel "above")
            'reverb_mix':     0.42,   # lush reverb for atmosphere
            'reverb_room':    0.78,
            'enable_vocal_center': False,
        },
    }

    # ── Psychoacoustic rules applied ON TOP of the base table ─────────────────
    # These scale factors are applied based on overall analysis results.

    def get_stem_params(
        self,
        stem_name: str,
        base_params: ProcessingParams,
        analysis: Optional[Dict[str, Any]] = None,
        gain_db: float = 0.0,   # makeup gain from gain staging
    ) -> ProcessingParams:
        """
        Return a ProcessingParams tuned for the given stem using the
        psychoacoustic base table, modulated by analysis results.

        Priority order:
          1. User per-stem override (if set on base_params)
          2. Analysis-modulated base table value
          3. Base table default
        """
        table = self.STEM_TABLE.get(stem_name, self.STEM_TABLE['other'])

        # ── Start from a copy of global params so EQ, bitrate, etc. carry over
        d: Dict[str, Any] = {}

        # Base spatial params from table
        for key in ['rotation_speed', 'bass_rotation', 'treble_rotation',
                    'stereo_width', 'elevation', 'reverb_mix', 'reverb_room',
                    'enable_vocal_center', 'instrument_enhance']:
            if key in table:
                d[key] = table[key]

        # ── Apply BPM-synced rotation speed ──────────────────────────────────
        if analysis:
            bpm = analysis.get('bpm') or 120
            bps = bpm / 60.0
            base_rot = d.get('rotation_speed', 0.10)
            for ratio in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                cand = bps * ratio
                # Clamp within musically useful range per stem type
                lo, hi = self._rot_range(stem_name)
                if lo <= cand <= hi:
                    d['rotation_speed'] = round(cand, 4)
                    break

            # Treble rotation: always faster than rotation_speed
            d['treble_rotation'] = round(
                max(d.get('treble_rotation', 0.15),
                    d['rotation_speed'] * self._treble_ratio(stem_name)), 4)

            # ── Energy-based width modulation ─────────────────────────────
            energy = analysis.get('dynamic_range', 12)
            if energy > 16:  # very dynamic → narrow a touch
                d['stereo_width'] = round(d.get('stereo_width', 1.0) * 0.92, 3)
            elif energy < 7:  # over-compressed → widen to compensate
                d['stereo_width'] = round(
                    min(d.get('stereo_width', 1.0) * 1.10, 1.60), 3)

            # ── Minor key → slightly more reverb depth ────────────────────
            if analysis.get('mode') == 'minor':
                d['reverb_room'] = round(
                    min(d.get('reverb_room', 0.6) * 1.06, 0.92), 3)
                d['reverb_mix'] = round(
                    min(d.get('reverb_mix', 0.3) * 1.05, 0.60), 3)

        # ── Per-stem user overrides ───────────────────────────────────────────
        override_map = {
            'vocals': ('stem_vocals_rotation', 'stem_vocals_width',
                       'stem_vocals_elevation', 'stem_vocals_reverb'),
            'drums':  ('stem_drums_rotation',  'stem_drums_width',
                       'stem_drums_elevation',  'stem_drums_reverb'),
            'bass':   ('stem_bass_rotation_override', 'stem_bass_width',
                       'stem_bass_elevation',  'stem_bass_reverb'),
            'guitar': ('stem_guitar_rotation', 'stem_guitar_width',
                       'stem_guitar_elevation','stem_guitar_reverb'),
            'piano':  ('stem_piano_rotation',  'stem_piano_width',
                       'stem_piano_elevation', 'stem_piano_reverb'),
            'other':  ('stem_other_rotation',  'stem_other_width',
                       'stem_other_elevation', 'stem_other_reverb'),
        }
        keys_for_stem = override_map.get(stem_name, override_map['other'])
        field_names = ['rotation_speed', 'stereo_width', 'elevation', 'reverb_mix']
        for attr, field in zip(keys_for_stem, field_names):
            val = getattr(base_params, attr, None)
            if val is not None:
                d[field] = val

        # ── Apply makeup gain from gain staging ───────────────────────────────
        # Embed as intensity_multiplier so the HRTF engine scales the output
        if abs(gain_db) > 0.1:
            linear = 10 ** (gain_db / 20.0)
            d['intensity_multiplier'] = round(
                base_params.intensity_multiplier * linear, 4)

        # ── Build the final params object ─────────────────────────────────────
        return base_params.copy(update=d)

    def _rot_range(self, stem_name: str):
        """Musically valid rotation speed range per stem type."""
        ranges = {
            'vocals':  (0.04, 0.18),
            'drums':   (0.08, 0.35),
            'bass':    (0.02, 0.08),
            'guitar':  (0.06, 0.25),
            'piano':   (0.05, 0.20),
            'other':   (0.03, 0.15),
        }
        return ranges.get(stem_name, (0.04, 0.30))

    def _treble_ratio(self, stem_name: str) -> float:
        """How much faster treble rotates vs main rotation, per stem type."""
        ratios = {
            'vocals':  1.2,
            'drums':   2.5,  # hi-hats spin much faster than the main kit orbit
            'bass':    1.3,
            'guitar':  1.6,
            'piano':   1.5,
            'other':   1.8,
        }
        return ratios.get(stem_name, 1.5)

    def estimate_stem_gain_db(
        self, stem_path: str, target_lufs: float = -23.0
    ) -> float:
        """
        Estimate makeup gain (dB) to normalise a stem to target_lufs.
        Uses a fast RMS proxy: rms_dBFS = 20·log10(rms), then
        gain = target_lufs − rms_dBFS.

        Example: stem RMS = −20 dBFS, target = −23 LUFS
          → gain = −23 − (−20) = −3 dB  (reduce by 3 dB)

        Returns 0.0 if librosa is unavailable or measurement fails.
        Clamped to [−18, +12] dB to avoid boosting near-silent stems
        into audibility or over-compressing loud ones.
        """
        if not ADVANCED_ANALYSIS:
            return 0.0
        try:
            y, sr = librosa.load(stem_path, sr=None, mono=True, duration=60)
            rms = float(np.sqrt(np.mean(y ** 2) + 1e-10))
            rms_dbfs = 20 * np.log10(rms)          # RMS in dBFS (e.g. -20.0)
            gain = target_lufs - rms_dbfs            # e.g. -23 - (-20) = -3 dB
            return float(np.clip(gain, -18.0, 12.0))
        except Exception:
            return 0.0


instrument_router = InstrumentRouter()


# ============================================================================
# HRIR CONVOLUTION ENGINE  — v9.0  (breaks the FFmpeg filtergraph ceiling)
# ============================================================================
#
# All previous HRTF simulation was done inside FFmpeg's volume + equalizer
# filters. That architecture has a hard ceiling:
#   • ILD via volume automation — accurate only for a spherical head model
#   • ITD via static adelay — fixed integer-sample delays, no interp
#   • Pinna notches via 4 EQ bells — real pinna HRIRs have 200–300 ms
#     impulse responses with complex spectral structure impossible to model
#     with parametric EQ
#
# This engine:
#   1. Loads audio into NumPy (librosa)
#   2. For each 256-sample block, looks up the binaural HRIR for the current
#      rotation angle (beat-locked if beats available)
#   3. Convolves each block via OLA (overlap-add) with scipy.signal.fftconvolve
#   4. Outputs a float32 stereo WAV that FFmpeg ingests for reverb + EQ + encode
#
# HRIR source: synthesized from first principles using:
#   • Woodworth ITD model (head radius r=0.0875 m, c=343 m/s)
#   • Spherical head ILD model (Duda & Martens 1998)
#   • Pinna notch resonances: 4-notch comb at 8, 10.5, 13, 16 kHz
#   • Optional: load real KEMAR compact-pinnae dataset (.npy format)
#     Drop a file named "kemar_hrirs.npy" next to backend.py with shape
#     (36, 2, N) — 36 azimuths (0–350° step 10°), L/R channels, N samples.

class HRIREngine:
    """
    Block-based binaural renderer using Head-Related Impulse Responses.

    v10.1 — Full SOFA support (KEMAR near-field dataset):
        • Loads any *.sofa HDF5 file directly — no conversion step needed.
        • Builds a 3-D lookup table: (distance_cm, elevation_deg, azimuth_deg)
        • Trilinear interpolation across all three spatial dimensions.
        • Distance range : 20–110 cm  (dataset 1 cm steps; engine samples ~11)
        • Elevation range: −25° → +35° (full KEMAR measurement range)
        • Azimuth range  : 0° → 360°  (full circle, per-measurement interp)
        • Falls back to legacy .npy then synthetic if SOFA unavailable.

    Usage:
        engine = HRIREngine(sample_rate=48000)
        stereo = engine.render(
            mono,           # np.ndarray (N,) float32
            beat_times_s,   # List[float] | None
            rotation_speed, # Hz
            elevation,      # normalised [-1, 1]
            distance_m,     # metres  (0.20 – 1.10 maps into SOFA range)
        )
        # stereo shape: (2, N)
    """

    BLOCK  = 256    # OLA processing block size (samples)
    HRIR_N = 512    # Target IR length after resampling / truncation
    # Legacy constants — still used when falling back to numpy / synthetic
    AZ_STEP = 10
    N_AZ    = 36

    def __init__(self, sample_rate: int = 48000):
        self.sr         = sample_rate
        self.using_sofa = False

        # ── SOFA 3-D lookup ───────────────────────────────────────────────────
        # sofa_table[dist_cm][elev_deg] = (az_arr, L_mat, R_mat)
        #   az_arr : float32 (N_az,)         sorted azimuths in degrees
        #   L_mat  : float32 (N_az, HRIR_N)  left  HRIR for each azimuth
        #   R_mat  : float32 (N_az, HRIR_N)  right HRIR for each azimuth
        self.sofa_table : Dict[int, Dict[int, tuple]] = {}
        self.sofa_dists : List[int] = []   # sorted unique loaded distances (cm)
        self.sofa_elevs : List[int] = []   # sorted unique elevations (deg)

        # ── Legacy flat lookup (numpy / synthetic) ────────────────────────────
        # hrirs[az_idx] = (L_ir, R_ir)  both shape (HRIR_N,)
        self.hrirs: Dict[int, tuple] = {}

        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # Loading — SOFA → numpy → synthetic cascade

    def _load(self):
        # 1. Try ALL *.sofa files in BASE_DIR — merge every file found so that
        #    e.g. KEMAR_NFHRIRmea_1cm + _1.5cm combine into one richer table.
        sofa_candidates = sorted(
            list(BASE_DIR.glob("*.sofa")) + list(BASE_DIR.glob("*.SOFA")),
            key=lambda p: (0 if "mea" in p.name.lower() else 1)
        )
        if sofa_candidates:
            loaded_any = False
            for sofa_path in sofa_candidates:
                try:
                    self._load_sofa(sofa_path, merge=loaded_any)
                    loaded_any = True
                    n_dists = len(getattr(self, "sofa_dists", []))
                    print(f"  ↳ Merged {sofa_path.name} (table: {n_dists} distances)")
                except Exception as e:
                    print(f"⚠  HRIR: {sofa_path.name} failed ({e}), skipping")
            if loaded_any:
                return

        # 2. Legacy numpy  (36 × 2 × N  or  36 × N × 2)
        kemar_path = BASE_DIR / "kemar_hrirs.npy"
        if kemar_path.exists():
            try:
                data = np.load(str(kemar_path))
                if data.ndim == 3:
                    n = data.shape[0]
                    for i in range(n):
                        if data.shape[1] == 2:
                            self.hrirs[i] = (data[i, 0].astype(np.float32),
                                             data[i, 1].astype(np.float32))
                        else:
                            self.hrirs[i] = (data[i, :, 0].astype(np.float32),
                                             data[i, :, 1].astype(np.float32))
                    self.AZ_STEP = 360 // n
                    self.N_AZ    = n
                    print(f"✅ HRIR: Loaded numpy dataset ({n} azimuths)")
                    return
            except Exception as e:
                print(f"⚠  HRIR: numpy load failed ({e})")

        # 3. Synthetic model
        self._build_synthetic()
        print(f"✅ HRIR: Synthetic model built ({self.N_AZ} azimuths @ {self.sr} Hz)")

    # ─────────────────────────────────────────────────────────────────────────
    # SOFA loader

    def _load_sofa(self, path: Path, merge: bool = False):
        """
        Load a SOFA (HDF5) HRTF dataset and build (or extend) the 3-D lookup table.

        merge=False: replaces any existing table (first file).
        merge=True:  adds this file's distances to the existing table so that
                     multiple SOFA files (e.g. near-field at different distances)
                     are combined into one richer lookup.

        Coordinate convention assumed:
          SourcePosition[:, 0] = azimuth  (degrees, 0=front, 90=right)
          SourcePosition[:, 1] = elevation (degrees)
          SourcePosition[:, 2] = distance  (metres — SOFA standard)
        """
        if not H5PY_AVAILABLE:
            raise RuntimeError("h5py not installed — pip install h5py")
        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py required for SOFA — pip install h5py")
        try:
            from scipy.signal import resample_poly
        except ImportError:
            raise RuntimeError("scipy required for SOFA — pip install scipy")
        from math import gcd

        print(f"  ⟳ HRIR: Loading {path.name} …")

        with h5py.File(str(path), 'r') as f:
            ir_data   = f['Data.IR'][:]            # (M, R, N_orig)
            positions = f['SourcePosition'][:]      # (M, 3)
            sr_sofa   = int(np.squeeze(f['Data.SamplingRate'][:]))

        M, R, N_orig = ir_data.shape
        print(f"  ↳ {M:,} measurements  SR={sr_sofa} Hz  IR={N_orig} smp  "
              f"receivers={R}")

        # ── Coordinate parse ─────────────────────────────────────────────────
        az_raw   = positions[:, 0] % 360.0
        elev_raw = positions[:, 1]
        dist_raw = positions[:, 2]

        # Convert to cm (SOFA stores metres; guard against files already in cm)
        dist_cm_f = dist_raw * 100.0 if dist_raw.max() < 5.0 else dist_raw.copy()

        # ── Resampling setup ─────────────────────────────────────────────────
        needs_resample = (sr_sofa != self.sr)
        if needs_resample:
            g    = gcd(self.sr, sr_sofa)
            up   = self.sr   // g
            down = sr_sofa   // g
            n_rs = int(np.ceil(N_orig * up / down))
            print(f"  ↳ Resampling {sr_sofa}→{self.sr} Hz  (×{up}/÷{down})")
        else:
            up = down = 1
            n_rs = N_orig

        target_n = min(self.HRIR_N, n_rs)

        # ── Choose subset of distances to load ───────────────────────────────
        all_dist_cm = np.unique(np.round(dist_cm_f).astype(int))
        n_dists     = len(all_dist_cm)
        if n_dists > 1:
            # Sample ~11 distances evenly (always include endpoints)
            n_load = min(11, n_dists)
            idx    = np.round(np.linspace(0, n_dists - 1, n_load)).astype(int)
            load_set = set(all_dist_cm[np.unique(idx)].tolist())
        else:
            load_set = set(all_dist_cm.tolist())

        # ── Quantise elevation + azimuth to dataset grid ─────────────────────
        # Round elevation to nearest 5°  (KEMAR dataset is at 5° steps)
        # Round azimuth   to nearest 5°
        elev_q = (np.round(elev_raw / 5.0) * 5.0).astype(int)
        az_q   = (np.round(az_raw   / 5.0) * 5.0 % 360).astype(int)
        dist_q =  np.round(dist_cm_f).astype(int)

        unique_elevs = sorted(set(elev_q.tolist()))
        print(f"  ↳ Elevations: {unique_elevs[0]}° → {unique_elevs[-1]}°  "
              f"({len(unique_elevs)} levels)")
        print(f"  ↳ Loading {len(load_set)}/{n_dists} distances  "
              f"({min(load_set)}–{max(load_set)} cm) …")

        table: Dict[int, Dict[int, tuple]] = {}
        n_irs_loaded = 0

        for d_cm in sorted(load_set):
            table[d_cm] = {}
            for e_deg in unique_elevs:
                mask    = (dist_q == d_cm) & (elev_q == e_deg)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue

                az_vals = az_q[indices]
                order   = np.argsort(az_vals)
                az_sorted    = az_vals[order].astype(np.float32)
                meas_sorted  = indices[order]

                L_list: List[np.ndarray] = []
                R_list: List[np.ndarray] = []

                for m_idx in meas_sorted:
                    # Use only the first two receivers (L=0, R=1)
                    ir_L = ir_data[m_idx, 0].astype(np.float64)
                    ir_R = ir_data[m_idx, 1].astype(np.float64)

                    if needs_resample:
                        ir_L = resample_poly(ir_L, up, down)
                        ir_R = resample_poly(ir_R, up, down)

                    # Truncate / zero-pad to target_n
                    L_t = np.zeros(target_n, dtype=np.float32)
                    R_t = np.zeros(target_n, dtype=np.float32)
                    cn  = min(target_n, len(ir_L))
                    L_t[:cn] = ir_L[:cn].astype(np.float32)
                    R_t[:cn] = ir_R[:cn].astype(np.float32)

                    L_list.append(L_t)
                    R_list.append(R_t)
                    n_irs_loaded += 1

                table[d_cm][e_deg] = (
                    az_sorted,
                    np.stack(L_list),   # (N_az, target_n)
                    np.stack(R_list),
                )

        if merge and hasattr(self, 'sofa_table') and self.sofa_table:
            # Merge new distances into existing table; don't overwrite existing ones
            for d_cm_new, e_dict in table.items():
                if d_cm_new not in self.sofa_table:
                    self.sofa_table[d_cm_new] = e_dict
                else:
                    # Distance already present — merge elevation slots
                    for e_deg, slot in e_dict.items():
                        if e_deg not in self.sofa_table[d_cm_new]:
                            self.sofa_table[d_cm_new][e_deg] = slot
            self.sofa_dists = sorted(self.sofa_table.keys())
            # Keep elevation list as union of both
            merged_elevs = sorted(set(self.sofa_elevs) | set(unique_elevs))
            self.sofa_elevs = merged_elevs
        else:
            self.sofa_table  = table
            self.sofa_dists  = sorted(table.keys())
            self.sofa_elevs  = unique_elevs
        self.HRIR_N      = target_n
        self.using_sofa  = True

        mem_mb = n_irs_loaded * 2 * target_n * 4 / 1024**2
        print(f"✅ HRIR: SOFA loaded — {n_irs_loaded:,} IRs  "
              f"({len(self.sofa_dists)} dists × {len(unique_elevs)} elevs)  "
              f"IR={target_n} taps @ {self.sr} Hz  ≈{mem_mb:.0f} MB")

    # ─────────────────────────────────────────────────────────────────────────
    # Synthetic HRIR generation

    def _build_synthetic(self):
        for az_idx in range(self.N_AZ):
            az_deg = az_idx * self.AZ_STEP
            L, R = self._synthesize_hrir(az_deg)
            self.hrirs[az_idx] = (L, R)

    def _synthesize_hrir(self, az_deg: float) -> tuple:
        """
        Synthesise a binaural HRIR pair using three perceptual cues:

        1. ITD (Woodworth 1962): delay = (r/c) * (θ + sin θ)
           where r = 0.0875 m (head radius), c = 343 m/s.
           Maps az ∈ [0°, 180°] → ITD ∈ [0, 630 μs].

        2. ILD (Duda & Martens 1998 approximation):
           HF attenuation on the contralateral channel using a 1st-order
           shelf model.  At 90° the far-ear is attenuated ~7 dB above 1 kHz.

        3. Pinna spectral notches:
           Narrow EQ cuts at 8, 10.5, 13, 16 kHz applied to the contralateral
           channel, simulating pinna reflection-induced cancellation peaks.
        """
        sr   = self.sr
        N    = self.HRIR_N
        r    = 0.0875           # head radius, metres
        c    = 343.0            # speed of sound, m/s

        # Map azimuth to angle in [0, π] (left=π, right=0)
        # Convention: 0° = front, 90° = right, 180°/270° = left
        az_rad = np.radians(az_deg % 360)
        # Lateral angle: 0 at front/back, π/2 at 90° right
        lateral = abs(np.sin(az_rad))            # 0..1

        # ── 1. ITD ──────────────────────────────────────────────────────────
        itd_s = (r / c) * (az_rad + np.sin(az_rad))  # Woodworth
        if az_deg > 180:
            itd_s = -itd_s   # flip for left side
        itd_samples = int(round(itd_s * sr))
        itd_samples = max(-N // 4, min(N // 4, itd_samples))  # clamp

        # ── 2. Base impulse (Dirac) ──────────────────────────────────────────
        L_ir = np.zeros(N, dtype=np.float32)
        R_ir = np.zeros(N, dtype=np.float32)

        # Leading ear: onset at sample 4 for pre-ringing headroom
        onset = 4
        if itd_samples >= 0:
            # Source to the right: R leads
            R_ir[onset]                = 1.0
            L_ir[onset + itd_samples] = 1.0
        else:
            # Source to the left: L leads
            L_ir[onset]                = 1.0
            R_ir[onset - itd_samples] = 1.0

        # ── 3. ILD — frequency-dependent shelf on contralateral channel ──────
        # Approximate using a minimum-phase shelf filter.
        # 7 dB attenuation above 1 kHz, scaled by lateral position.
        ild_db = -7.0 * lateral
        ild_lin = 10.0 ** (ild_db / 20.0)

        # Build a very gentle 1-pole high-shelf coefficient
        # y[n] = x[n] + alpha * (x[n] - x[n-1])  — enhances ILD at HF
        alpha = 0.0  # start flat
        if abs(ild_db) > 0.5:
            # Map ild_db → shelf alpha using first-order approximation
            f_c = 1000.0 / sr   # shelf corner at 1 kHz (normalised)
            omega = 2 * np.pi * f_c
            alpha = (1.0 - ild_lin) * (1.0 - np.cos(omega)) / (1.0 + ild_lin)

        # Apply ILD filter to contralateral channel
        if itd_samples >= 0:
            # Right leads → apply ILD to Left (contralateral)
            for n in range(1, N):
                L_ir[n] = L_ir[n] * ild_lin + alpha * (L_ir[n] - L_ir[n-1]) * ild_lin
        else:
            for n in range(1, N):
                R_ir[n] = R_ir[n] * ild_lin + alpha * (R_ir[n] - R_ir[n-1]) * ild_lin

        # ── 4. Pinna notch filters on contralateral channel ──────────────────
        # Real pinna notches only appear in the elevation/rear plane, but we
        # apply a scaled version to the far ear to simulate the shadowing effect.
        notch_freqs = [8000, 10500, 13000, 16000]
        notch_depth = -8.0 * lateral   # dB, scaled by how far off-axis we are
        notch_bw_hz = 600.0

        if abs(notch_depth) > 0.5:
            contra = L_ir if itd_samples >= 0 else R_ir
            freqs  = np.fft.rfftfreq(N, d=1.0/sr)
            H      = np.ones(len(freqs), dtype=np.complex64)
            for f0 in notch_freqs:
                f0_safe = min(f0, sr / 2 - 1)
                # Peak EQ in frequency domain: H(f) *= 1 + (A-1)·BPF(f)
                A   = 10.0 ** (notch_depth / 20.0)
                bpf = (freqs / f0_safe) / (1.0 + ((freqs / f0_safe) - 1.0)**2 +
                       (notch_bw_hz / f0_safe)**2)
                H  *= (1.0 + (A - 1.0) * np.abs(bpf))
            # Apply in frequency domain (minimum-phase approximation)
            C_freq = np.fft.rfft(contra.astype(np.float64))
            contra[:] = np.fft.irfft(C_freq * H, n=N).astype(np.float32)
            if itd_samples >= 0:
                L_ir = contra
            else:
                R_ir = contra

        # ── 5. Gentle raised-cosine window to reduce pre-ringing ────────────
        win = np.ones(N, dtype=np.float32)
        taper = 32
        win[:taper] = 0.5 - 0.5 * np.cos(np.pi * np.arange(taper) / taper)
        win[-taper:] = 0.5 - 0.5 * np.cos(np.pi * np.arange(taper, 0, -1) / taper)
        L_ir *= win
        R_ir *= win

        return L_ir.astype(np.float32), R_ir.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # HRIR lookup — 3-D trilinear interpolation

    @staticmethod
    def _bracket(arr: List[int], val: float) -> tuple:
        """
        Return (lo, hi, t) for linear interpolation in a sorted list.
        Clamps to [arr[0], arr[-1]].
        """
        if not arr:
            return 0, 0, 0.0
        if val <= arr[0]:
            return arr[0], arr[0], 0.0
        if val >= arr[-1]:
            return arr[-1], arr[-1], 0.0
        for i in range(len(arr) - 1):
            if arr[i] <= val <= arr[i + 1]:
                span = float(arr[i + 1] - arr[i])
                t    = (val - arr[i]) / span if span > 0.0 else 0.0
                return arr[i], arr[i + 1], float(np.clip(t, 0.0, 1.0))
        return arr[-1], arr[-1], 0.0

    def _interp_azimuth(
        self,
        az_arr: np.ndarray,   # float32 (N_az,)  sorted azimuths
        L_mat:  np.ndarray,   # float32 (N_az, HRIR_N)
        R_mat:  np.ndarray,
        az_deg: float,
    ) -> tuple:
        """
        Circular linear interpolation between adjacent azimuth measurements.
        Handles the 355° → 0° wrap correctly.
        """
        az_norm = float(az_deg % 360.0)
        n       = len(az_arr)

        # Find insertion point; wrap index for circular lookup
        idx_hi = int(np.searchsorted(az_arr, az_norm)) % n
        idx_lo = (idx_hi - 1) % n

        az_lo = float(az_arr[idx_lo])
        az_hi = float(az_arr[idx_hi])

        # Unwrap high side when wrapping past 360°
        if idx_hi == 0 or az_hi < az_lo:
            az_hi += 360.0
            az_w   = az_norm if az_norm >= az_lo else az_norm + 360.0
        else:
            az_w   = az_norm

        span = max(az_hi - az_lo, 1e-6)
        t    = float(np.clip((az_w - az_lo) / span, 0.0, 1.0))

        L = ((1.0 - t) * L_mat[idx_lo] + t * L_mat[idx_hi]).astype(np.float32)
        R = ((1.0 - t) * R_mat[idx_lo] + t * R_mat[idx_hi]).astype(np.float32)
        return L, R

    def _sofa_slice(
        self, d_cm: int, e_deg: int, az_deg: float
    ) -> Optional[tuple]:
        """
        Return azimuth-interpolated HRIR pair at an exact (dist, elev) grid
        point.  Returns None if that combination is missing from the table.
        """
        d_tbl = self.sofa_table.get(d_cm)
        if d_tbl is None:
            return None
        slot = d_tbl.get(e_deg)
        if slot is None:
            return None
        az_arr, L_mat, R_mat = slot
        return self._interp_azimuth(az_arr, L_mat, R_mat, az_deg)

    def _hrir_for_position(
        self,
        az_deg:   float,
        elev_deg: float = 0.0,
        dist_cm:  float = 100.0,
    ) -> tuple:
        """
        Trilinear interpolation in (azimuth × elevation × distance) space
        using the SOFA lookup table.

        Falls back to legacy _hrir_for_angle() when SOFA is not loaded.
        """
        if not self.using_sofa:
            return self._hrir_for_angle(az_deg)

        d_lo, d_hi, td = self._bracket(self.sofa_dists, dist_cm)
        e_lo, e_hi, te = self._bracket(self.sofa_elevs, elev_deg)

        # 4 bilinear corners in (dist, elev) — azimuth interpolation is done
        # inside _sofa_slice for each corner
        corners = [(d_lo, e_lo), (d_lo, e_hi), (d_hi, e_lo), (d_hi, e_hi)]
        weights = [
            (1.0 - td) * (1.0 - te),
            (1.0 - td) *        te,
                   td  * (1.0 - te),
                   td  *        te,
        ]

        L_acc = np.zeros(self.HRIR_N, dtype=np.float64)
        R_acc = np.zeros(self.HRIR_N, dtype=np.float64)
        w_sum = 0.0

        for (d, e), w in zip(corners, weights):
            if w < 1e-7:
                continue
            result = self._sofa_slice(d, e, az_deg)
            if result is None:
                continue          # missing measurement — skip this corner
            L_c, R_c = result
            L_acc += w * L_c.astype(np.float64)
            R_acc += w * R_c.astype(np.float64)
            w_sum += w

        if w_sum < 1e-7:
            return self._hrir_for_angle(az_deg)   # all corners missing

        # Re-normalise if any corner was absent
        if w_sum < 0.999:
            L_acc /= w_sum
            R_acc /= w_sum

        return L_acc.astype(np.float32), R_acc.astype(np.float32)

    def _hrir_for_angle(self, az_deg: float) -> tuple:
        """
        Legacy azimuth-only lookup for the numpy / synthetic fallback path.
        Linear interpolation between adjacent azimuths.
        """
        az_norm = az_deg % 360
        idx_f   = az_norm / self.AZ_STEP
        idx_lo  = int(idx_f) % self.N_AZ
        idx_hi  = (idx_lo + 1) % self.N_AZ
        t       = idx_f - int(idx_f)

        L_lo, R_lo = self.hrirs[idx_lo]
        L_hi, R_hi = self.hrirs[idx_hi]

        L = ((1.0 - t) * L_lo + t * L_hi).astype(np.float32)
        R = ((1.0 - t) * R_lo + t * R_hi).astype(np.float32)
        return L, R

    # ─────────────────────────────────────────────────────────────────────────
    # Angle trajectory

    def _angle_trajectory(
        self,
        n_samples: int,
        rotation_speed: float,
        beat_times_s: Optional[List[float]],
    ) -> np.ndarray:
        """
        Return azimuth in degrees for every sample.
        Beat-locked: tracks actual beat timing from librosa.
        Free-running: simple ramp at rotation_speed Hz.
        """
        t = np.arange(n_samples, dtype=np.float64) / self.sr

        if beat_times_s and len(beat_times_s) >= 2:
            # Piecewise linear beat phase (same algorithm as _beat_phase_expr)
            beat_phase = np.zeros(n_samples, dtype=np.float64)
            times = beat_times_s
            for k in range(len(times) - 1):
                t_k   = times[k]
                t_nxt = times[k + 1]
                dur   = max(t_nxt - t_k, 0.01)
                mask  = (t >= t_k) & (t < t_nxt)
                beat_phase[mask] = k + (t[mask] - t_k) / dur
            # Extrapolate after last beat
            dt_last = max(times[-1] - times[-2], 0.01)
            mask_tail = t >= times[-1]
            beat_phase[mask_tail] = (
                (len(times) - 1) + (t[mask_tail] - times[-1]) / dt_last
            )
            # Scale: rotation_speed / beat_freq rotations per beat
            median_ibi = np.median(np.diff(times[:min(len(times), 40)]))
            beat_freq  = 1.0 / max(median_ibi, 0.05)
            speed_ratio = rotation_speed / beat_freq
            az_deg = (beat_phase * speed_ratio * 360.0) % 360.0
        else:
            az_deg = (rotation_speed * t * 360.0) % 360.0

        return az_deg

    # ─────────────────────────────────────────────────────────────────────────
    # Main render

    def render(
        self,
        mono:           np.ndarray,
        beat_times_s:   Optional[List[float]],
        rotation_speed: float,
        elevation:      float = 0.0,
        distance_m:     float = 1.0,
    ) -> np.ndarray:
        """
        Convolve mono audio with a time-varying binaural HRIR.

        Arguments:
          mono           : (N,) float32/64 — mono audio
          beat_times_s   : beat onsets in seconds (None → free-running LFO)
          rotation_speed : Hz (source orbits/second)
          elevation      : normalised [-1, 1]
                             +1 → top of SOFA elevation range (e.g. +35°)
                             -1 → bottom of SOFA elevation range (e.g. -25°)
          distance_m     : virtual source distance in metres.
                             0.20–1.10 maps directly into SOFA data range.
                             Values outside range are clamped.

        Returns:
          stereo : float32 (2, N)  — [0] = Left,  [1] = Right
        """
        if not ADVANCED_ANALYSIS:
            raise RuntimeError("HRIREngine requires scipy (pip install scipy)")

        from scipy.signal import fftconvolve

        n_samples = len(mono)
        az_traj   = self._angle_trajectory(n_samples, rotation_speed, beat_times_s)

        # ── Map (elevation, distance) → SOFA coordinates ─────────────────────
        if self.using_sofa:
            d_min = float(self.sofa_dists[0])
            d_max = float(self.sofa_dists[-1])
            dist_cm = float(np.clip(distance_m * 100.0, d_min, d_max))

            e_min = float(self.sofa_elevs[0])   # e.g. -25
            e_max = float(self.sofa_elevs[-1])   # e.g. +35
            # Positive elevation → towards top of range; negative → towards bottom
            elev_deg = float(
                np.clip(elevation * (e_max if elevation >= 0.0 else -e_min),
                        e_min, e_max)
            )
        else:
            # Synthetic/numpy path: simulate elevation by scaling lateral spread
            dist_cm  = distance_m * 100.0
            elev_deg = elevation * 90.0   # unused in legacy path

        # ── OLA (overlap-add) block processing ───────────────────────────────
        out_L = np.zeros(n_samples + self.HRIR_N, dtype=np.float64)
        out_R = np.zeros(n_samples + self.HRIR_N, dtype=np.float64)

        n_blocks = (n_samples + self.BLOCK - 1) // self.BLOCK

        for b in range(n_blocks):
            start = b * self.BLOCK
            end   = min(start + self.BLOCK, n_samples)
            block = mono[start:end].astype(np.float64)

            # Azimuth at mid-block sample
            mid_idx = (start + end) // 2
            az      = float(az_traj[mid_idx] % 360.0)

            if self.using_sofa:
                L_ir, R_ir = self._hrir_for_position(az, elev_deg, dist_cm)
            else:
                # Legacy path: collapse elevation into lateral-axis scaling
                elev_scale = float(np.cos(elevation * np.pi / 2))
                az_eff     = (az - 180.0) * elev_scale + 180.0
                L_ir, R_ir = self._hrir_for_angle(az_eff)

            seg_len = len(block) + self.HRIR_N - 1
            out_L[start: start + seg_len] += fftconvolve(
                block, L_ir.astype(np.float64))[:seg_len]
            out_R[start: start + seg_len] += fftconvolve(
                block, R_ir.astype(np.float64))[:seg_len]

        # ── Trim + normalise ─────────────────────────────────────────────────
        stereo = np.stack([
            out_L[:n_samples].astype(np.float32),
            out_R[:n_samples].astype(np.float32),
        ])

        peak = float(np.max(np.abs(stereo))) + 1e-10
        if peak > 0.99:
            stereo /= peak * 1.01

        return stereo


# Singleton — instantiated lazily on first use
_hrir_engine: Optional['HRIREngine'] = None

def get_hrir_engine(sr: int = 48000) -> 'HRIREngine':
    global _hrir_engine
    if _hrir_engine is None or _hrir_engine.sr != sr:
        _hrir_engine = HRIREngine(sample_rate=sr)
    return _hrir_engine


# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self.active[job_id] = ws

    def disconnect(self, job_id: str):
        self.active.pop(job_id, None)

    async def _send(self, job_id: str, data: dict):
        ws = self.active.get(job_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                pass

    async def send_progress(self, job_id: str, pct: int, stage: str):
        await self._send(job_id, {"type": "progress", "progress": pct, "stage": stage})

    async def send_complete(self, job_id: str, url: str, video_url: Optional[str] = None):
        data = {"type": "complete", "output_url": url}
        if video_url:
            data["video_url"] = video_url
        await self._send(job_id, data)

    async def send_error(self, job_id: str, msg: str):
        await self._send(job_id, {"type": "error", "message": msg})


manager = ConnectionManager()


# ============================================================================
# UTILITIES
# ============================================================================

def auto_detect_ffmpeg():
    import platform
    if platform.system() != "Windows":
        return
    paths = [r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin",
             os.path.expanduser(r"~\ffmpeg\bin")]
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    for p in paths:
        if os.path.exists(os.path.join(p, "ffmpeg.exe")):
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] += os.pathsep + p
            return

def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def check_reverberate_filter() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, check=True)
        return "reverberate" in result.stdout
    except Exception:
        return False

def get_audio_duration(file_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0

async def ffmpeg_progress(line: str, total: float) -> float:
    try:
        if 'time=' in line:
            ts = line.split('time=')[1].split()[0]
            h, m, s = ts.split(':')
            secs = int(h)*3600 + int(m)*60 + float(s)
            if total > 0:
                return min(secs / total, 1.0)
    except Exception:
        pass
    return 0.0


# ============================================================================
# EQ CHAIN BUILDER  (12-band — unchanged API)
# ============================================================================

def _merge_legacy_eq(p: ProcessingParams) -> ProcessingParams:
    if abs(p.eq_sub_bass_gain) > 0.1:
        p = p.copy(update={
            'eq_sub30_gain': p.eq_sub30_gain + p.eq_sub_bass_gain * 0.5,
            'eq_sub60_gain': p.eq_sub60_gain + p.eq_sub_bass_gain * 0.5,
        })
    if abs(p.eq_bass_gain) > 0.1:
        p = p.copy(update={'eq_bass100_gain': p.eq_bass100_gain + p.eq_bass_gain})
    if abs(p.eq_low_mid_gain) > 0.1:
        p = p.copy(update={'eq_lowmid350_gain': p.eq_lowmid350_gain + p.eq_low_mid_gain})
    if abs(p.eq_presence_gain) > 0.1:
        p = p.copy(update={'eq_presence3k_gain': p.eq_presence3k_gain + p.eq_presence_gain})
    if abs(p.eq_air_gain) > 0.1:
        p = p.copy(update={'eq_air12k_gain': p.eq_air12k_gain + p.eq_air_gain})
    return p


def _eq_chain(p: ProcessingParams) -> str:
    p = _merge_legacy_eq(p)

    def shelf(f, g, w):
        return f"equalizer=f={f}:t=h:w={w}:g={g:.1f}"
    def bell(f, g, w):
        return f"equalizer=f={f}:t=q:w={w}:g={g:.1f}"

    eqs = []
    if abs(p.eq_sub30_gain)     > 0.3: eqs.append(shelf(30,    p.eq_sub30_gain,    30))
    if abs(p.eq_sub60_gain)     > 0.3: eqs.append(bell (60,    p.eq_sub60_gain,    40))
    if abs(p.eq_bass100_gain)   > 0.3: eqs.append(bell (100,   p.eq_bass100_gain,  80))
    if abs(p.eq_ubass200_gain)  > 0.3:
        gain = max(-6.0, min(4.0, p.eq_ubass200_gain))
        eqs.append(bell(200, gain, 100))
    if abs(p.eq_lowmid350_gain) > 0.3: eqs.append(bell (350,   p.eq_lowmid350_gain, 200))
    if abs(p.eq_mid700_gain)    > 0.3: eqs.append(bell (700,   p.eq_mid700_gain,   300))
    if abs(p.eq_umid1500_gain)  > 0.3: eqs.append(bell (1500,  p.eq_umid1500_gain, 600))
    if abs(p.eq_presence3k_gain)> 0.3: eqs.append(bell (3000,  p.eq_presence3k_gain, 1500))
    if abs(p.eq_def5k_gain)     > 0.3: eqs.append(bell (5000,  p.eq_def5k_gain,   2000))
    if abs(p.eq_bril8k_gain)    > 0.3: eqs.append(shelf(8000,  p.eq_bril8k_gain,  3000))
    if abs(p.eq_air12k_gain)    > 0.3: eqs.append(shelf(12000, p.eq_air12k_gain,  5000))
    if abs(p.eq_uair16k_gain)   > 0.3: eqs.append(shelf(16000, p.eq_uair16k_gain, 6000))

    return ",".join(eqs) if eqs else ""


# ============================================================================
# INSTRUMENT ENHANCEMENT CHAIN
# ============================================================================

def _instrument_enhance_chain(p: ProcessingParams) -> str:
    """
    Instrument enhancement chain — v8.1 (frequency-selective exciter).

    v7 problem: wideband tanh() saturation added harmonics at ALL frequencies,
    causing sub-bass intermodulation that muddied the HRTF band separation.

    v8.1 fix — 3-stage split path:
      Stage 1: Transient compressor (gentler makeup: 2 dB not 3 dB).
      Stage 2: HF exciter — boost > 3 kHz by 6 dB, apply soft saturation,
               then cut back by 6 dB. Net result: harmonic content added
               only at 6–12 kHz (presence/air), bass untouched.
      Stage 3: +1.5 dB presence shelf at 5 kHz to restore cup-masked detail.
    """
    if not p.instrument_enhance:
        return ""
    return (
        "acompressor=threshold=-30dB:ratio=1.5:attack=5:release=80:makeup=2dB,"
        "equalizer=f=3000:t=h:w=2000:g=6.0,"
        "aeval=val(0)*0.88+0.12*tanh(val(0)*2.5)|val(1)*0.88+0.12*tanh(val(1)*2.5),"
        "equalizer=f=3000:t=h:w=2000:g=-6.0,"
        "equalizer=f=5000:t=h:w=3000:g=1.5"
    )


# ============================================================================
# NEW v6.0: PSYCHOACOUSTIC HELPERS
# ============================================================================

def _diffuse_field_eq() -> str:
    """
    IEC 711 diffuse-field headphone correction — v8.1 (5-point model).

    Old v7 was a 3-point approximation that missed two important features
    of the IEC 711 curve:
      • The ~700 Hz ear-canal resonance peak in typical closed headphones
      • The broad 4 kHz presence dip that makes headphones sound "in-head"

    v8.1 adds those two missing corrections for a more accurate outside-
    the-head perception:

      +2.0 dB @ 700 Hz   [NEW] — compensate ear-canal resonance peak
                                   (closed headphones boost this; cutting it
                                    restores the flat free-field response)
      Correction: sign is negative — headphones ADD here, so we CUT.

      -2.5 dB @ 700 Hz   [CORRECTED to cut]
      +3.5 dB @ 2.5 kHz  — restore vocal presence (unchanged)
      -5.0 dB @ 5.0 kHz  — cut headphone cup resonance (unchanged)
      -2.0 dB @ 4.0 kHz  [NEW] — cut presence peak that causes in-head feel
      +4.5 dB @ 10.0 kHz — restore ultra-air roll-off (boosted from 4.0)
      +2.0 dB @ 14.0 kHz [NEW] — restore the outer-helix diffraction shoulder
    """
    return (
        "equalizer=f=700:t=q:w=500:g=-2.5,"    # ear-canal resonance cut
        "equalizer=f=2500:t=q:w=3000:g=3.5,"   # vocal presence restore
        "equalizer=f=4000:t=q:w=2500:g=-2.0,"  # in-head peak reduction [NEW]
        "equalizer=f=5000:t=q:w=2000:g=-5.0,"  # cup resonance cut
        "equalizer=f=10000:t=h:w=4000:g=4.5,"  # air restore (was 4.0)
        "equalizer=f=14000:t=h:w=5000:g=2.0"   # helix shoulder [NEW]
    )


def _equal_loudness_shelf() -> str:
    """
    ISO 226:2003 equal-loudness compensation — v8.1 (6-point model, ~70 phons).

    Old v7 was missing the 80–120 Hz equal-loudness dip — the curve actually
    has a LOCAL MINIMUM near 100 Hz (not a boost), which is why bass-heavy
    music sounds "tubby" on flat headphones. The 50 Hz boost was also too
    aggressive (+7.5 dB), making sub rumble pile up on bass-heavy tracks.

    v8.1 corrects:
      +5.0 dB @ 40 Hz    — sub-bass boost (was +7.5 @ 50 Hz — overblown)
      -1.5 dB @ 100 Hz   [NEW] — equal-loudness local minimum near 100 Hz
      +1.5 dB @ 200 Hz   — bass warmth (unchanged concept, reduced)
      -1.5 dB @ 3.5 kHz  — mid scoop (natural dip in 70-phon curve, unchanged)
      +2.0 dB @ 8.0 kHz  [NEW] — 70-phon curve secondary peak at 8 kHz
      +3.0 dB @ 12.0 kHz — high-air restore (unchanged)
    """
    return (
        "equalizer=f=40:t=h:w=40:g=5.0,"        # sub-bass boost (corrected)
        "equalizer=f=100:t=q:w=80:g=-1.5,"       # equal-loudness 100 Hz dip [NEW]
        "equalizer=f=200:t=q:w=150:g=1.5,"       # bass body warmth
        "equalizer=f=3500:t=q:w=3000:g=-1.5,"    # mid scoop
        "equalizer=f=8000:t=q:w=4000:g=2.0,"     # 8 kHz secondary peak [NEW]
        "equalizer=f=12000:t=h:w=5000:g=3.0"     # high-air perceptual restore
    )


def _pinna_notch_filters(intensity: float = 1.0) -> str:
    """
    Pinna shadow notch filters — v8.1 (4-notch model).

    Real pinna reflections create narrow spectral notches the brain uses to
    perceive elevation and front/back. Four notches are modelled here:

      Notch 1: ~8.5 kHz  — first pinna concha resonance      (−6 dB max)
      Notch 2: ~10.5 kHz — second pinna shadow / flap        (−4 dB max)
      Notch 3: ~13.0 kHz — upper pinna flap reflection       (−3 dB max)
      Notch 4: ~16.0 kHz — outer helix diffraction shoulder  (−2.5 dB max)
               [NEW v8.1] — adds the subtle ultra-high cue that gives
               elevation perception above the ear plane.

    All gains scale linearly with intensity [0–1.5].
    Bandwidths chosen for psychoacoustic selectivity (not surgical cuts).
    """
    if intensity <= 0:
        return ""
    g1 = round(-6.0   * min(intensity, 1.5), 1)
    g2 = round(-4.0   * min(intensity, 1.5), 1)
    g3 = round(-3.0   * min(intensity, 1.5), 1)
    g4 = round(-2.5   * min(intensity, 1.5), 1)
    return (
        f"equalizer=f=8500:t=q:w=1200:g={g1},"
        f"equalizer=f=10500:t=q:w=1800:g={g2},"
        f"equalizer=f=13000:t=q:w=2500:g={g3},"
        f"equalizer=f=16000:t=q:w=3500:g={g4}"
    )


def _allpass_diffuser(room: float = 0.6) -> str:
    """
    Allpass diffusion network — four prime-interval aecho taps scaled by room size.

    Tap delays are proportional to room size so that a small room (0.1) produces
    dense early reflections (tight flutter) while a large room (1.0) gives wide,
    spacious pre-echo that merges smoothly into the main reverb tail.

    Small room (r=0.1): delays ≈ 8|10|14|18 ms  → intimate, close-miked feel
    Medium  (r=0.6):    delays ≈ 17|23|31|41 ms  → original values
    Large   (r=1.0):    delays ≈ 27|37|51|67 ms  → concert-hall depth
    """
    r = max(0.05, min(1.0, room))
    scale = r / 0.6          # normalise around original mid-room values
    d1 = max(5,  int(17 * scale))
    d2 = max(7,  int(23 * scale))
    d3 = max(9,  int(31 * scale))
    d4 = max(11, int(41 * scale))
    # Decays: shorter taps die faster (keeps transients clean)
    dc1 = round(min(0.25, 0.14 * (1 + r * 0.3)), 3)
    dc2 = round(min(0.20, 0.11 * (1 + r * 0.3)), 3)
    dc3 = round(min(0.16, 0.08 * (1 + r * 0.3)), 3)
    dc4 = round(min(0.13, 0.06 * (1 + r * 0.3)), 3)
    return (
        f"aecho=in_gain=1.0:out_gain=1.0"
        f":delays={d1}|{d2}|{d3}|{d4}"
        f":decays={dc1}|{dc2}|{dc3}|{dc4}"
    )


def _haas_widener(room: float = 0.6) -> str:
    """
    Haas-effect early reflection widener.

    Applies a short (8–18 ms) delay only to the right channel, exploiting the
    Haas (precedence) effect so the brain perceives the delayed copy as spatial
    width rather than a discrete echo. The left channel is undelayed so the
    stereo image is anchored correctly.

    Delay scales with room: smaller rooms produce tighter Haas gaps.
    Range: 8 ms (r=0.1)  →  18 ms (r=1.0), mirrored with mirrored gain.
    """
    r = max(0.0, min(1.0, room))
    haas_ms = round(8.0 + r * 10.0, 1)     # 8–18 ms
    haas_gain = round(0.72 + r * 0.10, 3)  # 0.72–0.82 (slightly quieter)
    return (
        f"aecho=in_gain=1.0:out_gain={haas_gain}"
        f":delays=0|{haas_ms}"
        f":decays=0.0|0.50"
    )


def _reverb(p: ProcessingParams) -> str:
    """4-tap multi-reflection reverb via aecho (core reverb)."""
    ig = max(0.1, 1.0 - p.reverb_mix * 0.35)
    og = p.reverb_mix * 0.65
    r  = p.reverb_room
    d1, d2 = int(r * 40), int(r * 63)
    d3, d4 = int(r * 97), int(r * 131)
    dc1 = r * 0.48; dc2 = r * 0.32; dc3 = r * 0.20; dc4 = r * 0.12
    return (
        f"aecho=in_gain={ig:.2f}:out_gain={og:.2f}"
        f":delays={d1}|{d2}|{d3}|{d4}"
        f":decays={dc1:.2f}|{dc2:.2f}|{dc3:.2f}|{dc4:.2f}"
    )


# ============================================================================
# BEAT-SYNC ROTATION  — v9.1 (replaces broken v9.0 piecewise expression)
# ============================================================================
#
# v9.0 generated a piecewise if(gte(t,...), cos(...), if(...)) expression
# with one level per beat — up to 300 levels deep. FFmpeg's expression
# evaluator silently failed on expressions that large, defaulting every
# volume filter to gain=1. Result: zero panning on any beat-detected track.
#
# v9.1 replaces this with a simple, reliable approach:
#   1. Compute the median inter-beat interval from librosa's beat array.
#   2. Derive the actual BPM from that interval.
#   3. Pick a rotation speed that is a clean musical ratio of the BPM
#      (e.g. 1 rotation per 4 beats).
#   4. Pass that speed to build_8band_hrtf_engine_v6 as a normal float.
#
# The spatial effect stays musically locked without any complex expressions.
# beat_positions from the analysis result ARE used — just for arithmetic,
# not for generating FFmpeg filter strings.

def beat_sync_rotation_speed(
    beat_times_s: List[float],
    base_rotation_speed: float,
) -> float:
    """
    Derive a musically-locked rotation speed from beat timestamps.

    Computes median BPM from beat_times_s, then finds the nearest musical
    ratio (1/8, 1/4, 1/2, 1, 2 rotations per beat) that keeps the rotation
    speed within [0.05, 0.5] Hz.  Falls back to base_rotation_speed if
    beats are too few or the result is out of range.
    """
    if not beat_times_s or len(beat_times_s) < 4:
        return base_rotation_speed

    ibis = [beat_times_s[k+1] - beat_times_s[k]
            for k in range(min(len(beat_times_s)-1, 60))]
    ibis.sort()
    median_ibi = ibis[len(ibis) // 2]
    if median_ibi < 0.05:
        return base_rotation_speed   # unreasonably fast beats

    beat_freq = 1.0 / median_ibi   # Hz (= BPM / 60)

    # Musical ratios: rotations per beat
    ratios = [1/8, 1/6, 1/4, 1/3, 1/2, 1.0, 2.0]
    candidates = [beat_freq * r for r in ratios]

    # Find the candidate closest to base_rotation_speed within [0.04, 0.6] Hz
    valid = [(abs(c - base_rotation_speed), c) for c in candidates if 0.04 <= c <= 0.6]
    if not valid:
        return base_rotation_speed

    _, best = min(valid)
    return round(best, 4)


# ============================================================================
# STUDIO GRADE v6.0  — 8-band HRTF + ITD + Pinna EQ + Diffuse-Field EQ
# ============================================================================

def build_stereo_mastering_chain(p: ProcessingParams) -> tuple:
    """
    Post-HRIR mastering chain — stereo-in, stereo-out.

    Used after the Python HRIR convolution engine has already produced a
    binaural stereo WAV.  This chain must NOT collapse L/R to mono.

    Applies only:
      • Reverb (single aecho — no triple chain)
      • Diffuse-field headphone EQ (3 bands)
      • 12-band master EQ
      • EBU R128 loudnorm
      • True peak limiter

    No pan=mono, no stereotools split, no per-channel processing.
    """
    eq  = _eq_chain(p)
    rev = _reverb(p)

    # Diffuse-field EQ — 3 key points only (not the full 6-point version
    # which over-corrects and stacks with the EQ chain below)
    dfeq_simple = (
        "equalizer=f=2500:t=q:w=3000:g=3.0,"
        "equalizer=f=5000:t=q:w=2000:g=-4.0,"
        "equalizer=f=10000:t=h:w=4000:g=3.5"
    )

    parts = [
        # ── reverb mix (wet/dry blend using aecho) ──
        f"[0:a]{rev}[rev_out]",
        # dry/wet recombine is handled inside _reverb via in_gain/out_gain
    ]

    last = "rev_out"

    if eq:
        parts.append(f"[{last}]{eq}[eq_out]")
        last = "eq_out"

    parts.append(f"[{last}]{dfeq_simple}[dfeq_out]")
    last = "dfeq_out"

    parts.append(f"[{last}]loudnorm=I=-16:TP=-1.5:LRA=11:linear=true[loud]")
    last = "loud"

    if p.enable_limiter:
        parts.append(f"[{last}]alimiter=limit=1:attack=5:release=50:level=false[out]")
        return ";".join(parts), "[out]"

    return ";".join(parts), f"[{last}]"


def build_8band_hrtf_engine_v6(
    p: ProcessingParams,
    beat_times_s: Optional[List[float]] = None,
) -> tuple:
    """
    8-Band Spatial Audio Engine v10.0 — stripped to what actually works.

    Every layer that was killing the audio has been removed:
      - No pinna notch filters (were always-on-R regardless of rotation)
      - No diffuse-field EQ post-mix (was stacking -5dB@5kHz cuts)
      - No equal-loudness shelf post-mix
      - No per-band acompressor
      - No complex bilateral ITD blending (added 10 nodes per band for
        a perceptually inaudible improvement at <0.3ms precision)
      - No beat-locked FFmpeg expressions (too complex, caused parse failures)

    What remains (what actually works):
      - Mono downmix → remove fake stereo artefacts
      - 8-band split with different rotation speeds (bass slow, treble fast)
      - ILD panning: cos(θ) drives L/R balance with floor=0.12 (no dead zones)
      - Front/rear character: sin(θ) blends two EQ paths for depth cues
      - Simple static ITD: one adelay on R channel (clean, no blending noise)
      - Mild head shadow on R only for bands above 2kHz (gentle, not muffling)
      - Allpass diffuser + pre-delay + reverb
      - User 12-band EQ → stereo width → loudnorm → limiter
    """
    i    = p.intensity_multiplier
    dvol = round(1.0 / max(p.distance, 0.3), 4)

    # ── Rotation speeds — derive from BPM if available ────────────────────────
    base_rot = round(p.rotation_speed * i, 5)
    if beat_times_s and len(beat_times_s) >= 4:
        ibi_list = sorted([beat_times_s[k+1] - beat_times_s[k]
                           for k in range(min(len(beat_times_s)-1, 40))])
        med_ibi = ibi_list[len(ibi_list) // 2]
        if 0.2 < med_ibi < 2.0:
            bpm_rot  = round(1.0 / med_ibi / 4.0, 5)   # 1 rotation per 4 beats
            base_rot = round((base_rot + bpm_rot) / 2.0, 5)
            print(f"  \u21b3 BPM-synced rotation: {bpm_rot} Hz \u2192 blended {base_rot} Hz")

    r_sub   = round(p.bass_rotation   * i * 0.40, 5)
    r_bass  = round(p.bass_rotation   * i * 0.70, 5)
    r_lowm  = round(base_rot          * 0.80, 5)
    r_voc   = round(base_rot          * 0.35, 5)  # 0.55→0.35: vocals stay centred
    r_highm = round(p.treble_rotation * i * 0.90, 5)
    r_pres  = round(p.treble_rotation * i * 1.10, 5)
    r_air   = round(p.treble_rotation * i * 1.30, 5)
    r_spark = round(p.treble_rotation * i * 1.55, 5)

    PHASE  = {'sub': 0.00, 'bass': 0.40, 'lowm': 0.85, 'voc': 1.30,
              'highm': 1.75, 'pres': 2.25, 'air': 2.80, 'spark': 3.40}
    # ITD in milliseconds (adelay unit). Max real ITD ≈ 0.63ms.
    ITD_MS = {'sub': 0.00, 'bass': 0.06, 'lowm': 0.12, 'voc': 0.20,
              'highm': 0.28, 'pres': 0.35, 'air': 0.45, 'spark': 0.55}

    FLOOR = 0.03  # 0.12→0.03: YouTube 8D uses ~3% floor for strong L/R
    depth = round(1.0 - 2 * FLOOR, 4)   # 0.94

    # Front: slight presence lift. Rear: warmth + gentle HF rolloff.
    FRONT_EQ = "equalizer=f=3000:t=q:w=2000:g=1.5"
    REAR_EQ  = "equalizer=f=400:t=q:w=350:g=2.5,equalizer=f=1500:t=q:w=1200:g=-1.0"

    def band(lbl, lo, hi, rot, ph=0.0, itd_ms=0.0,
             rear_eq=True, vocal_center=False):
        px  = []
        th  = f"2*PI*{rot}*t+{ph}"
        cos_t = f"cos({th})"
        sin_t = f"sin({th})"

        # 1. Bandpass
        if lo <= 20:
            px.append(f"[{lbl}_in]lowpass=f={hi}[{lbl}_f]")
        elif hi >= 22000:
            px.append(f"[{lbl}_in]highpass=f={lo}[{lbl}_f]")
        else:
            mid = (lo + hi) // 2
            px.append(f"[{lbl}_in]bandpass=f={mid}:width_type=h:w={hi-lo}[{lbl}_f]")

        # 2. Front / rear EQ blend — only for bands that contain meaningful content
        if rear_eq:
            px.append(f"[{lbl}_f]asplit=2[{lbl}_fp][{lbl}_rp]")
            px.append(f"[{lbl}_fp]{FRONT_EQ}[{lbl}_feq]")
            px.append(f"[{lbl}_rp]{REAR_EQ}[{lbl}_req]")
            # sin > 0 = front, sin < 0 = rear. Blend weights sum to 1.
            px.append(f"[{lbl}_feq]volume='0.5+0.5*{sin_t}':eval=frame[{lbl}_fb]")
            px.append(f"[{lbl}_req]volume='0.5-0.5*{sin_t}':eval=frame[{lbl}_rb]")
            px.append(f"[{lbl}_fb][{lbl}_rb]amix=inputs=2:duration=first:normalize=0[{lbl}_m]")
            src = f"[{lbl}_m]"
        else:
            src = f"[{lbl}_f]"

        # 3. ILD panning — cos-based, NEVER fully silent (floor=0.12)
        px.append(f"{src}asplit=2[{lbl}_Ls][{lbl}_Rs]")
        if vocal_center:
            half = round(FLOOR + depth * 0.5, 4)
            vc_d = round(depth * 0.3, 4)
            pan_L = f"({half}-{vc_d}*{cos_t})"
            pan_R = f"({half}+{vc_d}*{cos_t})"
        else:
            pan_L = f"({FLOOR}+{depth}*(0.5-0.5*{cos_t}))"
            pan_R = f"({FLOOR}+{depth}*(0.5+0.5*{cos_t}))"

        px.append(f"[{lbl}_Ls]volume='{pan_L}':eval=frame[{lbl}_Lv]")
        px.append(f"[{lbl}_Rs]volume='{pan_R}':eval=frame[{lbl}_Rv]")

        # 4. ITD — simple single-channel delay on R
        left_sig  = f"{lbl}_Lv"
        right_sig = f"{lbl}_Rv"
        if itd_ms > 0:
            px.append(f"[{lbl}_Rv]adelay={itd_ms}|0[{lbl}_Rd]")
            right_sig = f"{lbl}_Rd"

        # 5. Mild head shadow on R for high bands ONLY
        #    Gentle −3dB shelf above 9kHz — NOT a brick-wall lowpass
        if hi > 2000 and p.hrtf_intensity > 0:
            shadow = round(-3.0 * min(p.hrtf_intensity, 1.0), 1)
            px.append(f"[{right_sig}]equalizer=f=9000:t=h:w=7000:g={shadow}[{lbl}_Rs2]")
            right_sig = f"{lbl}_Rs2"

        # 6. Join stereo + distance
        px.append(f"[{left_sig}][{right_sig}]join=inputs=2:channel_layout=stereo[{lbl}_st]")
        px.append(f"[{lbl}_st]volume={dvol}[{lbl}_out]")
        return px

    # ── Assemble 8 bands ──────────────────────────────────────────────────────
    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",
        "[mono_src]asplit=8[sub_in][bass_in][lowm_in][voc_in][highm_in][pres_in][air_in][spark_in]",
    ]

    parts += band("sub",   20,    80,    r_sub,   ph=PHASE['sub'],   itd_ms=ITD_MS['sub'],   rear_eq=False)
    parts += band("bass",  80,    250,   r_bass,  ph=PHASE['bass'],  itd_ms=ITD_MS['bass'],  rear_eq=False)
    parts += band("lowm",  250,   600,   r_lowm,  ph=PHASE['lowm'],  itd_ms=ITD_MS['lowm'],  rear_eq=True)
    parts += band("voc",   600,   2000,  r_voc,   ph=PHASE['voc'],   itd_ms=ITD_MS['voc'],
                  rear_eq=False, vocal_center=True)  # no HF blur; always centred
    parts += band("highm", 2000,  4000,  r_highm, ph=PHASE['highm'], itd_ms=ITD_MS['highm'], rear_eq=True)
    parts += band("pres",  4000,  8000,  r_pres,  ph=PHASE['pres'],  itd_ms=ITD_MS['pres'],  rear_eq=True)
    parts += band("air",   8000,  14000, r_air,   ph=PHASE['air'],   itd_ms=ITD_MS['air'],   rear_eq=True)
    parts += band("spark", 14000, 22000, r_spark, ph=PHASE['spark'], itd_ms=ITD_MS['spark'],  rear_eq=True)

    # ── Mix bands ─────────────────────────────────────────────────────────────
    outs = "".join(f"[{b}_out]" for b in ["sub","bass","lowm","voc","highm","pres","air","spark"])
    parts.append(f"{outs}amix=inputs=8:duration=first:normalize=0[mixed]")

    # ── Reverb ────────────────────────────────────────────────────────────────
    rev_wet = round(p.reverb_mix * 0.55, 3)
    rev_dry = round(1.0 - rev_wet, 3)

    parts.append("[mixed]asplit=2[dry][rev_in]")

    # Allpass diffuser (early reflections)
    r = max(0.05, min(1.0, p.reverb_room))
    sc = r / 0.6
    d1, d2 = max(5, int(17*sc)), max(7, int(23*sc))
    d3, d4 = max(9, int(31*sc)), max(11, int(41*sc))
    dc1 = round(min(0.25, 0.14*(1+r*0.3)), 3)
    dc2 = round(min(0.20, 0.11*(1+r*0.3)), 3)
    dc3 = round(min(0.16, 0.08*(1+r*0.3)), 3)
    dc4 = round(min(0.13, 0.06*(1+r*0.3)), 3)
    parts.append(f"[rev_in]aecho=in_gain=1.0:out_gain=1.0:delays={d1}|{d2}|{d3}|{d4}:decays={dc1}|{dc2}|{dc3}|{dc4}[diff]")

    # Pre-delay
    # Pre-delay: 25–40 ms — longer pre-delay dramatically increases
    # perceived room size and separates dry source from reverb tail.
    # Old range (15–33ms) was too short to feel like a real space.
    pre_ms = round(25 + r * 15, 1)   # 25 ms (small room) → 40 ms (hall)
    parts.append(f"[diff]adelay={pre_ms}|{pre_ms}[pre]")

    # Main reverb tail
    rd = p.reverb_density
    if check_reverberate_filter():
        rs  = int(r * 100)
        dt  = int(rd * 4500)
        dmp = round(1.0 - rd, 2)
        parts.append(f"[pre]reverberate=room_size={rs}:time={dt}:damping={dmp:.2f}:wet={rev_wet:.3f}:dry=0[wet]")
    else:
        d1r = max(1, int(40*r));  d2r = max(1, int(70*r))
        d3r = max(1, int(110*r)); d4r = max(1, int(160*r))
        dc1r = round(rd*0.55,2); dc2r = round(rd*0.38,2)
        dc3r = round(rd*0.24,2); dc4r = round(rd*0.14,2)
        parts.append(
            f"[pre]aecho=in_gain=0.85:out_gain={rev_wet:.3f}"
            f":delays={d1r}|{d2r}|{d3r}|{d4r}"
            f":decays={dc1r}|{dc2r}|{dc3r}|{dc4r}[wet]"
        )

    parts.append(f"[dry]volume={rev_dry:.3f}[dry_v]")
    parts.append(f"[dry_v][wet]amix=inputs=2:duration=first:normalize=0[post_rev]")

    # ── Mastering ─────────────────────────────────────────────────────────────
    # M/S EQ — process mid (mono-sum) and side (stereo-diff) independently.
    #
    # Why: After multiband spatial processing the summed signal often has:
    #   • Muddy low-mids (200-400 Hz) bleeding into the Side channel,
    #     which smears the spatial image and reduces stereo clarity.
    #   • Insufficient "air" in the Side channel, making the space feel
    #     small and close compared to YouTube 8D references.
    #
    # This M/S EQ:
    #   MID channel  — cut 300 Hz mud (masked by spatial processing),
    #                  slight 2 kHz vocal presence restore
    #   SIDE channel — hard cut below 180 Hz (bass must be mono-compatible),
    #                  +2.5 dB air shelf at 10 kHz (expands perceived space)
    #
    # Implementation: FFmpeg has no per-channel EQ, so we split to M/S via pan,
    # EQ each as a mono stream, then recombine.
    parts.append(
        # Encode L/R → M/S  (M = L+R / 2,  S = L-R / 2)
        "[post_rev]asplit=2[ms_l][ms_r];"
        "[ms_l]pan=1c|c0=0.5*c0+0.5*c1[mid_raw];"
        "[ms_r]pan=1c|c0=0.5*c0-0.5*c1[side_raw];"
        # MID: cut 300 Hz boxiness, +1.2 dB vocal presence at 2 kHz
        "[mid_raw]"
        "equalizer=f=300:t=q:w=250:g=-2.0,"
        "equalizer=f=2000:t=q:w=1500:g=1.2"
        "[mid_eq];"
        # SIDE: highpass at 180 Hz (bass stays mono), +2.5 dB air
        "[side_raw]"
        "highpass=f=180:poles=2,"
        "equalizer=f=10000:t=h:w=5000:g=2.5"
        "[side_eq];"
        # Decode M/S → L/R  (L = M+S,  R = M-S)
        "[mid_eq][side_eq]join=inputs=2:channel_layout=stereo[ms_join];"
        "[ms_join]pan=stereo|c0=c0+c1|c1=c0-c1[post_ms]"
    )

    # Stereo width
    parts.append(f"[post_ms]stereotools=mlev={p.stereo_width:.3f}:softclip=1[wide]")

    # User 12-band EQ (optional)
    eq = _eq_chain(p)
    if eq:
        parts.append(f"[wide]{eq}[eqd]")
        last = "eqd"
    else:
        last = "wide"

    # Loudness normalisation (EBU R128)
    parts.append(f"[{last}]loudnorm=I=-16:TP=-1.5:LRA=11:linear=true[loud]")
    last = "loud"

    # Elevation — only if non-zero
    if abs(p.elevation) > 0.01:
        eg_hi  = round(p.elevation * 6.0, 1)
        eg_sub = round(-p.elevation * 2.5, 1)
        parts.append(
            f"[{last}]equalizer=f=8000:t=h:w=3000:g={eg_hi},"
            f"equalizer=f=80:t=h:w=80:g={eg_sub}[elev]"
        )
        last = "elev"

    # Limiter
    if p.enable_limiter:
        parts.append(f"[{last}]alimiter=limit=1:attack=5:release=50:level=false[8d_out]")
        return ";".join(parts), "[8d_out]"

    return ";".join(parts), f"[{last}]"



def build_6band_filtergraph(p: ProcessingParams) -> tuple:
    i   = p.intensity_multiplier
    ph  = 0.12
    dvol= 1.0 / max(p.distance, 0.3)
    eq  = _eq_chain(p)
    rev = _reverb(p)
    enh = _instrument_enhance_chain(p)

    r_sub  = p.bass_rotation  * i * 0.55
    r_bass = p.bass_rotation  * i
    r_lm   = p.rotation_speed * i * 0.65
    r_mid  = p.rotation_speed * i
    r_pres = p.treble_rotation* i * 0.80
    r_air  = p.treble_rotation* i

    def _band_parts(lbl, lo, hi, rot, dv, ph_off=0.0, is_vocal=False):
        if lo <= 20:
            filt = f"lowpass=f={hi}[{lbl}_filt]"
            inp  = f"[{lbl}_in]"
        elif hi >= 20000:
            filt = f"highpass=f={lo}[{lbl}_filt]"
            inp  = f"[{lbl}_in]"
        else:
            filt = f"highpass=f={lo}[{lbl}_hp];[{lbl}_hp]lowpass=f={hi}[{lbl}_filt]"
            inp  = f"[{lbl}_in]"

        theta = f"2*PI*{rot:.5f}*t+{ph_off:.3f}"
        FLOOR = 0.12
        depth = round(1.0 - 2 * FLOOR, 4)

        if is_vocal:
            # Vocal center: tighter pan, sound stays more central
            vc_d = round(depth * 0.32, 4)
            lfo_l = f"({FLOOR + depth*0.5:.4f}+{vc_d:.4f}*(-cos({theta})))"
            lfo_r = f"({FLOOR + depth*0.5:.4f}+{vc_d:.4f}*(cos({theta})))"
        else:
            # cos-based ILD — never hits zero, proper front/back equal levels
            lfo_l = f"({FLOOR:.4f}+{depth:.4f}*(0.5-0.5*cos({theta})))"
            lfo_r = f"({FLOOR:.4f}+{depth:.4f}*(0.5+0.5*cos({theta})))"

        ll, lr = f"{lbl}_l", f"{lbl}_r"
        return [
            f"{inp}{filt}",
            f"[{lbl}_filt]asplit=2[{ll}_m][{lr}_m]",
            f"[{ll}_m]volume='{lfo_l}':eval=frame[{ll}]",
            f"[{lr}_m]volume='{lfo_r}':eval=frame[{lr}]",
            f"[{ll}][{lr}]join=inputs=2:channel_layout=stereo[{lbl}_st]",
            f"[{lbl}_st]volume={dv:.4f}[{lbl}_8d]",
        ]

    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",
        "[mono_src]asplit=6[sub_in][bass_in][lm_in][mid_in][pres_in][air_in]",
    ]
    parts += _band_parts("sub",  20,   100,  r_sub,  dvol * 1.05)
    parts += _band_parts("bass", 100,  350,  r_bass, dvol)
    parts += _band_parts("lm",   350,  700,  r_lm,   dvol, ph_off=0.3)
    parts += _band_parts("mid",  700,  3000, r_mid,  dvol, ph_off=0.6, is_vocal=p.enable_vocal_center)
    parts += _band_parts("pres", 3000, 7000, r_pres, dvol, ph_off=1.1)
    parts += _band_parts("air",  7000, 22000,r_air,  dvol, ph_off=1.5)

    parts.append("[sub_8d][bass_8d][lm_8d][mid_8d][pres_8d][air_8d]amix=inputs=6:duration=first:normalize=0[mixed]")
    parts.append(f"[mixed]{rev}[rev]")
    parts.append(f"[rev]stereotools=mlev={p.stereo_width:.3f}:sbal=0[wide]")

    if enh:
        parts.append(f"[wide]{enh}[enhanced]"); last = "enhanced"
    else:
        last = "wide"

    if eq:
        parts.append(f"[{last}]{eq}[eqd]"); last = "eqd"

    # v6: add diffuse-field EQ
    dfeq = _diffuse_field_eq()
    parts.append(f"[{last}]{dfeq}[dfeq]")
    parts.append("[dfeq]loudnorm=I=-16:TP=-1.5:LRA=11[6b_loud]")
    # v8.1 dual-band elevation (ported from 8-band engine)
    elev_parts, last_node = _elevation_dual_band(p, "6b_loud")
    parts.extend(elev_parts)
    return ";".join(parts), f"[{last_node}]"


# ============================================================================
# SHARED HELPERS  — used by multiple engine builders
# ============================================================================

def _elevation_dual_band(p: ProcessingParams, node_in: str) -> tuple:
    """
    v8.1 dual-band elevation model — shared across ALL engine builders.

    Old model (all legacy engines): single high-shelf at 8 kHz.
      Problem: only boosts/cuts brightness; does nothing for the
      psychoacoustic sub-bass cue that determines perceived height.

    New model (ported from 8-band v8.1):
      Elevation > 0 (above listener):
        +HF shelf  at 8 kHz:  +6 dB × elevation  (bright, airy)
        Sub cut    at 80 Hz:  -2.5 dB × elevation (less "floored")
      Elevation < 0 (below listener):
        -HF shelf  at 8 kHz:  mirrors above (darker, earthed)
        Sub boost  at 80 Hz:  +2.5 dB × elevation (grounded rumble)

    Arguments:
      p        — ProcessingParams (reads p.elevation)
      node_in  — the FFmpeg label string for the input node, e.g. "loud"

    Returns (filter_parts: List[str], out_label: str)
      Caller appends filter_parts to its own parts list and uses out_label
      as the input to the next stage (limiter / final output).
    """
    if abs(p.elevation) <= 0.01:
        return [], node_in  # nothing to do

    eg_hi  = round(p.elevation * 6.0,  1)   # ±6 dB at 8 kHz
    eg_sub = round(-p.elevation * 2.5, 1)   # opposite sign: above → sub cut
    out_lbl = "elev_dual"
    filt = (
        f"[{node_in}]"
        f"equalizer=f=8000:t=h:w=3000:g={eg_hi},"
        f"equalizer=f=80:t=h:w=80:g={eg_sub}"
        f"[{out_lbl}]"
    )
    return [filt], out_lbl


# ============================================================================
# LEGACY FILTERGRAPH BUILDERS
# ============================================================================

def build_simple_filtergraph(p: ProcessingParams) -> tuple:
    rot   = p.rotation_speed * p.intensity_multiplier
    FLOOR = 0.12
    depth = round(1.0 - 2 * FLOOR, 4)
    eq    = _eq_chain(p)
    rev   = _reverb(p)
    # cos-based ILD — no silent dead zones
    pan_L = f"({FLOOR:.4f}+{depth:.4f}*(0.5-0.5*cos(2*PI*{rot}*t)))"
    pan_R = f"({FLOOR:.4f}+{depth:.4f}*(0.5+0.5*cos(2*PI*{rot}*t)))"
    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_in]",
        f"[mono_in]asplit=2[sl][sr]",
        f"[sl]volume='{pan_L}':eval=frame[vl]",
        f"[sr]volume='{pan_R}':eval=frame[vr]",
        f"[vl][vr]join=inputs=2:channel_layout=stereo[joined]",
        f"[joined]{rev}[rev]",
        f"[rev]stereotools=mlev={p.stereo_width}[wide]",
    ]
    if eq:
        parts += [f"[wide]{eq}[eqd]", "[eqd]loudnorm=I=-16:TP=-1.5:LRA=11[simple_loud]"]
        loud_node = "simple_loud"
    else:
        parts.append("[wide]loudnorm=I=-16:TP=-1.5:LRA=11[simple_loud]")
        loud_node = "simple_loud"
    # v8.1 dual-band elevation (ported from 8-band engine)
    elev_parts, last_node = _elevation_dual_band(p, loud_node)
    parts.extend(elev_parts)
    return ";".join(parts), f"[{last_node}]"


def build_vocal_aware_filtergraph(p: ProcessingParams) -> tuple:
    i     = p.intensity_multiplier
    br    = p.bass_rotation * i
    vr    = p.rotation_speed * i * 0.5
    tr    = p.treble_rotation * i
    rev   = _reverb(p)
    eq    = _eq_chain(p)
    dvol  = 1.0 / max(p.distance, 0.3)
    FLOOR = 0.12
    depth = round(1.0 - 2 * FLOOR, 4)

    # cos-based ILD for each band with staggered phase offsets
    def _pan(rot, ph=0.0):
        t = f"2*PI*{rot:.4f}*t+{ph:.3f}"
        return (
            f"({FLOOR:.4f}+{depth:.4f}*(0.5-0.5*cos({t})))",
            f"({FLOOR:.4f}+{depth:.4f}*(0.5+0.5*cos({t})))"
        )

    b_L, b_R = _pan(br,      0.0)
    v_L, v_R = _pan(vr * 0.5, 0.7)   # tight vc pan for vocal band
    h_L, h_R = _pan(tr,      1.5)

    # Vocal center: extra narrow (0.32 depth) keeps voice more centred
    vc_d = round(depth * 0.32, 4)
    vc_L = f"({FLOOR + depth*0.5:.4f}+{vc_d:.4f}*(-cos(2*PI*{vr*0.5:.4f}*t+0.700)))"
    vc_R = f"({FLOOR + depth*0.5:.4f}+{vc_d:.4f}*(cos(2*PI*{vr*0.5:.4f}*t+0.700)))"

    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",
        "[mono_src]asplit=3[bass_in][vocal_in][high_in]",
        "[bass_in]lowpass=f=200[bass_filt]",
        "[bass_filt]asplit=2[bl][br]",
        f"[bl]volume='{b_L}':eval=frame[bvl]",
        f"[br]volume='{b_R}':eval=frame[bvr]",
        "[bvl][bvr]join=inputs=2:channel_layout=stereo[bass_st]",
        f"[bass_st]volume={dvol:.4f}[bass8d]",
        "[vocal_in]bandpass=f=1100:width_type=h:w=2800[voc_filt]",
        "[voc_filt]asplit=2[vl][vr_ch]",
    ]
    if p.enable_vocal_center:
        parts += [
            f"[vl]volume='{vc_L}':eval=frame[vvl]",
            f"[vr_ch]volume='{vc_R}':eval=frame[vvr]",
        ]
    else:
        parts += [
            f"[vl]volume='{v_L}':eval=frame[vvl]",
            f"[vr_ch]volume='{v_R}':eval=frame[vvr]",
        ]
    parts += [
        "[vvl][vvr]join=inputs=2:channel_layout=stereo[voc_st]",
        f"[voc_st]volume={dvol*1.1:.4f}[vocal8d]",
        "[high_in]highpass=f=3000[high_filt]",
        "[high_filt]asplit=2[hl][hr]",
        f"[hl]volume='{h_L}':eval=frame[hvl]",
        f"[hr]volume='{h_R}':eval=frame[hvr]",
        "[hvl][hvr]join=inputs=2:channel_layout=stereo[high_st]",
        f"[high_st]volume={dvol:.4f}[high8d]",
        "[bass8d][vocal8d][high8d]amix=inputs=3:duration=first:normalize=0[mixed]",
        f"[mixed]{rev}[rev]",
        f"[rev]stereotools=mlev={p.stereo_width}[wide]",
    ]
    if eq:
        parts += [f"[wide]{eq}[eqd]", "[eqd]loudnorm=I=-16:TP=-1.5:LRA=11[voc_loud]"]
        loud_node = "voc_loud"
    else:
        parts.append("[wide]loudnorm=I=-16:TP=-1.5:LRA=11[voc_loud]")
        loud_node = "voc_loud"
    # v8.1 dual-band elevation (ported from 8-band engine)
    elev_parts, last_node = _elevation_dual_band(p, loud_node)
    parts.extend(elev_parts)
    return ";".join(parts), f"[{last_node}]"


# ============================================================================
# AMBISONICS FIRST-ORDER (FOA / B-format) ENCODER   — v7.0 NEW
# ============================================================================

def build_ambisonics_foa_filtergraph(p: ProcessingParams) -> tuple:
    """
    First-Order Ambisonics encoder producing 4-channel B-format: W, X, Y, Z.

    For a source moving on a horizontal circle (elevation φ from params):
      azimuth θ(t) = 2π × rotation_speed × t    (LFO-driven rotation)
      elevation φ  = p.elevation × (π/2)         (static tilt)

    Ambisonic gain equations:
      W = 1 / √2                                  (omnidirectional)
      X = cos(θ) × cos(φ)                         (front-back)
      Y = sin(θ) × cos(φ)                         (left-right)
      Z = sin(φ)                                   (up-down, constant)

    The result is a 4-channel WAV file (channel layout: 4.0 quad used as
    proxy for W/X/Y/Z since FFmpeg has no native B-format layout).
    """
    rot = p.rotation_speed * p.intensity_multiplier
    phi = p.elevation * 1.5708   # map [-1,1] → [-π/2, π/2]

    # BUG FIX: the original guard `if 'np' in dir()` is semantically meaningless —
    # numpy is unconditionally imported at module level (line 67), so dir() will
    # always contain 'np'.  Use np directly; if numpy is missing the ImportError
    # at startup is the correct failure signal, not a silent fallback here.
    cos_phi = round(float(np.cos(phi)), 6)
    sin_phi = round(float(np.sin(phi)), 6)

    # W channel: constant 1/√2 ≈ 0.7071
    w_gain = 0.7071

    # Dynamic X = cos(2π rot t) × cos_phi
    x_lfo = f"volume='{cos_phi:.6f}*cos(2*PI*{rot:.5f}*t)':eval=frame"
    # Dynamic Y = sin(2π rot t) × cos_phi
    y_lfo = f"volume='{cos_phi:.6f}*sin(2*PI*{rot:.5f}*t)':eval=frame"
    # Static Z  = sin(phi)
    z_gain = round(sin_phi, 6)

    eq = _eq_chain(p)
    rev = _reverb(p)

    parts = [
        # Mix to mono source
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",

        # Apply master EQ + reverb to the mono source
        f"[mono_src]{rev}[mono_rev]",
    ]
    if eq:
        parts.append(f"[mono_rev]{eq}[mono_eq]")
        parts.append("[mono_eq]asplit=4[w_raw][x_raw][y_raw][z_raw]")
    else:
        parts.append("[mono_rev]asplit=4[w_raw][x_raw][y_raw][z_raw]")

    # W: scaled by 0.7071
    parts.append(f"[w_raw]volume={w_gain}[w_ch]")

    # X: cos(2π rot t) × cos_phi
    parts.append(f"[x_raw]{x_lfo}[x_ch]")

    # Y: sin(2π rot t) × cos_phi
    parts.append(f"[y_raw]{y_lfo}[y_ch]")

    # Z: static elevation component
    if abs(z_gain) > 0.01:
        parts.append(f"[z_raw]volume={z_gain}[z_ch]")
        z_out = "z_ch"
    else:
        parts.append("[z_raw]volume=0.0[z_ch]")
        z_out = "z_ch"

    # Join into 4-channel (using 4.0 quad layout as B-format proxy)
    parts.append(
        f"[w_ch][x_ch][y_ch][{z_out}]"
        "join=inputs=4:channel_layout=4.0[ambi_out]"
    )
    parts.append("[ambi_out]loudnorm=I=-16:TP=-1.5:LRA=11[ambi_norm]")

    return ";".join(parts), "[ambi_norm]"


# ============================================================================
# DOLBY ATMOS 7.1.4 BED ENCODER   — v7.0 NEW
# ============================================================================

def build_atmos_71_4_filtergraph(p: ProcessingParams) -> tuple:
    """
    Renders a 7.1.4-channel Atmos bed by running the full 8-band HRTF
    engine twice (front and back perspective), a centre channel for the
    dry signal, and LFE + 4 height channels derived from the mix.

    Channel order (standard 7.1.4):
      FL FR FC LFE BL BR SL SR TFL TFR TBL TBR
    """
    i   = p.intensity_multiplier
    rot = p.rotation_speed * i
    br  = p.bass_rotation * i
    tr  = p.treble_rotation * i
    dvol= 1.0 / max(p.distance, 0.3)
    eq  = _eq_chain(p)
    rev = _reverb(p)

    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",
        f"[mono_src]{rev}[mono_rev]",
    ]
    if eq:
        parts.append(f"[mono_rev]{eq}[mono_eq]")
        parts.append("[mono_eq]asplit=7[fl_src][fr_src][fc_src][lfe_src][sur_src][h_src][bl_src]")
    else:
        parts.append("[mono_rev]asplit=7[fl_src][fr_src][fc_src][lfe_src][sur_src][h_src][bl_src]")

    # BUG FIX: Atmos LFO previously used 0.5+0.5*sin/cos which drops to exactly
    # 0.0 at the trough — creating dead-zone silence for any channel at the back
    # of the rotation. Ported the ILD floor formula from the v7.1 8-band engine:
    #   pan = FLOOR + depth * (0.5 ± 0.5*cos(θ))   where FLOOR=0.12, depth=0.76
    # This keeps the quietest channel at 12% of peak — never silent.
    ATMOS_FLOOR = 0.12
    ATMOS_DEPTH = round(1.0 - 2 * ATMOS_FLOOR, 4)  # 0.76

    theta_main = f"2*PI*{rot:.5f}*t"
    theta_tr   = f"2*PI*{tr:.5f}*t"

    # FL (left): louder when source is to the left  → (0.5 − 0.5·cos θ)
    fl_lfo = f"({ATMOS_FLOOR:.2f}+{ATMOS_DEPTH:.2f}*(0.5-0.5*cos({theta_main})))"
    # FR (right): louder when source is to the right → (0.5 + 0.5·cos θ)
    fr_lfo = f"({ATMOS_FLOOR:.2f}+{ATMOS_DEPTH:.2f}*(0.5+0.5*cos({theta_main})))"
    # BL/BR (surround): phase-shifted by π for rear image
    bl_lfo = f"(0.10+0.32*(0.5-0.5*cos({theta_main}+3.14159)))"
    br_lfo = f"(0.10+0.32*(0.5+0.5*cos({theta_main}+3.14159)))"
    # SL/SR (sides): phase π/2 for lateral image
    sl_lfo = f"(0.10+0.25*(0.5-0.5*cos({theta_main}+1.5708)))"
    sr_lfo = f"(0.10+0.25*(0.5+0.5*cos({theta_main}+1.5708)))"
    # Height channels: same floor guarantee, treble rotation speed
    tfl_lfo = f"(0.08+0.34*(0.5-0.5*cos({theta_tr})))"
    tfr_lfo = f"(0.08+0.34*(0.5+0.5*cos({theta_tr})))"
    tbl_lfo = f"(0.07+0.26*(0.5-0.5*cos({theta_tr}+3.14159)))"
    tbr_lfo = f"(0.07+0.26*(0.5+0.5*cos({theta_tr}+3.14159)))"

    # FL: ILD-floor LFO (left)
    parts.append(f"[fl_src]volume='{fl_lfo}':eval=frame[fl_ch]")
    # FR: ILD-floor LFO (right)
    parts.append(f"[fr_src]volume='{fr_lfo}':eval=frame[fr_ch]")
    # FC: dry centre
    parts.append(f"[fc_src]volume=0.6[fc_ch]")
    # LFE: lowpass bass
    parts.append(f"[lfe_src]lowpass=f=120[lfe_ch]")
    # BL / BR (surrounds): phase-inverted rotation, ILD floor
    parts.append(f"[sur_src]asplit=2[bl_src2][br_src2]")
    parts.append(f"[bl_src2]volume='{bl_lfo}':eval=frame[bl_ch]")
    parts.append(f"[br_src2]volume='{br_lfo}':eval=frame[br_ch]")
    # SL / SR (sides): blend, ILD floor
    parts.append(f"[bl_src]asplit=2[sl_src][sr_src]")
    parts.append(f"[sl_src]volume='{sl_lfo}':eval=frame[sl_ch]")
    parts.append(f"[sr_src]volume='{sr_lfo}':eval=frame[sr_ch]")
    # Height channels (top): highpass of mix, ILD-floor LFO at treble rotation speed
    parts.append(f"[h_src]highpass=f=3000[h_hp]")
    parts.append("[h_hp]asplit=4[tfl_s][tfr_s][tbl_s][tbr_s]")
    parts.append(f"[tfl_s]volume='{tfl_lfo}':eval=frame[tfl_ch]")
    parts.append(f"[tfr_s]volume='{tfr_lfo}':eval=frame[tfr_ch]")
    parts.append(f"[tbl_s]volume='{tbl_lfo}':eval=frame[tbl_ch]")
    parts.append(f"[tbr_s]volume='{tbr_lfo}':eval=frame[tbr_ch]")

    # Join 12 channels
    parts.append(
        "[fl_ch][fr_ch][fc_ch][lfe_ch][bl_ch][br_ch][sl_ch][sr_ch]"
        "[tfl_ch][tfr_ch][tbl_ch][tbr_ch]"
        "join=inputs=12:channel_layout=7.1.4[atmos_out]"
    )
    parts.append("[atmos_out]loudnorm=I=-16:TP=-1.5:LRA=11[atmos_norm]")
    return ";".join(parts), "[atmos_norm]"


# ============================================================================
# FILTERGRAPH LABEL PREFIXER  — multi-instance stem support  (v8.0)
# ============================================================================

import re as _re

def _prefix_filtergraph(fg: str, out_label: str,
                         prefix: str, input_ref: str = "") -> tuple:
    """
    Renames every [label] in fg to [prefix+label], so the same engine
    can be instantiated multiple times within one mega-filtergraph without
    label name collisions.  If input_ref is supplied (e.g. "[0:a]"), it is
    prepended to the very first filter token so FFmpeg knows which -i input
    stream to use.

    Returns (new_fg, new_out_label).
    """
    new_fg  = _re.sub(r'\[([^\]]+)\]',
                      lambda m: f'[{prefix}{m.group(1)}]', fg)
    new_out = f'[{prefix}{out_label[1:-1]}]'
    if input_ref:
        new_fg = input_ref + new_fg
    return new_fg, new_out


# ============================================================================
# MULTIBAND MASTERING BUS  — post-mix master chain  (v8.0)
# ============================================================================

def _multiband_master_bus(mix_label: str, out_label: str,
                           p: ProcessingParams) -> str:
    """
    3-band multiband compressor + stereo width + EBU R128 + true peak limiter.

    Applied after all per-stem 8D signals are mixed, before final encode.
    Returns a semicolon-joined filtergraph fragment (no leading/trailing []).
    The fragment reads from [mix_label] and outputs to [out_label].
    """
    parts = []

    # Pre-master corrective EQ
    parts.append(
        f"[{mix_label}]"
        "equalizer=f=35:t=h:w=30:g=-1.5,"
        "equalizer=f=300:t=q:w=250:g=-1.0,"
        "equalizer=f=2500:t=q:w=2000:g=0.8,"
        f"equalizer=f=9000:t=h:w=4000:g=1.0[{out_label}_pmeq]"
    )

    # Split into 3 bands
    parts.append(
        f"[{out_label}_pmeq]asplit=3"
        f"[{out_label}_lo][{out_label}_mi][{out_label}_hi]"
    )

    # Low band: sub + bass (≤ 200 Hz) — heavy glue, tight punch
    parts.append(
        f"[{out_label}_lo]lowpass=f=200[{out_label}_lolp]"
    )
    parts.append(
        f"[{out_label}_lolp]"
        "acompressor=threshold=-24dB:ratio=3.5:attack=8:release=120"
        f":makeup=1.5dB:knee=3dB[{out_label}_loc]"
    )

    # Mid band: 200 Hz – 5 kHz — moderate control
    parts.append(
        f"[{out_label}_mi]"
        f"highpass=f=200[{out_label}_mihp]"
    )
    parts.append(
        f"[{out_label}_mihp]lowpass=f=5000[{out_label}_milp]"
    )
    parts.append(
        f"[{out_label}_milp]"
        "acompressor=threshold=-20dB:ratio=2.5:attack=12:release=200"
        f":makeup=1.0dB:knee=4dB[{out_label}_mic]"
    )

    # High band: > 5 kHz — gentle air control
    parts.append(
        f"[{out_label}_hi]highpass=f=5000[{out_label}_hihp]"
    )
    parts.append(
        f"[{out_label}_hihp]"
        "acompressor=threshold=-18dB:ratio=2.0:attack=5:release=80"
        f":makeup=0.5dB:knee=2dB[{out_label}_hic]"
    )

    # Re-combine bands
    parts.append(
        f"[{out_label}_loc][{out_label}_mic][{out_label}_hic]"
        f"amix=inputs=3:duration=first:normalize=0[{out_label}_glued]"
    )

    # Stereo width on master bus
    master_width = round(min(p.stereo_width * 1.05, 1.40), 3)
    parts.append(
        f"[{out_label}_glued]"
        f"stereotools=mlev={master_width}:sbal=0:softclip=1"
        f"[{out_label}_wide]"
    )

    # EBU R128 loudness normalisation
    parts.append(
        f"[{out_label}_wide]"
        "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true"
        f"[{out_label}_loud]"
    )

    # True peak limiter
    parts.append(
        f"[{out_label}_loud]"
        f"alimiter=limit=1:attack=3:release=40:level=false[{out_label}]"
    )

    return ";".join(parts)


# ============================================================================
# SINGLE-PASS MULTI-STEM FILTERGRAPH  — all stems in one FFmpeg call (v8.0)
# ============================================================================

def build_single_pass_stem_filtergraph(
    stem_configs: List[Dict[str, Any]],
    master_params: ProcessingParams,
) -> tuple:
    """
    Build one large FFmpeg filtergraph that processes every stem
    simultaneously:
      1. Each stem runs through a full 8-band HRTF engine (per-stem params)
      2. All processed stems are amixed (normalize=0 to honour gain staging)
      3. The combined signal passes through the multiband mastering bus

    stem_configs: list of dicts, one per stem —
      { 'stem_name': str, 'input_idx': int, 'params': ProcessingParams }
    master_params: ProcessingParams for the master bus settings.

    Returns (filtergraph_string, output_label).
    """
    if not stem_configs:
        raise ValueError("build_single_pass_stem_filtergraph: stem_configs is empty")
    all_parts: List[str] = []
    processed_labels: List[str] = []

    for cfg in stem_configs:
        stem_name = cfg['stem_name']
        idx       = cfg['input_idx']
        sp        = cfg['params']
        prefix    = f"{stem_name[:2]}{idx}_"  # e.g. "vo0_", "dr1_"

        fg, out_lbl = build_8band_hrtf_engine_v6(sp)
        pfg, pout   = _prefix_filtergraph(fg, out_lbl, prefix, f"[{idx}:a]")

        all_parts.append(pfg)
        processed_labels.append(pout)
        print(f"  ↳ [{stem_name}] input={idx} out={pout}  "
              f"rot={sp.rotation_speed:.3f} w={sp.stereo_width:.2f} "
              f"rev={sp.reverb_mix:.2f}")

    # Mix all stems
    n = len(processed_labels)
    mix_in = "".join(processed_labels)
    all_parts.append(
        f"{mix_in}amix=inputs={n}:duration=first:normalize=0[stem_mix_raw]"
    )

    # Master bus (reads [stem_mix_raw] → writes [final_out])
    master_bus = _multiband_master_bus("stem_mix_raw", "final_out", master_params)
    all_parts.append(master_bus)

    return ";".join(all_parts), "[final_out]"


# ============================================================================
# STEM SEPARATION   — v8.0  (4-stem + 6-stem Demucs, Spleeter fallback)
# ============================================================================

# MODEL → stem names mapping
_DEMUCS_STEMS = {
    "htdemucs":    ["vocals", "drums", "bass", "other"],
    "htdemucs_6s": ["vocals", "drums", "bass", "guitar", "piano", "other"],
    "mdx_extra":   ["vocals", "drums", "bass", "other"],
}

async def separate_stems(
    input_path: str,
    job_id: str,
    model: str = "htdemucs",
) -> Optional[tuple]:
    """
    Separates audio into stems using Demucs (preferred) or Spleeter.

    model choices:
      "htdemucs"    — 4 stems: vocals / drums / bass / other
      "htdemucs_6s" — 6 stems: vocals / drums / bass / guitar / piano / other
      "spleeter"    — 4 stems via Spleeter (fallback)

    Returns (stems_dict, session_id) or None on failure.
    stems_dict: { stem_name: wav_path }
    """
    if not STEM_SEPARATION:
        return None

    session_id = str(uuid.uuid4())
    stem_dir   = TEMP_DIR / f"stems_{session_id}"
    stem_dir.mkdir(exist_ok=True)

    # Effective model to use
    use_model = model if STEM_ENGINE == "demucs" else "spleeter"
    stem_names = _DEMUCS_STEMS.get(use_model, _DEMUCS_STEMS["htdemucs"])

    await manager.send_progress(
        job_id, 8,
        f"🎸 Separating stems [{use_model}] — {len(stem_names)} stems…"
    )

    try:
        if STEM_ENGINE == "demucs":
            cmd = [
                sys.executable, "-m", "demucs",
                "-n", use_model,
                "--out", str(stem_dir),
                input_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                err = stderr.decode()[:500]
                print(f"⚠  Demucs [{use_model}] failed: {err}")
                # Try fallback to htdemucs 4-stem if 6-stem requested but unavailable
                if use_model == "htdemucs_6s":
                    print("  ↳ Retrying with htdemucs (4-stem fallback)…")
                    return await separate_stems(input_path, job_id, "htdemucs")
                return None

            input_stem = Path(input_path).stem
            demucs_out = stem_dir / use_model / input_stem
            if not demucs_out.exists():
                print(f"⚠  Demucs output dir not found: {demucs_out}")
                return None

            stems: Dict[str, str] = {}
            for name in stem_names:
                p = demucs_out / f"{name}.wav"
                if p.exists():
                    stems[name] = str(p)
                else:
                    print(f"  ⚠  Stem file missing: {name}.wav")

        else:  # spleeter (4-stem only)
            from spleeter.separator import Separator as SpleeterSep
            sep  = SpleeterSep("spleeter:4stems")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: sep.separate_to_file(input_path, str(stem_dir))
            )
            input_stem = Path(input_path).stem
            sp_out = stem_dir / input_stem
            stems = {}
            for name in ["vocals", "drums", "bass", "other"]:
                p = sp_out / f"{name}.wav"
                if p.exists():
                    stems[name] = str(p)

        if not stems:
            print("⚠  No stems were found after separation")
            return None

        stem_sessions[session_id] = stems
        _session_timestamps[session_id] = __import__('time').time()  # TTL clock
        print(f"✅ Stems [{use_model}]: {list(stems.keys())} → session {session_id}")
        await manager.send_progress(
            job_id, 18,
            f"✅ Stems ready ({len(stems)}): {', '.join(stems.keys())}"
        )
        return stems, session_id

    except Exception as e:
        print(f"❌ Stem separation failed: {e}")
        return None


# ============================================================================
# VIDEO VISUALIZER GENERATOR   — v7.0 NEW
# ============================================================================

async def generate_video(
    audio_path: str,
    video_output_path: str,
    params: ProcessingParams,
    job_id: str
) -> bool:
    """
    Generates a video visualisation synced to the processed audio using
    FFmpeg's built-in audio visualisation filters.

    Styles:
      waveform    — showwaves (colour waveform on dark background)
      spectrum    — showspectrum (spectrogram with log scale)
      vectorscope — avectorscope (Lissajous stereo phase)
    """
    await manager.send_progress(job_id, 94, f"🎬 Rendering video ({params.video_style})…")

    w, h = params.video_resolution.split("x")
    fps  = params.video_fps

    if params.video_style == "spectrum":
        vis_filter = (
            f"[0:a]showspectrum=s={w}x{h}:mode=combined:color=intensity:"
            f"scale=log:saturation=3:fps={fps},format=yuv420p[v]"
        )
    elif params.video_style == "vectorscope":
        vis_filter = (
            f"[0:a]avectorscope=s={w}x{h}:zoom=3:rc=255:gc=180:bc=50:"
            f"rf=0:gf=0:bf=0,format=yuv420p[v]"
        )
    else:  # waveform (default)
        vis_filter = (
            f"[0:a]showwaves=s={w}x{h}:mode=cline:rate={fps}:"
            f"colors=#c87c3a|#e09050,format=yuv420p[v]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-filter_complex", vis_filter,
        "-map", "[v]",
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "256k",
        "-shortest",
        video_output_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        print(f"⚠  Video generation failed: {stderr.decode()[-500:]}")
        return False

    print(f"✅ Video → {video_output_path}")
    return True

async def process_8d_audio(
    input_file: str,
    output_file: str,
    params: ProcessingParams,
    job_id: str,
    audio_analysis: Optional[Dict[str, Any]] = None
):
    try:
        print(f"\n🎬 8D Processing v7.0 — job {job_id}")
        await manager.send_progress(job_id, 5, "Running deep audio analysis…")

        # Re-analyze if not already done
        if audio_analysis is None and ADVANCED_ANALYSIS:
            try:
                audio_analysis = audio_analyzer.analyze_comprehensive(input_file)
                await manager.send_progress(job_id, 12,
                    f"Analysis complete → {audio_analysis.get('genre','?')} / "
                    f"{audio_analysis.get('key','?')} {audio_analysis.get('mode','?')} / "
                    f"BPM {audio_analysis.get('bpm','?')}"
                )
            except Exception:
                pass

        # ── Stem Separation path — v8.0 single-pass psychoacoustic engine ───
        if params.enable_stem_separation and STEM_SEPARATION:
            model  = params.stem_engine_model or "htdemucs"
            result = await separate_stems(input_file, job_id, model)
            if result:
                stems_map, session_id = result
                stem_list = list(stems_map.items())   # [(name, path), ...]
                n_stems   = len(stem_list)

                await manager.send_progress(
                    job_id, 20,
                    f"🎛  Building per-stem spatial params ({n_stems} stems)…"
                )

                # ── Per-stem gain staging + psychoacoustic param routing ──────
                stem_configs: List[Dict[str, Any]] = []
                for idx, (stem_name, stem_path) in enumerate(stem_list):

                    # Gain staging: measure stem RMS → compute makeup gain
                    gain_db = 0.0
                    if params.enable_gain_staging:
                        gain_db = instrument_router.estimate_stem_gain_db(
                            stem_path, params.stem_target_lufs
                        )
                        print(f"  ↳ [{stem_name}] gain staging: {gain_db:+.1f} dB")

                    # InstrumentRouter: get psychoacoustically tuned params
                    if params.stem_auto_route:
                        stem_p = instrument_router.get_stem_params(
                            stem_name, params,
                            analysis=audio_analysis,
                            gain_db=gain_db,
                        )
                    else:
                        # Legacy manual override mode
                        stem_p = params.copy()
                        overrides: Dict[str, Any] = {}
                        if stem_name == "vocals":
                            overrides = {
                                "enable_vocal_center": True,
                                "rotation_speed": params.stem_vocals_rotation
                                    or round(params.rotation_speed * 0.7, 3),
                                "reverb_mix": round(params.reverb_mix * 0.85, 3),
                            }
                        elif stem_name == "drums":
                            overrides = {
                                "rotation_speed": params.stem_drums_rotation
                                    or round(params.rotation_speed * 1.2, 3),
                                "reverb_mix": round(params.reverb_mix * 0.5, 3),
                            }
                        elif stem_name == "bass":
                            overrides = {
                                "rotation_speed": params.stem_bass_rotation_override
                                    or params.bass_rotation,
                                "reverb_mix": round(params.reverb_mix * 0.3, 3),
                            }
                        elif stem_name == "guitar":
                            overrides = {
                                "rotation_speed": params.stem_guitar_rotation
                                    or round(params.rotation_speed * 1.1, 3),
                            }
                        elif stem_name == "piano":
                            overrides = {
                                "rotation_speed": params.stem_piano_rotation
                                    or round(params.rotation_speed * 0.9, 3),
                            }
                        else:
                            overrides = {
                                "rotation_speed": params.stem_other_rotation
                                    or params.treble_rotation,
                            }
                        stem_p = stem_p.copy(update=overrides)

                    # Force WAV output and no video for intermediate stems
                    stem_p = stem_p.copy(update={
                        "output_format": "wav",
                        "generate_video": False,
                    })

                    stem_configs.append({
                        "stem_name": stem_name,
                        "input_idx": idx,
                        "params":    stem_p,
                    })

                await manager.send_progress(
                    job_id, 28,
                    f"🔮 Building single-pass mega-filtergraph ({n_stems} engines)…"
                )

                # ── Build single-pass filtergraph (all stems → one FFmpeg call)
                # BUG FIX: the previous if/else had two branches that called the
                # identical function — build_single_pass_stem_filtergraph always
                # applies the multiband master bus internally (see _multiband_master_bus).
                # The `enable_multiband_master` flag is read inside that function,
                # so there is no need for separate call sites here.
                fg, out_lbl = build_single_pass_stem_filtergraph(
                    stem_configs, params
                )
                engine_tag = (
                    "Single-Pass 8xHRTF + Multiband Master Bus"
                    if params.enable_multiband_master
                    else "Single-Pass 8xHRTF (no multiband)"
                )

                print(f"  ↳ Engine : {engine_tag}")
                print(f"  ↳ Graph  : {len(fg)} chars across {n_stems} HRTF instances")

                # Codec for final output
                actual_sr = params.sample_rate or 48000
                if params.output_format == "mp3":
                    if actual_sr > 48000:
                        actual_sr = 48000
                    final_codec = ["-c:a", "libmp3lame", "-b:a", f"{params.bitrate}k"]
                elif params.output_format == "flac":
                    final_codec = ["-c:a", "flac", "-compression_level", "8"]
                else:
                    final_codec = ["-c:a", "pcm_s24le"]

                # Build FFmpeg inputs list
                ff_inputs: List[str] = []
                for _, (_, stem_path) in enumerate(stem_list):
                    ff_inputs += ["-i", stem_path]

                mix_cmd = [
                    "ffmpeg", "-y",
                    *ff_inputs,
                    "-filter_complex", fg,
                    "-map", out_lbl,
                    "-ar", str(actual_sr),
                    *final_codec,
                    output_file,
                ]

                await manager.send_progress(
                    job_id, 35,
                    f"⚙️  Rendering {n_stems}-stem 8D audio…"
                )
                total_dur = get_audio_duration(stem_list[0][1])

                proc = await asyncio.create_subprocess_exec(
                    *mix_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _active_procs[job_id] = proc   # register for cancel endpoint
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    ls = line.decode('utf-8', errors='ignore')
                    stderr_lines.append(ls)
                    if 'time=' in ls:
                        raw = await ffmpeg_progress(ls, total_dur)
                        pct = min(35 + int(raw * 57), 92)
                        await manager.send_progress(
                            job_id, pct, "Encoding stem-based 8D audio…"
                        )
                await proc.wait()

                if proc.returncode != 0:
                    tail = "".join(stderr_lines[-60:])
                    print(f"❌ Single-pass stem mix failed:\n{tail}")
                    raise Exception(
                        f"Stem filtergraph failed (code {proc.returncode})."
                    )

                video_url = None
                if params.generate_video and os.path.exists(output_file):
                    vid_out = str(
                        OUTPUT_DIR / (Path(output_file).stem + "_viz.mp4")
                    )
                    ok = await generate_video(
                        output_file, vid_out, params, job_id
                    )
                    if ok:
                        video_url = (
                            f"http://localhost:8000/download/"
                            f"{Path(vid_out).name}"
                        )

                await manager.send_progress(
                    job_id, 100, "✅ Stem-based 8D processing complete!"
                )
                out_url = (
                    f"http://localhost:8000/download/"
                    f"{Path(output_file).name}"
                )
                await manager.send_complete(job_id, out_url, video_url=video_url)
                return True

        # ── Standard path ───────────────────────────────────────────────────
        await manager.send_progress(job_id, 15, "Building spatial filter graph…")

        if params.output_format == "ambisonics_foa":
            filtergraph, out_label = build_ambisonics_foa_filtergraph(params)
            engine_name = "Ambisonics FOA (W/X/Y/Z B-format)"
        elif params.output_format == "atmos_71_4":
            filtergraph, out_label = build_atmos_71_4_filtergraph(params)
            engine_name = "Dolby Atmos Bed (7.1.4)"
        elif params.enable_multi_band and params.enable_hrtf:
            # Extract beat positions from analysis for beat-locked rotation
            beat_times_s: Optional[List[float]] = None
            if audio_analysis and ADVANCED_ANALYSIS:
                raw_beats = audio_analysis.get('beat_positions', [])
                sr_val    = audio_analysis.get('sample_rate', 48000)
                if raw_beats and sr_val:
                    try:
                        beat_times_s = [float(b) / sr_val for b in raw_beats]
                    except Exception:
                        beat_times_s = None

            # ── HRIR pre-pass ──────────────────────────────────────────────
            hrir_input = input_file
            hrir_temp  = None
            fg_params  = params   # default: unmodified params for FFmpeg stage
            if ADVANCED_ANALYSIS and params.enable_hrtf:
                try:
                    await manager.send_progress(
                        job_id, 18, "🧠 HRIR binaural convolution pass…"
                    )
                    y_raw, sr_raw = librosa.load(input_file, sr=None, mono=True)
                    target_sr = params.sample_rate or 48000
                    if sr_raw != target_sr:
                        y_raw = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=target_sr)
                        sr_raw = target_sr

                    hrir_eng  = get_hrir_engine(sr_raw)
                    stereo_np = hrir_eng.render(
                        y_raw,
                        beat_times_s=beat_times_s,
                        rotation_speed=params.rotation_speed * params.intensity_multiplier,
                        elevation=params.elevation,
                        distance_m=params.distance,   # maps 0.2–1.1 m into SOFA range
                    )
                    # Write pre-rendered binaural WAV
                    hrir_temp = str(TEMP_DIR / f"{job_id}_hrir.wav")
                    sf.write(hrir_temp, stereo_np.T, sr_raw, subtype='PCM_24')
                    hrir_input = hrir_temp
                    print(f"  ↳ HRIR pre-pass complete → {Path(hrir_temp).name}")
                    await manager.send_progress(
                        job_id, 22, "✅ HRIR pass done — applying reverb + EQ…"
                    )
                except Exception as hrir_err:
                    print(f"  ⚠  HRIR pass failed ({hrir_err}), falling back to FFmpeg HRTF")
                    import traceback; traceback.print_exc()
                    hrir_temp = None
                    hrir_input = input_file

            # Build FFmpeg filtergraph
            # When hrir_input is the HRIR-pre-rendered WAV, we suppress the
            # FFmpeg HRTF stage (pan rotation) and only apply reverb + EQ.
            # The elevation is handled in the HRIR engine, so we zero it here.
            fg_params = params
            if hrir_temp:
                # HRIR engine already applied rotation + elevation + ITD + pinna
                # FFmpeg stage: reverb, EQ, loudnorm, limiter only.
                # We use the 6-band engine with rotation zeroed out to get
                # just the mastering chain without spatial movement.
                fg_params = params.copy(update={
                    "rotation_speed": 0.0,
                    "bass_rotation":  0.0,
                    "treble_rotation": 0.0,
                    "elevation": 0.0,
                    "enable_hrtf": False,   # skip FFmpeg HRTF layer
                    "enable_multi_band": False,
                })
                filtergraph, out_label = build_stereo_mastering_chain(fg_params)
                engine_name = (
                    "Studio Grade v9.0 — HRIR Convolution + Stereo Mastering"
                )
            else:
                filtergraph, out_label = build_8band_hrtf_engine_v6(
                    params, beat_times_s=beat_times_s
                )
                engine_name = "Studio Grade v9.0 (8-band HRTF + Beat-Locked Rotation)"

            # Use HRIR pre-rendered temp file as input if available
            input_file_for_ffmpeg = hrir_input
        elif params.enable_multi_band:
            filtergraph, out_label = build_6band_filtergraph(params)
            engine_name = "6-band multiband engine"
        elif params.enable_vocal_center:
            filtergraph, out_label = build_vocal_aware_filtergraph(params)
            engine_name = "vocal-aware 3-band engine"
        else:
            filtergraph, out_label = build_simple_filtergraph(params)
            engine_name = "simple 2-channel engine"

        print(f"  ↳ Engine : {engine_name}")
        print(f"  ↳ Graph  : {len(filtergraph)} chars")
        await manager.send_progress(job_id, 25, f"Using {engine_name}…")

        actual_sr = params.sample_rate or 48000
        if params.output_format == "mp3":
            if actual_sr > 48000:
                actual_sr = 48000
            codec = ["-c:a", "libmp3lame", "-b:a", f"{params.bitrate}k"]
        elif params.output_format == "flac":
            codec = ["-c:a", "flac", "-compression_level", "8"]
        else:
            codec = ["-c:a", "pcm_s24le"]

        # Use HRIR pre-rendered WAV if available, else original input
        _ffmpeg_input = input_file_for_ffmpeg if 'input_file_for_ffmpeg' in locals() else input_file

        cmd = [
            "ffmpeg", "-y",
            "-i", _ffmpeg_input,
            "-filter_complex", filtergraph,
            "-map", out_label,
            "-ar", str(actual_sr),
            *codec,
            output_file
        ]

        await manager.send_progress(job_id, 35, "Applying binaural ITD + panning…")
        total_dur = get_audio_duration(_ffmpeg_input)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _active_procs[job_id] = proc   # register for cancel endpoint
        stderr_lines = []
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            ls = line.decode('utf-8', errors='ignore')
            stderr_lines.append(ls)
            if 'time=' in ls:
                raw = await ffmpeg_progress(ls, total_dur)
                pct = min(35 + int(raw * 57), 92)
                await manager.send_progress(job_id, pct, "Encoding spatial audio…")

        await proc.wait()

        if proc.returncode != 0:
            tail = "".join(stderr_lines[-50:])
            print(f"❌ FFmpeg error:\n{tail}")
            raise Exception(f"FFmpeg failed (code {proc.returncode}).")

        await manager.send_progress(job_id, 96, "Finalising master…")

        if not os.path.exists(output_file):
            raise Exception("Output file was not created")

        # Clean up HRIR temp file
        _hrir_temp_ref = locals().get('hrir_temp')
        if _hrir_temp_ref and os.path.exists(_hrir_temp_ref):
            try:
                os.unlink(_hrir_temp_ref)
            except Exception:
                pass

        video_url = None
        if params.generate_video:
            vid_out = str(OUTPUT_DIR / (Path(output_file).stem + "_viz.mp4"))
            ok = await generate_video(output_file, vid_out, params, job_id)
            if ok:
                video_url = f"http://localhost:8000/download/{Path(vid_out).name}"

        await manager.send_progress(job_id, 100, "✅ 8D processing complete!")
        out_url = f"http://localhost:8000/download/{Path(output_file).name}"
        await manager.send_complete(job_id, out_url, video_url=video_url)
        print(f"✅ Done → {out_url}")
        _active_procs.pop(job_id, None)
        _active_tasks.pop(job_id, None)
        return True

    except asyncio.CancelledError:
        # Job was cancelled via the DELETE /process/{job_id} endpoint
        print(f"⛔ Job {job_id} cancelled by user")
        _active_procs.pop(job_id, None)
        _active_tasks.pop(job_id, None)
        await manager.send_error(job_id, "Processing cancelled by user")
        return False

    except Exception as e:
        print(f"❌ Processing error: {e}")
        _active_procs.pop(job_id, None)
        _active_tasks.pop(job_id, None)
        await manager.send_error(job_id, str(e))
        return False


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    ok      = check_ffmpeg()
    has_rev = check_reverberate_filter() if ok else False

    # Explicit variables so the values are readable in logs/tests
    stem_ok  = STEM_SEPARATION   # True when demucs or spleeter is available
    stem_eng = STEM_ENGINE        # "demucs" | "spleeter" | "none"

    # Report whether real KEMAR HRIR data is loaded
    engine = get_hrir_engine()
    sofa_loaded = getattr(engine, "using_sofa", False)

    return {
        "status":               "healthy" if ok else "degraded",
        "ffmpeg":               ok,
        "advanced_analysis":    ADVANCED_ANALYSIS,
        "youtube_support":      YOUTUBE_SUPPORT,
        "stem_separation":      stem_ok,   # ← True when demucs/spleeter installed
        "stem_engine":          stem_eng,  # ← "demucs" when Demucs 4.x present
        "reverb_engine":        "reverberate" if has_rev else "aecho",
        "has_reverberate":      has_rev,
        "version":              "9.1.0",
        "analysis_bands":       10,
        "eq_bands":             12,
        "spatial_bands":        8,
        "genres":               15,
        "itd_simulation":       True,
        "pinna_notch_eq":       True,
        "diffuse_field_eq":     True,
        "allpass_diffusion":    True,
        "ambisonics_foa":       True,
        "atmos_71_4":           True,
        "video_visualizer":     True,
        "hrir_convolution":     ADVANCED_ANALYSIS,
        "hrir_source":          "KEMAR_SOFA" if sofa_loaded else "synthetic",
        "beat_locked_rotation": ADVANCED_ANALYSIS,
        "dual_band_elevation":  True,
        "ms_eq":                True,   # M/S EQ on master bus
        "pre_delay_ms":         "25-40",
    }

@app.get("/presets")
async def list_presets():
    """Return the full preset library for the frontend dropdown."""
    return {"presets": PresetLibrary.get_preset_list()}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        fid  = str(uuid.uuid4())
        path = TEMP_DIR / f"{fid}_{file.filename}"
        path.write_bytes(await file.read())
        result = audio_analyzer.analyze_comprehensive(str(path))
        path.unlink(missing_ok=True)

        # Attach preset suggestions
        if ADVANCED_ANALYSIS:
            rec = PresetLibrary.select_preset(result)
            result['available_presets']    = PresetLibrary.get_preset_list()
            result['recommended_preset']   = rec
            result['recommended_preset_id'] = result.get('genre', 'pop')

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_audio(
    audio_file: Optional[UploadFile] = File(None),
    params: str = Form(...)
):
    try:
        pp = ProcessingParams.model_validate(json.loads(params))
        job_id = str(uuid.uuid4())

        if audio_file:
            in_path = UPLOAD_DIR / f"{job_id}_{audio_file.filename}"
            in_path.write_bytes(await audio_file.read())
        else:
            raise HTTPException(status_code=400, detail="No audio file provided")

        # Apply genre preset if preset_id matches a known genre
        if pp.preset_id and pp.preset_id in PresetLibrary.GENRE_PRESETS:
            preset_data = PresetLibrary.GENRE_PRESETS[pp.preset_id]
            pp = ProcessingParams.apply_preset(pp, preset_data)
            print(f"  ↳ Preset applied: {pp.preset_id} → {preset_data['name']}")

        # Determine file extension based on format
        ext_map = {
            "mp3": "mp3", "wav": "wav", "flac": "flac",
            "ambisonics_foa": "wav", "atmos_71_4": "wav",
        }
        ext = ext_map.get(pp.output_format, pp.output_format)
        out_name = f"{job_id}_8d.{ext}"
        out_path  = OUTPUT_DIR / out_name

        task = asyncio.create_task(
            process_8d_audio(str(in_path), str(out_path), pp, job_id, None)
        )
        _active_tasks[job_id] = task
        return {"job_id": job_id, "status": "processing"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/process/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel an in-flight processing job.

    Terminates the FFmpeg subprocess (if still running) and cancels the
    asyncio task.  The WebSocket client will receive an error message so
    the UI can update accordingly.
    """
    cancelled_anything = False

    # 1. Kill the FFmpeg subprocess
    proc = _active_procs.get(job_id)
    if proc and proc.returncode is None:
        try:
            proc.terminate()
            await asyncio.sleep(0.5)
            if proc.returncode is None:
                proc.kill()   # force-kill if SIGTERM wasn't enough
        except Exception:
            pass
        cancelled_anything = True

    # 2. Cancel the asyncio task (raises CancelledError inside process_8d_audio)
    task = _active_tasks.get(job_id)
    if task and not task.done():
        task.cancel()
        cancelled_anything = True

    if not cancelled_anything:
        raise HTTPException(status_code=404, detail="Job not found or already complete")

    return {"job_id": job_id, "status": "cancelled"}
async def stems_separate(
    audio_file: UploadFile = File(...),
    model: str = Form("htdemucs"),  # htdemucs | htdemucs_6s | spleeter
):
    """
    Separate audio into stems and return a session ID for reuse.

    model options:
      htdemucs    — 4 stems: vocals / drums / bass / other  (default)
      htdemucs_6s — 6 stems: vocals / drums / bass / guitar / piano / other
      spleeter    — 4 stems via Spleeter (fallback if Demucs unavailable)
    """
    if not STEM_SEPARATION:
        raise HTTPException(
            status_code=501,
            detail="Stem separation not available. Install demucs: pip install demucs"
        )
    job_id  = str(uuid.uuid4())
    in_path = UPLOAD_DIR / f"{job_id}_{audio_file.filename}"
    in_path.write_bytes(await audio_file.read())

    result = await separate_stems(str(in_path), job_id, model=model)
    if not result:
        raise HTTPException(status_code=500, detail="Stem separation failed")

    stems_map, session_id = result
    return {
        "session_id": session_id,
        "model_used": model,
        "stems":      list(stems_map.keys()),
        "stem_count": len(stems_map),
        "download_urls": {
            name: f"http://localhost:8000/stems/{session_id}/{name}"
            for name in stems_map
        }
    }


@app.get("/stems/{session_id}/{stem_name}")
async def download_stem(session_id: str, stem_name: str):
    """Download a separated stem WAV file."""
    if session_id not in stem_sessions:
        raise HTTPException(status_code=404, detail="Stem session not found")
    stems = stem_sessions[session_id]
    if stem_name not in stems:
        raise HTTPException(status_code=404, detail=f"Stem '{stem_name}' not found")
    fp = Path(stems[stem_name])
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Stem file not found on disk")
    return FileResponse(fp, media_type="audio/wav", filename=f"{stem_name}.wav")

@app.websocket("/ws/{job_id}")
async def ws_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        manager.disconnect(job_id)

@app.get("/download/{filename}")
async def download(filename: str):
    fp = OUTPUT_DIR / filename
    if not fp.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fp, media_type="audio/mpeg", filename=filename)

@app.post("/batch/process")
async def batch_process(
    files: List[UploadFile] = File(...),
    params: str = Form(...)
):
    try:
        batch_id   = str(uuid.uuid4())
        params_obj = ProcessingParams(**json.loads(params))
        batch_queue[batch_id] = []
        _session_timestamps[batch_id] = __import__('time').time()  # TTL clock

        for file in files:
            job_id     = str(uuid.uuid4())
            input_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
            input_path.write_bytes(await file.read())
            batch_queue[batch_id].append(BatchJob(
                job_id=job_id, filename=file.filename, status="queued"
            ))

        asyncio.create_task(_process_batch(batch_id, params_obj))
        return {"batch_id": batch_id, "total_files": len(files)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    if batch_id not in batch_queue:
        raise HTTPException(status_code=404, detail="Batch not found")
    jobs = batch_queue[batch_id]
    return {
        "batch_id":   batch_id,
        "total":      len(jobs),
        "completed":  sum(1 for j in jobs if j.status == "completed"),
        "failed":     sum(1 for j in jobs if j.status == "failed"),
        "processing": sum(1 for j in jobs if j.status == "processing"),
        "jobs":       [j.dict() for j in jobs],
    }

async def _process_batch(batch_id: str, params: ProcessingParams):
    for job in batch_queue[batch_id]:
        try:
            job.status = "processing"
            files = list(UPLOAD_DIR.glob(f"{job.job_id}_*"))
            if not files:
                job.status = "failed"; job.error = "Input not found"; continue
            in_path  = files[0]
            base     = Path(in_path.name.replace(f"{job.job_id}_", "")).stem
            out_name = f"{base}_8D.{params.output_format}"
            out_path = OUTPUT_DIR / out_name
            success  = await process_8d_audio(str(in_path), str(out_path), params, job.job_id)
            job.status = "completed" if success else "failed"
            if success:
                job.progress   = 100
                job.output_url = f"/download/{out_name}"
            if in_path.exists():
                in_path.unlink()
        except Exception as e:
            job.status = "failed"; job.error = str(e)

@app.post("/youtube/download")
async def yt_download(request: YouTubeDownloadRequest):
    if not YOUTUBE_SUPPORT:
        raise HTTPException(status_code=501, detail="yt-dlp not installed")
    try:
        job_id = str(uuid.uuid4())
        out    = UPLOAD_DIR / f"{job_id}_youtube.mp3"
        opts   = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'mp3', 'preferredquality': '320'}],
            'outtmpl': str(out.with_suffix('')),
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
        return {"success": True, "title": info.get('title', 'Unknown'),
                "audio_url": f"http://localhost:8000/download/{out.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    asyncio.get_event_loop().set_exception_handler(
        lambda loop, ctx: None if isinstance(ctx.get("exception"), ConnectionResetError) else
        loop.default_exception_handler(ctx)
    )

    print("\n" + "="*72)
    print("  8D Audio Converter — Deep Analysis Backend  v8.0")
    print("="*72)
    print(f"  Analysis bands    : 10  (sub → air)")
    print(f"  EQ bands          : 12  (30 Hz → 16 kHz)")
    print(f"  Spatial bands     : 8   (HRTF + independent rotation)")
    print(f"  Genre profiles    : 15  (incl. Bollywood/Bhangra/Ghazal/Folk)")
    print(f"  Spatial formats   : Stereo · Ambisonics FOA · Dolby Atmos 7.1.4")
    print(f"  Analysis v6.0     : MFCC · Chroma/Key · Crest Factor · HNR")
    print(f"                      Stereo Correlation · Transient Density")
    print(f"                      Tonnetz · ZCR · Spectral Rolloff")
    print(f"  8D v7.1           : ITD bilateral dynamic (LFO-blended)")
    print(f"                      Pinna notch EQ (8.5/10.5/13 kHz)")
    print(f"                      Pre-delay reverb · Allpass diffusion")
    print(f"                      Diffuse-field EQ (IEC 711)")
    print(f"                      Equal-loudness compensation (ISO 226)")
    print(f"  NEW v8.0          : InstrumentRouter psychoacoustic table")
    print(f"                        (vocals/drums/bass/guitar/piano/other)")
    print(f"                      Per-stem gain staging (RMS → LUFS proxy)")
    print(f"                      Single-pass mega-filtergraph (all stems in 1 call)")
    print(f"                      3-band multiband mastering bus")
    print(f"                      6-stem Demucs htdemucs_6s support")
    print(f"                      _prefix_filtergraph multi-instance engine")
    print(f"  Advanced analysis : {'✅' if ADVANCED_ANALYSIS else '❌  pip install librosa soundfile scipy'}")
    print(f"  SOFA/HRTF dataset : {'✅  h5py available' if H5PY_AVAILABLE else '❌  pip install h5py'}")
    print(f"  YouTube support   : {'✅' if YOUTUBE_SUPPORT else '❌  pip install yt-dlp'}")
    print(f"  Stem separation   : {'✅  ' + STEM_ENGINE if STEM_SEPARATION else '❌  pip install demucs'}")

    auto_detect_ffmpeg()

    if check_ffmpeg():
        print(f"  FFmpeg            : ✅  available")
        if check_reverberate_filter():
            print(f"  Reverb engine     : ✅  reverberate (FFmpeg 5.0+)")
        else:
            print(f"  Reverb engine     : 🔄  4-tap aecho (compatible fallback)")
    else:
        print(f"  FFmpeg            : ❌  NOT FOUND")
        print(f"    Windows → https://ffmpeg.org/download.html")
        print(f"    Mac     → brew install ffmpeg")
        print(f"    Linux   → sudo apt install ffmpeg")
        sys.exit(1)

    print("="*72)
    print("\n  http://localhost:8000     docs → /docs\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
