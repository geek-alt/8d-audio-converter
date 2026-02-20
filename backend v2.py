#!/usr/bin/env python3
"""
8D Audio Converter - AI-Powered Professional Backend Server
===========================================================
FIXED VERSION — Bugs resolved:
  1. Filter chain was using commas for complex filtergraphs (needs semicolons + -filter_complex)
  2. send_complete() was never called → frontend never switched to output tab
  3. applyRecommendedSettings only applied 3 of 9 params (fixed in companion frontend)
  4. Analyzer only had 3 frequency bands and 7 genres → upgraded to 6 bands, 9 genres,
     per-band EQ filters actually applied to FFmpeg chain
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
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
    print("⚠️  Advanced audio analysis disabled. Install: pip install librosa soundfile scipy")
    ADVANCED_ANALYSIS = False

try:
    import yt_dlp
    YOUTUBE_SUPPORT = True
except ImportError:
    print("⚠️  YouTube download disabled. Install: pip install yt-dlp")
    YOUTUBE_SUPPORT = False

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(title="8D Audio Converter AI API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR   = BASE_DIR / "temp"

for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# ENHANCED AI AUDIO ANALYZER
# ============================================================================

class IntelligentAudioAnalyzer:
    """
    6-band spectral analyzer with genre detection (9 genres), rhythm complexity,
    energy-buildup detection, and per-band EQ recommendations fed into FFmpeg.
    """

    def __init__(self):
        # Extended to 9 genres
        self.genre_signatures = {
            'electronic': {'low_energy_ratio': (0.6, 1.0), 'spectral_centroid': (2000, 8000)},
            'classical':  {'low_energy_ratio': (0.3, 0.6), 'spectral_centroid': (1000, 4000)},
            'rock':       {'low_energy_ratio': (0.4, 0.7), 'spectral_centroid': (1500, 5000)},
            'hip_hop':    {'low_energy_ratio': (0.7, 1.0), 'spectral_centroid': (800,  3000)},
            'jazz':       {'low_energy_ratio': (0.3, 0.6), 'spectral_centroid': (1200, 4500)},
            'pop':        {'low_energy_ratio': (0.5, 0.7), 'spectral_centroid': (1500, 5000)},
            'ambient':    {'low_energy_ratio': (0.4, 0.7), 'spectral_centroid': (500,  2000)},
            'rnb':        {'low_energy_ratio': (0.6, 0.9), 'spectral_centroid': (900,  3500)},
            'metal':      {'low_energy_ratio': (0.5, 0.8), 'spectral_centroid': (2500, 7000)},
        }

    def analyze_comprehensive(self, file_path: str) -> Dict[str, Any]:
        if not ADVANCED_ANALYSIS:
            return self._basic_analysis(file_path)

        try:
            print(f"🧠 AI Analysis: Loading {file_path}...")
            y, sr = librosa.load(file_path, sr=None, mono=False)

            y_mono = librosa.to_mono(y) if len(y.shape) > 1 else y
            duration = librosa.get_duration(y=y_mono, sr=sr)

            analysis = {
                'duration': round(duration, 2),
                'sample_rate': sr,
                'channels': 2 if len(y.shape) > 1 else 1,
            }

            # BPM
            tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if tempo.size > 0 else 120.0
            else:
                tempo = float(tempo)
            analysis['bpm'] = round(tempo)
            analysis['beat_positions'] = beats.tolist() if len(beats) < 100 else []

            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0]
            analysis['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            analysis['spectral_centroid_std']  = float(np.std(spectral_centroids))

            # === 6-BAND ENERGY ANALYSIS ===
            S    = np.abs(librosa.stft(y_mono))
            freqs = librosa.fft_frequencies(sr=sr)

            def band_energy(lo, hi):
                idx = (freqs >= lo) & (freqs < hi)
                return float(np.sum(S[idx, :]))

            total_energy = float(np.sum(S)) + 1e-10

            sub_bass  = band_energy(20,   80)
            bass      = band_energy(80,  250)
            low_mid   = band_energy(250, 500)
            mids      = band_energy(500, 2000)
            hi_mid    = band_energy(2000, 6000)
            air       = band_energy(6000, sr // 2)

            analysis['sub_bass_ratio'] = sub_bass  / total_energy
            analysis['bass_ratio']     = bass      / total_energy
            analysis['low_mid_ratio']  = low_mid   / total_energy
            analysis['mid_ratio']      = mids      / total_energy
            analysis['hi_mid_ratio']   = hi_mid    / total_energy
            analysis['air_ratio']      = air       / total_energy

            # Legacy field for genre classifier compatibility
            analysis['high_ratio'] = (hi_mid + air) / total_energy

            # Dynamic range
            rms = librosa.feature.rms(y=y_mono)[0]
            analysis['dynamic_range'] = float(20 * np.log10(np.max(rms) / (np.mean(rms) + 1e-10)))

            # Vocal detection
            y_harmonic, y_percussive = librosa.effects.hpss(y_mono)
            harmonic_energy   = float(np.sum(np.abs(y_harmonic)))
            percussive_energy = float(np.sum(np.abs(y_percussive)))
            analysis['has_vocals']       = bool(harmonic_energy > percussive_energy * 0.8)
            analysis['vocal_prominence'] = harmonic_energy / (harmonic_energy + percussive_energy + 1e-10)

            # Rhythm complexity
            onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
            analysis['rhythm_complexity'] = float(np.std(onset_env))

            # Genre
            analysis['genre'] = self._classify_genre(analysis)

            # Energy profile
            analysis['energy_profile'] = self._calculate_energy_profile(y_mono, sr)

            print(f"🧠 Analysis complete: genre={analysis['genre']} bpm={analysis['bpm']} "
                  f"sub_bass={analysis['sub_bass_ratio']:.2f} bass={analysis['bass_ratio']:.2f} "
                  f"mids={analysis['mid_ratio']:.2f} air={analysis['air_ratio']:.2f}")

            analysis['recommended_settings'] = self._optimize_parameters(analysis)
            return analysis

        except Exception as e:
            print(f"❌ AI Analysis error: {e}")
            import traceback; traceback.print_exc()
            return self._basic_analysis(file_path)

    def _basic_analysis(self, file_path: str) -> Dict[str, Any]:
        duration = get_audio_duration(file_path)
        return {
            "duration": round(duration, 2),
            "sample_rate": 48000,
            "channels": 2,
            "bpm": None,
            "genre": "unknown",
            "sub_bass_ratio": 0.1,
            "bass_ratio": 0.2,
            "low_mid_ratio": 0.15,
            "mid_ratio": 0.35,
            "hi_mid_ratio": 0.15,
            "air_ratio": 0.05,
            "recommended_settings": {
                "rotation_speed": 0.15,
                "reverb_room": 0.6,
                "reverb_mix": 0.3,
                "bass_rotation": 0.08,
                "treble_rotation": 0.2,
                "stereo_width": 1.0,
                "elevation": 0.0,
                "distance": 1.0,
                "enable_vocal_center": False,
                "eq_sub_bass_gain": 3.0,
                "eq_bass_gain": 3.0,
                "eq_low_mid_gain": -2.0,
                "eq_presence_gain": 2.0,
                "eq_air_gain": 2.0,
            }
        }

    def _classify_genre(self, features: Dict[str, Any]) -> str:
        low_energy = features['bass_ratio'] + features.get('sub_bass_ratio', 0)
        centroid   = features['spectral_centroid_mean']

        best_match = 'unknown'
        best_score = float('inf')

        for genre, sig in self.genre_signatures.items():
            lo, hi  = sig['low_energy_ratio']
            clo, chi = sig['spectral_centroid']
            low_dist  = max(0, lo - low_energy, low_energy - hi)
            cent_dist = max(0, clo - centroid, centroid - chi)
            score = low_dist + cent_dist / 1000.0
            if score < best_score:
                best_score = score
                best_match = genre

        return best_match

    def _calculate_energy_profile(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        segment_length = sr * 2
        num_segments   = len(y) // segment_length
        energy_curve = [
            float(np.sqrt(np.mean(y[i*segment_length:(i+1)*segment_length]**2)))
            for i in range(num_segments)
        ]
        return {
            'curve': energy_curve[:50],
            'mean': float(np.mean(energy_curve)) if energy_curve else 0.0,
            'std':  float(np.std(energy_curve))  if energy_curve else 0.0,
            'has_buildup': self._detect_buildup(energy_curve),
        }

    def _detect_buildup(self, energy_curve: list) -> bool:
        if len(energy_curve) < 10:
            return False
        half = len(energy_curve) // 2
        return bool(np.mean(energy_curve[half:]) > np.mean(energy_curve[:half]) * 1.3)

    def _optimize_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        genre            = analysis['genre']
        bpm              = analysis.get('bpm', 120) or 120
        bass_ratio       = analysis.get('bass_ratio', 0.2)
        sub_bass_ratio   = analysis.get('sub_bass_ratio', 0.1)
        has_vocals       = analysis.get('has_vocals', False)
        vocal_prominence = analysis.get('vocal_prominence', 0.5)
        dynamic_range    = analysis.get('dynamic_range', 10)
        rhythm_complexity = analysis.get('rhythm_complexity', 0.5)

        params = {
            'rotation_speed':     0.15,
            'reverb_room':        0.60,
            'reverb_mix':         0.30,
            'bass_rotation':      0.08,
            'treble_rotation':    0.20,
            'stereo_width':       1.0,
            'elevation':          0.0,
            'distance':           1.0,
            'enable_vocal_center': has_vocals,
            'intensity_multiplier': 1.0,
            # EQ gains in dB (applied via FFmpeg equalizer filters)
            'eq_sub_bass_gain':   4.0,
            'eq_bass_gain':       3.0,
            'eq_low_mid_gain':   -2.0,
            'eq_presence_gain':   2.0,
            'eq_air_gain':        2.0,
        }

        # Genre-specific tuning
        if genre == 'electronic':
            params.update({
                'rotation_speed': 0.25, 'treble_rotation': 0.35, 'bass_rotation': 0.12,
                'reverb_room': 0.7, 'stereo_width': 1.3,
                'eq_sub_bass_gain': 5.0, 'eq_bass_gain': 4.0, 'eq_air_gain': 3.0,
            })
        elif genre == 'classical':
            params.update({
                'rotation_speed': 0.10, 'treble_rotation': 0.15, 'bass_rotation': 0.06,
                'reverb_room': 0.85, 'reverb_mix': 0.5, 'elevation': 0.15,
                'eq_sub_bass_gain': 1.0, 'eq_bass_gain': 1.0, 'eq_low_mid_gain': -1.0,
                'eq_presence_gain': 3.0, 'eq_air_gain': 2.5,
            })
        elif genre == 'rock':
            params.update({
                'rotation_speed': 0.18, 'bass_rotation': 0.10, 'treble_rotation': 0.25,
                'stereo_width': 1.2, 'reverb_room': 0.5,
                'eq_sub_bass_gain': 3.0, 'eq_bass_gain': 4.0, 'eq_presence_gain': 3.5,
            })
        elif genre == 'hip_hop':
            params.update({
                'rotation_speed': 0.12, 'bass_rotation': 0.06, 'treble_rotation': 0.20,
                'reverb_mix': 0.25, 'distance': 0.8,
                'eq_sub_bass_gain': 7.0, 'eq_bass_gain': 5.0, 'eq_low_mid_gain': -3.0,
                'eq_presence_gain': 2.0,
            })
        elif genre == 'rnb':
            params.update({
                'rotation_speed': 0.13, 'bass_rotation': 0.07, 'stereo_width': 1.1,
                'reverb_room': 0.65, 'reverb_mix': 0.35,
                'eq_sub_bass_gain': 5.0, 'eq_bass_gain': 4.0, 'eq_presence_gain': 2.5,
            })
        elif genre == 'metal':
            params.update({
                'rotation_speed': 0.22, 'bass_rotation': 0.12, 'treble_rotation': 0.35,
                'stereo_width': 1.3, 'reverb_room': 0.45,
                'eq_sub_bass_gain': 4.0, 'eq_bass_gain': 3.0, 'eq_low_mid_gain': -4.0,
                'eq_presence_gain': 5.0, 'eq_air_gain': 3.0,
            })
        elif genre == 'jazz':
            params.update({
                'rotation_speed': 0.14, 'reverb_room': 0.65, 'reverb_mix': 0.35,
                'stereo_width': 1.1,
                'eq_sub_bass_gain': 2.0, 'eq_bass_gain': 2.0, 'eq_presence_gain': 2.0,
            })
        elif genre == 'ambient':
            params.update({
                'rotation_speed': 0.08, 'bass_rotation': 0.04, 'treble_rotation': 0.12,
                'reverb_room': 0.9, 'reverb_mix': 0.6, 'stereo_width': 1.4, 'distance': 1.5,
                'eq_air_gain': 4.0, 'eq_sub_bass_gain': 1.0,
            })

        # BPM-synced rotation
        beats_per_second = bpm / 60.0
        for ratio in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            candidate = beats_per_second * ratio
            if 0.05 <= candidate <= 0.5:
                params['rotation_speed'] = round(candidate, 3)
                break

        # Heavy sub-bass → slow down rotation a touch to avoid clash
        if (sub_bass_ratio + bass_ratio) > 0.5:
            params['bass_rotation'] = round(params['bass_rotation'] * 0.75, 3)
            params['reverb_mix']    = round(params['reverb_mix'] * 0.85, 3)

        if has_vocals and vocal_prominence > 0.6:
            params['enable_vocal_center'] = True
            params['rotation_speed'] = round(params['rotation_speed'] * 0.85, 3)

        if dynamic_range > 15:
            params['reverb_mix']    = round(params['reverb_mix'] * 0.8, 3)
            params['stereo_width']  = round(params['stereo_width'] * 0.9, 3)
        elif dynamic_range < 8:
            params['stereo_width']      = round(params['stereo_width'] * 1.1, 3)
            params['intensity_multiplier'] = 1.2

        if rhythm_complexity > 0.7:
            params['rotation_speed'] = round(params['rotation_speed'] * 0.85, 3)

        # Clamp EQ gains to safe ranges
        for key in ('eq_sub_bass_gain', 'eq_bass_gain', 'eq_low_mid_gain',
                    'eq_presence_gain', 'eq_air_gain'):
            params[key] = max(-12.0, min(12.0, round(params[key], 1)))

        return params


audio_analyzer = IntelligentAudioAnalyzer()

# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect(self, job_id: str):
        self.active_connections.pop(job_id, None)

    async def send_progress(self, job_id: str, progress: int, stage: str):
        ws = self.active_connections.get(job_id)
        if ws:
            try:
                await ws.send_json({"type": "progress", "progress": progress, "stage": stage})
            except Exception as e:
                print(f"Progress send error: {e}")

    async def send_complete(self, job_id: str, output_url: str):
        ws = self.active_connections.get(job_id)
        if ws:
            try:
                await ws.send_json({"type": "complete", "output_url": output_url})
            except Exception as e:
                print(f"Complete send error: {e}")

    async def send_error(self, job_id: str, message: str):
        ws = self.active_connections.get(job_id)
        if ws:
            try:
                await ws.send_json({"type": "error", "message": message})
            except Exception as e:
                print(f"Error send error: {e}")

manager = ConnectionManager()

# ============================================================================
# MODELS
# ============================================================================

class ProcessingParams(BaseModel):
    rotation_speed: float = 0.15
    reverb_room: float = 0.6
    reverb_mix: float = 0.3
    bass_rotation: float = 0.08
    treble_rotation: float = 0.2
    stereo_width: float = 1.0
    elevation: float = 0.0
    distance: float = 1.0
    output_format: str = "mp3"
    bitrate: int = 320
    enable_hrtf: bool = True
    enable_convolution_reverb: bool = True
    enable_multi_band: bool = True
    sample_rate: int = 48000
    bit_depth: int = 24
    enable_vocal_center: bool = False
    intensity_multiplier: float = 1.0
    # Per-band EQ gains (dB) from AI analysis
    eq_sub_bass_gain: float = 3.0
    eq_bass_gain: float = 3.0
    eq_low_mid_gain: float = -2.0
    eq_presence_gain: float = 2.0
    eq_air_gain: float = 2.0

class YouTubeDownloadRequest(BaseModel):
    url: str

# ============================================================================
# UTILITY
# ============================================================================

def auto_detect_ffmpeg():
    import platform
    if platform.system() != "Windows":
        return
    common_paths = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        os.path.expanduser(r"~\ffmpeg\bin"),
    ]
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    for path in common_paths:
        if os.path.exists(os.path.join(path, "ffmpeg.exe")):
            if path not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + path
            return

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_audio_duration(file_path: str) -> float:
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

async def get_ffmpeg_progress(line: str, total_duration: float) -> float:
    try:
        if 'time=' in line:
            time_str = line.split('time=')[1].split()[0]
            parts = time_str.split(':')
            if len(parts) == 3:
                total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                if total_duration > 0:
                    return min(total_seconds / total_duration, 1.0)
    except Exception:
        pass
    return 0.0

# ============================================================================
# FILTER BUILDERS  (BUG FIX: use semicolons + -filter_complex)
# ============================================================================

def _eq_chain(params: ProcessingParams) -> str:
    """
    Build a chain of equalizer filters driven by AI analysis results.
    Returned as a linear filter string to append after the 8D panning stage.
    """
    eq_filters = []

    # Sub-bass shelf (40 Hz)
    if abs(params.eq_sub_bass_gain) > 0.5:
        eq_filters.append(
            f"equalizer=f=40:t=h:w=60:g={params.eq_sub_bass_gain:.1f}"
        )
    # Bass shelf (150 Hz)
    if abs(params.eq_bass_gain) > 0.5:
        eq_filters.append(
            f"equalizer=f=150:t=h:w=200:g={params.eq_bass_gain:.1f}"
        )
    # Low-mid cut (400 Hz) — remove muddiness
    if abs(params.eq_low_mid_gain) > 0.5:
        eq_filters.append(
            f"equalizer=f=400:t=h:w=200:g={params.eq_low_mid_gain:.1f}"
        )
    # Presence / vocal cut-through (3 kHz)
    if abs(params.eq_presence_gain) > 0.5:
        eq_filters.append(
            f"equalizer=f=3000:t=h:w=2000:g={params.eq_presence_gain:.1f}"
        )
    # Air / shimmer (12 kHz)
    if abs(params.eq_air_gain) > 0.5:
        eq_filters.append(
            f"equalizer=f=12000:t=h:w=4000:g={params.eq_air_gain:.1f}"
        )

    return ",".join(eq_filters) if eq_filters else ""


def build_simple_filtergraph(params: ProcessingParams) -> str:
    """
    Simple single-band 8D.  Returns a -filter_complex string.
    BUG FIX: complex filtergraph nodes are separated by semicolons, not commas.
    """
    rot   = params.rotation_speed * params.intensity_multiplier
    phase = 0.1
    eq    = _eq_chain(params)

    reverb = (
        f"aecho=in_gain={1-params.reverb_mix*0.3:.2f}"
        f":out_gain={params.reverb_mix*0.7:.2f}"
        f":delays={int(params.reverb_room*50)}|{int(params.reverb_room*73)}"
        f":decays={params.reverb_room*0.5:.2f}|{params.reverb_room*0.3:.2f}"
    )

    # Build the complex filtergraph
    # We must use semicolons between each named-stream segment
    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_in]",
        f"[mono_in]asplit=2[sl][sr]",
        f"[sl]volume='sin(2*PI*{rot}*t)':eval=frame[vl]",
        f"[sr]volume='cos(2*PI*{rot}*t+{phase})':eval=frame[vr]",
        f"[vl][vr]join=inputs=2:channel_layout=stereo[joined]",
        f"[joined]{reverb}[rev]",
        f"[rev]stereotools=mlev={params.stereo_width}[wide]",
    ]

    if eq:
        parts.append(f"[wide]{eq}[eqd]")
        parts.append("[eqd]loudnorm=I=-16:TP=-1.5:LRA=11[out]")
        out_label = "[out]"
    else:
        parts.append("[wide]loudnorm=I=-16:TP=-1.5:LRA=11[out]")
        out_label = "[out]"

    return ";".join(parts), out_label


def build_multiband_filtergraph(params: ProcessingParams) -> tuple:
    """
    3-band 8D (bass / mid / treble) with per-band rotation.
    BUG FIX: named-stream segments separated by semicolons, uses -filter_complex.
    """
    bass_freq   = 200
    treble_freq = 2000
    phase       = 0.1
    dvol        = 1.0 / max(params.distance, 0.3)
    intensity   = params.intensity_multiplier
    br          = params.bass_rotation   * intensity
    tr          = params.treble_rotation * intensity
    mr          = params.rotation_speed  * intensity

    reverb = (
        f"aecho=in_gain={1-params.reverb_mix*0.3:.2f}"
        f":out_gain={params.reverb_mix*0.7:.2f}"
        f":delays={int(params.reverb_room*50)}|{int(params.reverb_room*73)}"
        f":decays={params.reverb_room*0.5:.2f}|{params.reverb_room*0.3:.2f}"
    )
    eq = _eq_chain(params)

    parts = [
        # Downmix to mono then split into 3 bands
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",
        "[mono_src]asplit=3[bass_in][mid_in][treble_in]",

        # ── Bass band ──────────────────────────────────────────────────────
        f"[bass_in]lowpass=f={bass_freq}[bass_filt]",
        "[bass_filt]asplit=2[bl][br]",
        f"[bl]volume='sin(2*PI*{br:.4f}*t)':eval=frame[bvl]",
        f"[br]volume='cos(2*PI*{br:.4f}*t+{phase})':eval=frame[bvr]",
        "[bvl][bvr]join=inputs=2:channel_layout=stereo[bass_st]",
        f"[bass_st]volume={dvol:.4f}[bass8d]",

        # ── Mid band ───────────────────────────────────────────────────────
        f"[mid_in]highpass=f={bass_freq}[mid_hp]",
        f"[mid_hp]lowpass=f={treble_freq}[mid_filt]",
        "[mid_filt]asplit=2[ml][mr_ch]",
        f"[ml]volume='sin(2*PI*{mr:.4f}*t)':eval=frame[mvl]",
        f"[mr_ch]volume='cos(2*PI*{mr:.4f}*t+{phase})':eval=frame[mvr]",
        "[mvl][mvr]join=inputs=2:channel_layout=stereo[mid_st]",
        f"[mid_st]volume={dvol:.4f}[mid8d]",

        # ── Treble band ────────────────────────────────────────────────────
        f"[treble_in]highpass=f={treble_freq}[treble_filt]",
        "[treble_filt]asplit=2[tl][tr_ch]",
        f"[tl]volume='sin(2*PI*{tr:.4f}*t)':eval=frame[tvl]",
        f"[tr_ch]volume='cos(2*PI*{tr:.4f}*t+{phase})':eval=frame[tvr]",
        "[tvl][tvr]join=inputs=2:channel_layout=stereo[treble_st]",
        f"[treble_st]volume={dvol:.4f}[treble8d]",

        # ── Mix ────────────────────────────────────────────────────────────
        "[bass8d][mid8d][treble8d]amix=inputs=3:duration=first:normalize=0[mixed]",
        f"[mixed]{reverb}[rev]",
        f"[rev]stereotools=mlev={params.stereo_width}[wide]",
    ]

    if eq:
        parts.append(f"[wide]{eq}[eqd]")
        parts.append("[eqd]loudnorm=I=-16:TP=-1.5:LRA=11[out]")
    else:
        parts.append("[wide]loudnorm=I=-16:TP=-1.5:LRA=11[out]")

    # Elevation EQ (high-shelf tilt)
    if params.elevation != 0:
        elev_gain = round(params.elevation * 6, 1)
        parts.append(f"[out]equalizer=f=8000:t=h:w=2000:g={elev_gain}[elev_out]")
        return ";".join(parts), "[elev_out]"

    return ";".join(parts), "[out]"


def build_vocal_aware_filtergraph(params: ProcessingParams) -> tuple:
    """
    4-band 8D with a centred vocal band.
    BUG FIX: semicolons between named-stream segments, -filter_complex.
    """
    bass_freq  = 200
    vocal_lo   = 200
    vocal_hi   = 3000
    phase      = 0.1
    dvol       = 1.0 / max(params.distance, 0.3)
    intensity  = params.intensity_multiplier
    br         = params.bass_rotation   * intensity
    tr         = params.treble_rotation * intensity
    vr         = params.rotation_speed  * 0.5 * intensity

    reverb = (
        f"aecho=in_gain={1-params.reverb_mix*0.3:.2f}"
        f":out_gain={params.reverb_mix*0.7:.2f}"
        f":delays={int(params.reverb_room*50)}|{int(params.reverb_room*73)}"
        f":decays={params.reverb_room*0.5:.2f}|{params.reverb_room*0.3:.2f}"
    )
    eq = _eq_chain(params)

    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",
        "[mono_src]asplit=3[bass_in][vocal_in][high_in]",

        # Bass
        f"[bass_in]lowpass=f={bass_freq}[bass_filt]",
        "[bass_filt]asplit=2[bl][br]",
        f"[bl]volume='sin(2*PI*{br:.4f}*t)':eval=frame[bvl]",
        f"[br]volume='cos(2*PI*{br:.4f}*t+{phase})':eval=frame[bvr]",
        "[bvl][bvr]join=inputs=2:channel_layout=stereo[bass_st]",
        f"[bass_st]volume={dvol:.4f}[bass8d]",

        # Vocals (gently centred — smaller pan swing)
        f"[vocal_in]bandpass=f={int((vocal_lo+vocal_hi)/2)}:width_type=h:w={vocal_hi-vocal_lo}[voc_filt]",
        "[voc_filt]asplit=2[vl][vr_ch]",
        f"[vl]volume='0.7+0.3*sin(2*PI*{vr:.4f}*t)':eval=frame[vvl]",
        f"[vr_ch]volume='0.7+0.3*cos(2*PI*{vr:.4f}*t+{phase})':eval=frame[vvr]",
        "[vvl][vvr]join=inputs=2:channel_layout=stereo[voc_st]",
        f"[voc_st]volume={dvol*1.1:.4f}[vocal8d]",

        # High
        f"[high_in]highpass=f={vocal_hi}[high_filt]",
        "[high_filt]asplit=2[hl][hr]",
        f"[hl]volume='sin(2*PI*{tr:.4f}*t)':eval=frame[hvl]",
        f"[hr]volume='cos(2*PI*{tr:.4f}*t+{phase})':eval=frame[hvr]",
        "[hvl][hvr]join=inputs=2:channel_layout=stereo[high_st]",
        f"[high_st]volume={dvol:.4f}[high8d]",

        # Mix
        "[bass8d][vocal8d][high8d]amix=inputs=3:duration=first:normalize=0[mixed]",
        f"[mixed]{reverb}[rev]",
        f"[rev]stereotools=mlev={params.stereo_width}[wide]",
    ]

    if eq:
        parts.append(f"[wide]{eq}[eqd]")
        parts.append("[eqd]loudnorm=I=-16:TP=-1.5:LRA=11[out]")
    else:
        parts.append("[wide]loudnorm=I=-16:TP=-1.5:LRA=11[out]")

    if params.elevation != 0:
        elev_gain = round(params.elevation * 6, 1)
        parts.append(f"[out]equalizer=f=8000:t=h:w=2000:g={elev_gain}[elev_out]")
        return ";".join(parts), "[elev_out]"

    return ";".join(parts), "[out]"


# ============================================================================
# AI-POWERED 8D PROCESSING ENGINE
# ============================================================================

async def process_8d_audio(
    input_file: str,
    output_file: str,
    params: ProcessingParams,
    job_id: str,
    audio_analysis: Optional[Dict[str, Any]] = None
):
    """AI-Enhanced 8D audio processing with intelligent parameter adaptation."""
    try:
        print(f"\n🎬 Starting AI-powered 8D processing for job {job_id}")
        await manager.send_progress(job_id, 10, "AI building optimised filter graph...")

        # Choose filter graph
        if params.enable_multi_band:
            if audio_analysis and params.enable_vocal_center:
                filtergraph, out_label = build_vocal_aware_filtergraph(params)
            else:
                filtergraph, out_label = build_multiband_filtergraph(params)
        else:
            filtergraph, out_label = build_simple_filtergraph(params)

        print(f"🧠 Filter graph ready ({len(filtergraph)} chars)")

        await manager.send_progress(job_id, 30, "Applying AI-optimised binaural panning...")

        # Codec
        if params.output_format == "mp3":
            codec_opts = ["-c:a", "libmp3lame", "-b:a", f"{params.bitrate}k"]
        elif params.output_format == "flac":
            codec_opts = ["-c:a", "flac", "-compression_level", "8"]
        else:
            codec_opts = ["-c:a", "pcm_s24le"]

        # BUG FIX: use -filter_complex + -map (not -af) for named-stream graphs
        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-filter_complex", filtergraph,
            "-map", out_label,
            "-ar", str(params.sample_rate),
            *codec_opts,
            output_file
        ]

        print(f"🔧 FFmpeg command: {' '.join(cmd[:8])} ...")
        await manager.send_progress(job_id, 50, "Processing with professional-grade FFmpeg...")

        total_duration = get_audio_duration(input_file)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stderr_lines = []
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            line_str = line.decode('utf-8', errors='ignore')
            stderr_lines.append(line_str)
            if 'time=' in line_str:
                raw = await get_ffmpeg_progress(line_str, total_duration)
                progress = min(50 + int(raw * 40), 90)
                await manager.send_progress(job_id, progress, "Encoding immersive audio...")

        await process.wait()

        if process.returncode != 0:
            err_tail = "".join(stderr_lines[-30:])
            print(f"❌ FFmpeg failed (code {process.returncode}):\n{err_tail}")
            raise Exception(f"FFmpeg failed (code {process.returncode}). Check server logs.")

        await manager.send_progress(job_id, 95, "Finalising masterpiece...")

        if not os.path.exists(output_file):
            raise Exception("Output file was not created")

        await manager.send_progress(job_id, 100, "AI processing complete!")

        # ── BUG FIX: send_complete was never called ──────────────────────────
        output_filename = Path(output_file).name
        output_url = f"http://localhost:8000/download/{output_filename}"
        await manager.send_complete(job_id, output_url)
        # ────────────────────────────────────────────────────────────────────

        print(f"✅ Done — {output_url}")
        return True

    except Exception as e:
        print(f"❌ Processing error for job {job_id}: {e}")
        await manager.send_error(job_id, str(e))
        return False

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    ffmpeg_ok = check_ffmpeg()
    return {
        "status": "healthy" if ffmpeg_ok else "degraded",
        "ffmpeg": ffmpeg_ok,
        "advanced_analysis": ADVANCED_ANALYSIS,
        "youtube_support": YOUTUBE_SUPPORT,
        "version": "3.0.0"
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        file_id   = str(uuid.uuid4())
        file_path = TEMP_DIR / f"{file_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        analysis = audio_analyzer.analyze_comprehensive(str(file_path))
        file_path.unlink(missing_ok=True)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_audio(
    audio_file: Optional[UploadFile] = File(None),
    params: str = Form(...)
):
    try:
        processing_params = ProcessingParams(**json.loads(params))
        job_id = str(uuid.uuid4())

        if audio_file:
            input_path = UPLOAD_DIR / f"{job_id}_{audio_file.filename}"
            with open(input_path, "wb") as f:
                f.write(await audio_file.read())
        else:
            raise HTTPException(status_code=400, detail="No audio file provided")

        output_filename = f"{job_id}_8d.{processing_params.output_format}"
        output_path     = OUTPUT_DIR / output_filename

        asyncio.create_task(
            process_8d_audio(
                str(input_path),
                str(output_path),
                processing_params,
                job_id,
                None
            )
        )

        return {"job_id": job_id, "status": "processing"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket closed: {e}")
    finally:
        manager.disconnect(job_id)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)

@app.post("/youtube/download")
async def download_youtube(request: YouTubeDownloadRequest):
    if not YOUTUBE_SUPPORT:
        raise HTTPException(status_code=501, detail="yt-dlp not installed")
    try:
        job_id      = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{job_id}_youtube.mp3"
        ydl_opts    = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'mp3', 'preferredquality': '320'}],
            'outtmpl': str(output_path.with_suffix('')),
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info  = ydl.extract_info(request.url, download=True)
            title = info.get('title', 'Unknown')
        return {"success": True, "title": title,
                "audio_url": f"http://localhost:8000/download/{output_path.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Suppress Windows connection reset errors (harmless client disconnects)
    def ignore_connection_reset_errors(loop, context):
        """Filter out harmless Windows ConnectionResetError from logs"""
        if context.get("exception"):
            exc = context["exception"]
            if isinstance(exc, ConnectionResetError):
                return  # Silently ignore
        loop.default_exception_handler(context)
    
    # Apply handler to asyncio event loop
    asyncio.get_event_loop().set_exception_handler(ignore_connection_reset_errors)

    print("\n" + "="*70)
    print("🎵 8D Audio Converter - Professional Backend Server v3.0")
    print("="*70)
    print(f"Advanced Analysis : {'✅ Enabled' if ADVANCED_ANALYSIS else '❌ Disabled'}")
    print(f"YouTube Support   : {'✅ Enabled' if YOUTUBE_SUPPORT else '❌ Disabled'}")

    auto_detect_ffmpeg()

    if check_ffmpeg():
        print("FFmpeg            : ✅ Available")
    else:
        print("FFmpeg            : ❌ NOT FOUND")
        print("  Windows: https://ffmpeg.org/download.html")
        print("  Mac:     brew install ffmpeg")
        print("  Linux:   sudo apt install ffmpeg")
        sys.exit(1)

    print("="*70)
    print("\n🚀 Starting on http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
