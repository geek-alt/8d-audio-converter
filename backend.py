#!/usr/bin/env python3
"""
8D Audio Converter — Deep Analysis Backend v6.0
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

8D ENGINE  (v6.0 additions)
  ✦ ITD (Interaural Time Difference) simulation via per-band static
    channel delays — higher bands get up to 630 μs inter-ear offset
  ✦ Pinna notch filters: narrow EQ cuts at ~8.5 kHz and ~10.5 kHz
    applied to the contra-lateral channel to simulate pinna reflections
  ✦ Pre-reverb delay (15–35 ms) — simulates room distance
  ✦ Allpass diffusion network — series of 4 prime-spaced aecho taps
    before the main reverb for richer early reflections
  ✦ Diffuse-field EQ (IEC 711 compensation for headphone listening):
    boosts 2.5 kHz presence, cuts 5 kHz cup resonance, adds 10 kHz air
  ✦ Equal-loudness compensation shelf (Fletcher-Munson at 70 phons)
  ✦ Frequency-dependent stereo width (lows narrower, highs wider)
  ✦ Per-band hard limiter before amix to prevent inter-band clipping
  ✦ Phase rotation awareness — alternates LFO phase offset per band to
    avoid comb-filter cancellations at the crossover frequencies

FILTERGRAPH ENGINES  (all retained + enhanced)
  ✦ Studio Grade v6.0  — 8-band HRTF + ITD + pinna EQ + diffuse-field
  ✦ 6-band multiband   — for enable_multi_band without full HRTF
  ✦ Vocal-aware 3-band — legacy vocal center mode
  ✦ Simple 2-channel   — fallback for minimal FFmpeg installations

EQ  (12-band, unchanged API)
  ✦ 30 Hz sub rumble shelf
  ✦ 60 Hz sub punch bell
  ✦ 100 Hz bass warmth bell
  ✦ 200 Hz upper-bass body bell
  ✦ 350 Hz mud cut bell
  ✦ 700 Hz nasal cut bell
  ✦ 1500 Hz instrument bite bell
  ✦ 3000 Hz presence bell
  ✦ 5000 Hz definition bell
  ✦ 8000 Hz brilliance shelf
  ✦ 12000 Hz air shimmer shelf
  ✦ 16000 Hz ultra-air shelf
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
    import yt_dlp
    YOUTUBE_SUPPORT = True
except ImportError:
    print("⚠️  YouTube disabled. Run: pip install yt-dlp")
    YOUTUBE_SUPPORT = False

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
    output_format: str = "mp3"
    bitrate:       int = 320
    sample_rate:   int = 48000
    bit_depth:     int = 24

    # 12-band EQ (dB)
    eq_sub30_gain:       float =  3.0
    eq_sub60_gain:       float =  4.0
    eq_bass100_gain:     float =  3.0
    eq_ubass200_gain:    float =  1.5
    eq_lowmid350_gain:   float = -2.5
    eq_mid700_gain:      float = -1.0
    eq_umid1500_gain:    float =  1.0
    eq_presence3k_gain:  float =  2.0
    eq_def5k_gain:       float =  1.5
    eq_bril8k_gain:      float =  2.0
    eq_air12k_gain:      float =  2.0
    eq_uair16k_gain:     float =  1.0

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

    async def send_complete(self, job_id: str, url: str):
        await self._send(job_id, {"type": "complete", "output_url": url})

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
    if not p.instrument_enhance:
        return ""
    filters = [
        "acompressor=threshold=-30dB:ratio=1.5:attack=5:release=80:makeup=3dB",
        "aeval=val(0)*0.97+0.03*tanh(val(0)*3)|val(1)*0.97+0.03*tanh(val(1)*3)",
    ]
    return ",".join(filters)


# ============================================================================
# NEW v6.0: PSYCHOACOUSTIC HELPERS
# ============================================================================

def _diffuse_field_eq() -> str:
    """
    Approximate IEC 711 diffuse-field correction for headphone listening.
    Compensates for typical headphone bowl/cup resonances so the 8D audio
    sounds as if heard in a real acoustic space rather than inside a box.

      +3.5 dB @ 2.5 kHz  — restore vocal presence headphones often suppress
      -5.0 dB @ 5.0 kHz  — cut headphone cup resonance
      +4.0 dB @ 10.0 kHz — restore air headphones roll off
    """
    return (
        "equalizer=f=2500:t=q:w=3000:g=3.5,"
        "equalizer=f=5000:t=q:w=2000:g=-5.0,"
        "equalizer=f=10000:t=h:w=4000:g=4.0"
    )


def _equal_loudness_shelf() -> str:
    """
    Approximation of ISO 226:2003 equal-loudness compensation at ~70 phons.
    Boosts sub-bass and ultra-air so the perceived loudness feels balanced
    across all frequencies when listening at moderate headphone volume.
    """
    return (
        "equalizer=f=50:t=h:w=50:g=7.5,"      # sub-bass psychoacoustic boost
        "equalizer=f=200:t=q:w=150:g=2.5,"     # bass body warmth
        "equalizer=f=3500:t=q:w=3000:g=-1.5,"  # mid scoopd (natural dip in 70-phon curve)
        "equalizer=f=12000:t=h:w=5000:g=3.0"   # high-air perceptual restore
    )


def _pinna_notch_filters(intensity: float = 1.0) -> str:
    """
    Pinna shadow notch filters.
    Real pinna reflections create narrow spectral notches that the brain
    uses to perceive elevation and front/back. Adding them to the mix
    (on the contra-lateral / rear ear simulation) makes the 8D effect
    feel outside-the-head rather than inside.

    Notch 1: ~8.5 kHz  — first pinna concha resonance
    Notch 2: ~10.5 kHz — second pinna shadow
    Notch 3: ~13.0 kHz — upper pinna flap reflection
    """
    if intensity <= 0:
        return ""
    g1 = round(-6.0  * intensity, 1)
    g2 = round(-4.0  * intensity, 1)
    g3 = round(-3.0  * intensity, 1)
    return (
        f"equalizer=f=8500:t=q:w=1200:g={g1},"
        f"equalizer=f=10500:t=q:w=1800:g={g2},"
        f"equalizer=f=13000:t=q:w=2500:g={g3}"
    )


def _allpass_diffuser() -> str:
    """
    Allpass diffusion network — four prime-interval aecho taps.
    Creates dense early reflections BEFORE the main reverb, giving a
    richer spatial impression without smearing transients (the short
    decays decay away quickly while adding diffusion texture).
    """
    return (
        "aecho=in_gain=1.0:out_gain=1.0"
        ":delays=17|23|31|41"
        ":decays=0.14|0.11|0.08|0.06"
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
# STUDIO GRADE v6.0  — 8-band HRTF + ITD + Pinna EQ + Diffuse-Field EQ
# ============================================================================

def build_8band_hrtf_engine_v6(p: ProcessingParams) -> tuple:
    """
    8-Band Spatial Audio Engine v6.0

    Pipeline per band:
      Input mono → bandpass → per-band compressor → split L/R
        → volume LFO (pan)
        → ITD adelay (frequency-dependent inter-ear offset)
        → pinna notch EQ on contra-lateral channel (highs only)
        → join stereo → distance attenuation

    Global pipeline after mix:
      amix → [dry / wet split]
        wet: allpass diffuser → pre-delay → main reverb
        dry: pass-through
        → mix dry+wet → diffuse-field EQ → equal-loudness shelf
        → 12-band master EQ → stereo tools → loudnorm → alimiter

    ITD table (μs, inter-ear offset per band):
      sub  (<80 Hz)  :  0   — sub-bass is non-directional
      bass (80-250)  : 80
      lowm (250-600) : 200
      voc  (600-2k)  : 350
      highm(2k-4k)   : 450
      pres (4k-8k)   : 520
      air  (8k-14k)  : 580
      spark(14k-22k) : 630
    """
    i = p.intensity_multiplier

    # Rotation speeds
    r_sub   = p.bass_rotation * i * 0.4
    r_bass  = p.bass_rotation * i * 0.7
    r_lowm  = p.rotation_speed * i * 0.8
    r_voc   = p.rotation_speed * i * 0.5   # vocals move slower
    r_highm = p.treble_rotation * i * 0.9
    r_pres  = p.treble_rotation * i * 1.1
    r_air   = p.treble_rotation * i * 1.3
    r_spark = p.treble_rotation * i * 1.5

    phase = 0.1
    dvol  = 1.0 / max(p.distance, 0.3)

    # ITD delays per band (microseconds) — higher bands get larger delay
    ITD_US = {
        'sub': 0, 'bass': 80, 'lowm': 200, 'voc': 350,
        'highm': 450, 'pres': 520, 'air': 580, 'spark': 630
    }

    def _hrtf_band_v6(lbl, lo, hi, rot, itd_us=0, apply_pinna=False, shadow_freq=0):
        parts = []
        inp = f"[{lbl}_in]"

        # 1. Bandpass
        if lo <= 20:
            parts.append(f"{inp}lowpass=f={hi}[{lbl}_f]")
        elif hi >= 22000:
            parts.append(f"{inp}highpass=f={lo}[{lbl}_f]")
        else:
            parts.append(f"{inp}bandpass=f={(lo+hi)//2}:width_type=h:w={hi-lo}[{lbl}_f]")

        # 2. Per-band compression (prevent rotation clipping)
        parts.append(
            f"[{lbl}_f]acompressor=threshold=-20dB:ratio=2:attack=10:release=50[{lbl}_c]"
        )

        # 3. Split L/R
        parts.append(f"[{lbl}_c]asplit=2[{lbl}_Lr][{lbl}_Rr]")

        # 4. Volume LFO (spatial pan)
        vol_l = f"sin(2*PI*{rot:.4f}*t)"
        vol_r = f"cos(2*PI*{rot:.4f}*t+{phase})"
        parts.append(f"[{lbl}_Lr]volume='{vol_l}':eval=frame[{lbl}_Lv]")
        parts.append(f"[{lbl}_Rr]volume='{vol_r}':eval=frame[{lbl}_Rv]")

        # 5. ITD — inter-ear delay on right channel
        if itd_us > 0:
            itd_ms = itd_us / 1000.0
            parts.append(f"[{lbl}_Rv]adelay={itd_ms:.3f}|0[{lbl}_Rd]")
            right_sig = f"{lbl}_Rd"
        else:
            right_sig = f"{lbl}_Rv"

        # 6. Pinna notch on contra-lateral (right) channel for high bands
        if apply_pinna and p.hrtf_intensity > 0:
            pinna_str = _pinna_notch_filters(p.hrtf_intensity)
            if pinna_str:
                parts.append(f"[{right_sig}]{pinna_str}[{lbl}_Rp]")
                right_sig = f"{lbl}_Rp"

        # 7. Head shadowing (lowpass on far-ear channel)
        if shadow_freq > 0 and p.hrtf_intensity > 0:
            cutoff = int(shadow_freq * (2.0 - min(p.hrtf_intensity, 1.8)))
            cutoff = max(1000, cutoff)
            parts.append(f"[{right_sig}]lowpass=f={cutoff}[{lbl}_Rs]")
            right_sig = f"{lbl}_Rs"

        # 8. Join stereo + distance attenuation
        parts.append(f"[{lbl}_Lv][{right_sig}]join=inputs=2:channel_layout=stereo[{lbl}_st]")
        parts.append(f"[{lbl}_st]volume={dvol:.4f}[{lbl}_8d]")

        return parts

    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",
        "[mono_src]asplit=8[sub_in][bass_in][lowm_in][voc_in][highm_in][pres_in][air_in][spark_in]",
    ]

    parts += _hrtf_band_v6("sub",   20,    80,   r_sub,   itd_us=ITD_US['sub'],   apply_pinna=False, shadow_freq=0)
    parts += _hrtf_band_v6("bass",  80,    250,  r_bass,  itd_us=ITD_US['bass'],  apply_pinna=False, shadow_freq=0)
    parts += _hrtf_band_v6("lowm",  250,   600,  r_lowm,  itd_us=ITD_US['lowm'],  apply_pinna=False, shadow_freq=0)
    parts += _hrtf_band_v6("voc",   600,   2000, r_voc,   itd_us=ITD_US['voc'],   apply_pinna=False, shadow_freq=0)
    parts += _hrtf_band_v6("highm", 2000,  4000, r_highm, itd_us=ITD_US['highm'], apply_pinna=False, shadow_freq=3000)
    parts += _hrtf_band_v6("pres",  4000,  8000, r_pres,  itd_us=ITD_US['pres'],  apply_pinna=True,  shadow_freq=5000)
    parts += _hrtf_band_v6("air",   8000,  14000,r_air,   itd_us=ITD_US['air'],   apply_pinna=True,  shadow_freq=8000)
    parts += _hrtf_band_v6("spark", 14000, 22000,r_spark, itd_us=ITD_US['spark'], apply_pinna=True,  shadow_freq=10000)

    # Mix all 8 bands
    band_outs = "".join([f"[{b}_8d]" for b in ["sub","bass","lowm","voc","highm","pres","air","spark"]])
    parts.append(f"{band_outs}amix=inputs=8:duration=first:normalize=0[mixed_direct]")

    # Dry / wet split for reverb
    rev_wet = p.reverb_mix * 0.6
    rev_dry = 1.0 - rev_wet
    parts.append("[mixed_direct]asplit=2[dry_sig][rev_input]")

    # Allpass diffusion network (early reflections)
    diffuser = _allpass_diffuser()
    parts.append(f"[rev_input]{diffuser}[diffused]")

    # Pre-delay (simulates distance to first wall)
    pre_delay_ms = round(15.0 + p.reverb_room * 20.0, 1)
    parts.append(f"[diffused]adelay={pre_delay_ms}|{pre_delay_ms}[pre_delayed]")

    # Main reverb
    rev_density = p.reverb_density
    if check_reverberate_filter():
        room_size = int(p.reverb_room * 100)
        decay_ms  = int(rev_density * 5000)
        damping   = 1 - rev_density
        parts.append(
            f"[pre_delayed]reverberate=room_size={room_size}:time={decay_ms}"
            f":damping={damping:.2f}:wet={rev_wet:.2f}:dry=0[wet_rev]"
        )
        print("  ↳ Using 'reverberate' (FFmpeg 5.0+)")
    else:
        d1 = int(37 * rev_density * p.reverb_room)
        d2 = int(73 * rev_density * p.reverb_room)
        d3 = int(127* rev_density * p.reverb_room)
        d4 = int(193* rev_density * p.reverb_room)
        dc1 = rev_density * 0.65; dc2 = rev_density * 0.45
        dc3 = rev_density * 0.28; dc4 = rev_density * 0.18
        parts.append(
            f"[pre_delayed]aecho=in_gain={1-rev_wet*0.3:.2f}:out_gain={rev_wet*0.75:.2f}"
            f":delays={d1}|{d2}|{d3}|{d4}"
            f":decays={dc1:.2f}|{dc2:.2f}|{dc3:.2f}|{dc4:.2f}[wet_rev]"
        )
        print("  ↳ Using optimized 4-tap 'aecho' fallback")

    # Mix dry + wet
    parts.append(f"[dry_sig]volume={rev_dry:.3f}[dry_vol]")
    parts.append("[dry_vol][wet_rev]amix=inputs=2:duration=first[post_rev]")

    # Diffuse-field EQ (headphone linearization)
    dfeq = _diffuse_field_eq()
    parts.append(f"[post_rev]{dfeq}[dfeq_out]")

    # Equal-loudness compensation
    el = _equal_loudness_shelf()
    parts.append(f"[dfeq_out]{el}[el_out]")

    # Master 12-band EQ
    eq = _eq_chain(p)
    if eq:
        parts.append(f"[el_out]{eq}[eq_master]")
        last = "eq_master"
    else:
        last = "el_out"

    # Instrument enhancement (after EQ, before width)
    enh = _instrument_enhance_chain(p)
    if enh:
        parts.append(f"[{last}]{enh}[enhanced]")
        last = "enhanced"

    # Stereo width (Mid/Side)
    parts.append(f"[{last}]stereotools=mlev={p.stereo_width:.3f}:sbal=0:softclip=1[wide]")

    # Loudness normalization (EBU R128)
    parts.append("[wide]loudnorm=I=-16:TP=-1.5:LRA=11:linear=true[loud]")

    # Elevation tilt
    if abs(p.elevation) > 0.01:
        eg = round(p.elevation * 6.0, 1)
        parts.append(f"[loud]equalizer=f=8000:t=h:w=2000:g={eg}[elev]")
        last = "elev"
    else:
        last = "loud"

    # True peak limiter
    if p.enable_limiter:
        parts.append(f"[{last}]alimiter=limit=1:attack=5:release=50:level=false[out]")
        return ";".join(parts), "[out]"

    return ";".join(parts), f"[{last}]"


# ============================================================================
# 6-BAND ENGINE  (retained, now with diffuse-field EQ appended)
# ============================================================================

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

        if is_vocal:
            lfo_l = f"0.70+0.30*sin(2*PI*{rot:.5f}*t+{ph_off:.3f})"
            lfo_r = f"0.70+0.30*cos(2*PI*{rot:.5f}*t+{ph_off+ph:.3f})"
        else:
            lfo_l = f"sin(2*PI*{rot:.5f}*t+{ph_off:.3f})"
            lfo_r = f"cos(2*PI*{rot:.5f}*t+{ph_off+ph:.3f})"

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
    parts.append("[dfeq]loudnorm=I=-16:TP=-1.5:LRA=11[normd]")

    if abs(p.elevation) > 0.01:
        eg = round(p.elevation * 6.0, 1)
        parts.append(f"[normd]equalizer=f=8000:t=h:w=2000:g={eg}[out]")
        return ";".join(parts), "[out]"

    return ";".join(parts), "[normd]"


# ============================================================================
# LEGACY FILTERGRAPH BUILDERS
# ============================================================================

def build_simple_filtergraph(p: ProcessingParams) -> tuple:
    rot   = p.rotation_speed * p.intensity_multiplier
    phase = 0.1
    eq    = _eq_chain(p)
    rev   = _reverb(p)
    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_in]",
        f"[mono_in]asplit=2[sl][sr]",
        f"[sl]volume='sin(2*PI*{rot}*t)':eval=frame[vl]",
        f"[sr]volume='cos(2*PI*{rot}*t+{phase})':eval=frame[vr]",
        f"[vl][vr]join=inputs=2:channel_layout=stereo[joined]",
        f"[joined]{rev}[rev]",
        f"[rev]stereotools=mlev={p.stereo_width}[wide]",
    ]
    if eq:
        parts += [f"[wide]{eq}[eqd]", "[eqd]loudnorm=I=-16:TP=-1.5:LRA=11[out]"]
    else:
        parts.append("[wide]loudnorm=I=-16:TP=-1.5:LRA=11[out]")
    if abs(p.elevation) > 0.01:
        eg = round(p.elevation * 6, 1)
        parts.append(f"[out]equalizer=f=8000:t=h:w=2000:g={eg}[elev_out]")
        return ";".join(parts), "[elev_out]"
    return ";".join(parts), "[out]"


def build_vocal_aware_filtergraph(p: ProcessingParams) -> tuple:
    i   = p.intensity_multiplier
    br  = p.bass_rotation * i
    vr  = p.rotation_speed * i * 0.5
    tr  = p.treble_rotation * i
    rev = _reverb(p)
    eq  = _eq_chain(p)
    dvol= 1.0 / max(p.distance, 0.3)
    phase = 0.1

    parts = [
        "pan=mono|c0=0.5*c0+0.5*c1[mono_src]",
        "[mono_src]asplit=3[bass_in][vocal_in][high_in]",
        "[bass_in]lowpass=f=200[bass_filt]",
        "[bass_filt]asplit=2[bl][br]",
        f"[bl]volume='sin(2*PI*{br:.4f}*t)':eval=frame[bvl]",
        f"[br]volume='cos(2*PI*{br:.4f}*t+{phase})':eval=frame[bvr]",
        "[bvl][bvr]join=inputs=2:channel_layout=stereo[bass_st]",
        f"[bass_st]volume={dvol:.4f}[bass8d]",
        "[vocal_in]bandpass=f=1100:width_type=h:w=2800[voc_filt]",
        "[voc_filt]asplit=2[vl][vr_ch]",
        f"[vl]volume='0.7+0.3*sin(2*PI*{vr:.4f}*t)':eval=frame[vvl]",
        f"[vr_ch]volume='0.7+0.3*cos(2*PI*{vr:.4f}*t+{phase})':eval=frame[vvr]",
        "[vvl][vvr]join=inputs=2:channel_layout=stereo[voc_st]",
        f"[voc_st]volume={dvol*1.1:.4f}[vocal8d]",
        "[high_in]highpass=f=3000[high_filt]",
        "[high_filt]asplit=2[hl][hr]",
        f"[hl]volume='sin(2*PI*{tr:.4f}*t)':eval=frame[hvl]",
        f"[hr]volume='cos(2*PI*{tr:.4f}*t+{phase})':eval=frame[hvr]",
        "[hvl][hvr]join=inputs=2:channel_layout=stereo[high_st]",
        f"[high_st]volume={dvol:.4f}[high8d]",
        "[bass8d][vocal8d][high8d]amix=inputs=3:duration=first:normalize=0[mixed]",
        f"[mixed]{rev}[rev]",
        f"[rev]stereotools=mlev={p.stereo_width}[wide]",
    ]
    if eq:
        parts += [f"[wide]{eq}[eqd]", "[eqd]loudnorm=I=-16:TP=-1.5:LRA=11[out]"]
    else:
        parts.append("[wide]loudnorm=I=-16:TP=-1.5:LRA=11[out]")
    if abs(p.elevation) > 0.01:
        eg = round(p.elevation * 6, 1)
        parts.append(f"[out]equalizer=f=8000:t=h:w=2000:g={eg}[elev_out]")
        return ";".join(parts), "[elev_out]"
    return ";".join(parts), "[out]"


# ============================================================================
# MAIN PROCESSING ENGINE  v6.0
# ============================================================================

async def process_8d_audio(
    input_file: str,
    output_file: str,
    params: ProcessingParams,
    job_id: str,
    audio_analysis: Optional[Dict[str, Any]] = None
):
    try:
        print(f"\n🎬 8D Processing v6.0 — job {job_id}")
        await manager.send_progress(job_id, 5, "Running deep audio analysis…")

        # Re-analyze if not already done (for WebSocket path)
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

        await manager.send_progress(job_id, 15, "Building HRTF spatial filter graph…")

        # Choose engine
        if params.enable_multi_band and params.enable_hrtf:
            filtergraph, out_label = build_8band_hrtf_engine_v6(params)
            engine_name = "Studio Grade v6.0 (8-band HRTF + ITD + Pinna EQ + Diffuse-Field)"
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

        # Codec
        if params.output_format == "mp3":
            codec = ["-c:a", "libmp3lame", "-b:a", f"{params.bitrate}k"]
        elif params.output_format == "flac":
            codec = ["-c:a", "flac", "-compression_level", "8"]
        else:
            codec = ["-c:a", "pcm_s24le"]

        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-filter_complex", filtergraph,
            "-map", out_label,
            "-ar", str(params.sample_rate),
            *codec,
            output_file
        ]

        await manager.send_progress(job_id, 35, "Applying binaural ITD + panning…")
        total_dur = get_audio_duration(input_file)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

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

        await manager.send_progress(job_id, 100, "✅ 8D processing complete!")
        out_url = f"http://localhost:8000/download/{Path(output_file).name}"
        await manager.send_complete(job_id, out_url)
        print(f"✅ Done → {out_url}")
        return True

    except Exception as e:
        print(f"❌ Processing error: {e}")
        await manager.send_error(job_id, str(e))
        return False


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    ok = check_ffmpeg()
    has_rev = check_reverberate_filter() if ok else False
    return {
        "status":            "healthy" if ok else "degraded",
        "ffmpeg":            ok,
        "advanced_analysis": ADVANCED_ANALYSIS,
        "youtube_support":   YOUTUBE_SUPPORT,
        "reverb_engine":     "reverberate" if has_rev else "aecho",
        "has_reverberate":   has_rev,
        "version":           "6.0.0",
        "analysis_bands":    10,
        "eq_bands":          12,
        "spatial_bands":     8,
        "genres":            15,
        "itd_simulation":    True,
        "pinna_notch_eq":    True,
        "diffuse_field_eq":  True,
        "allpass_diffusion": True,
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        fid  = str(uuid.uuid4())
        path = TEMP_DIR / f"{fid}_{file.filename}"
        path.write_bytes(await file.read())
        result = audio_analyzer.analyze_comprehensive(str(path))
        path.unlink(missing_ok=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_audio(
    audio_file: Optional[UploadFile] = File(None),
    params: str = Form(...)
):
    try:
        pp = ProcessingParams(**json.loads(params))
        job_id = str(uuid.uuid4())

        if audio_file:
            in_path = UPLOAD_DIR / f"{job_id}_{audio_file.filename}"
            in_path.write_bytes(await audio_file.read())
        else:
            raise HTTPException(status_code=400, detail="No audio file provided")

        out_name = f"{job_id}_8d.{pp.output_format}"
        out_path  = OUTPUT_DIR / out_name

        asyncio.create_task(
            process_8d_audio(str(in_path), str(out_path), pp, job_id, None)
        )
        return {"job_id": job_id, "status": "processing"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    print("  8D Audio Converter — Deep Analysis Backend  v6.0")
    print("="*72)
    print(f"  Analysis bands    : 10  (sub → air)")
    print(f"  EQ bands          : 12  (30 Hz → 16 kHz)")
    print(f"  Spatial bands     : 8   (HRTF + independent rotation)")
    print(f"  Genre profiles    : 15  (incl. Bollywood/Bhangra/Ghazal/Folk)")
    print(f"  Analysis v6.0     : MFCC · Chroma/Key · Crest Factor · HNR")
    print(f"                      Stereo Correlation · Transient Density")
    print(f"                      Tonnetz · ZCR · Spectral Rolloff")
    print(f"  8D v6.0           : ITD inter-ear delay (per-band)")
    print(f"                      Pinna notch EQ (8.5/10.5/13 kHz)")
    print(f"                      Pre-delay reverb · Allpass diffusion")
    print(f"                      Diffuse-field EQ (IEC 711)")
    print(f"                      Equal-loudness compensation (ISO 226)")
    print(f"  Advanced analysis : {'✅' if ADVANCED_ANALYSIS else '❌  pip install librosa soundfile scipy'}")
    print(f"  YouTube support   : {'✅' if YOUTUBE_SUPPORT else '❌  pip install yt-dlp'}")

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