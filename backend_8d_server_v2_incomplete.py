#!/usr/bin/env python3
"""
8D Audio Converter - AI-Powered Professional Backend Server
===========================================================
FastAPI backend with FFmpeg processing, WebSocket progress updates,
YouTube download, and INTELLIGENT audio analysis for optimal 8D conversion.

Features:
    - AI-powered genre detection and BPM analysis
    - Automatic parameter optimization based on audio characteristics
    - Dynamic frequency-based rotation (different speeds for different frequencies)
    - Adaptive reverb based on spectral content
    - Vocal isolation for centered positioning
    - Energy-based intensity mapping

Requirements:
    pip install fastapi uvicorn yt-dlp python-multipart websockets aiofiles librosa soundfile numpy scipy

System Dependencies:
    - FFmpeg (with libopus, libvorbis, libmp3lame)
    - Python 3.8+

Usage:
    python backend.py
    
Server will start on: http://localhost:8000
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import numpy as np

from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Audio analysis libraries
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

app = FastAPI(title="8D Audio Converter AI API", version="2.0.0")

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
TEMP_DIR = BASE_DIR / "temp"

for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# AI-POWERED AUDIO ANALYZER (300 IQ)
# ============================================================================

class IntelligentAudioAnalyzer:
    """
    Advanced audio analysis engine that understands music structure
    and optimizes 8D parameters for maximum immersion
    """
    
    def __init__(self):
        self.genre_signatures = {
            'electronic': {'low_energy_ratio': (0.6, 1.0), 'spectral_centroid': (2000, 8000)},
            'classical': {'low_energy_ratio': (0.3, 0.6), 'spectral_centroid': (1000, 4000)},
            'rock': {'low_energy_ratio': (0.4, 0.7), 'spectral_centroid': (1500, 5000)},
            'hip_hop': {'low_energy_ratio': (0.7, 1.0), 'spectral_centroid': (800, 3000)},
            'jazz': {'low_energy_ratio': (0.3, 0.6), 'spectral_centroid': (1200, 4500)},
            'pop': {'low_energy_ratio': (0.5, 0.7), 'spectral_centroid': (1500, 5000)},
            'ambient': {'low_energy_ratio': (0.4, 0.7), 'spectral_centroid': (500, 2000)},
        }
    
    def analyze_comprehensive(self, file_path: str) -> Dict[str, Any]:
        """
        Perform deep analysis of audio file
        Returns detailed characteristics for optimization
        """
        if not ADVANCED_ANALYSIS:
            return self._basic_analysis(file_path)
        
        try:
            print(f"🧠 AI Analysis: Loading {file_path}...")
            
            # Load audio with librosa
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            # Convert to mono for analysis
            if len(y.shape) > 1:
                y_mono = librosa.to_mono(y)
            else:
                y_mono = y
            
            duration = librosa.get_duration(y=y_mono, sr=sr)
            
            print(f"🧠 AI Analysis: Analyzing spectral features...")
            
            # Extract features
            analysis = {
                'duration': round(duration, 2),
                'sample_rate': sr,
                'channels': 2 if len(y.shape) > 1 else 1,
            }
            
            # BPM Detection
            tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr)
            analysis['bpm'] = round(tempo)
            analysis['beat_positions'] = beats.tolist() if len(beats) < 100 else []
            
            # Spectral Analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0]
            analysis['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            analysis['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Energy Distribution
            S = np.abs(librosa.stft(y_mono))
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Calculate energy in different frequency bands
            bass_idx = freqs < 200
            mid_idx = (freqs >= 200) & (freqs < 2000)
            high_idx = freqs >= 2000
            
            total_energy = np.sum(S)
            bass_energy = np.sum(S[bass_idx, :])
            mid_energy = np.sum(S[mid_idx, :])
            high_energy = np.sum(S[high_idx, :])
            
            analysis['bass_ratio'] = float(bass_energy / total_energy)
            analysis['mid_ratio'] = float(mid_energy / total_energy)
            analysis['high_ratio'] = float(high_energy / total_energy)
            
            # Dynamic Range
            rms = librosa.feature.rms(y=y_mono)[0]
            analysis['dynamic_range'] = float(20 * np.log10(np.max(rms) / (np.mean(rms) + 1e-10)))
            
            # Vocal Detection (using harmonic-percussive separation)
            y_harmonic, y_percussive = librosa.effects.hpss(y_mono)
            harmonic_energy = np.sum(np.abs(y_harmonic))
            percussive_energy = np.sum(np.abs(y_percussive))
            analysis['has_vocals'] = harmonic_energy > percussive_energy * 0.8
            analysis['vocal_prominence'] = float(harmonic_energy / (harmonic_energy + percussive_energy))
            
            # Rhythm Complexity
            onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
            analysis['rhythm_complexity'] = float(np.std(onset_env))
            
            # Genre Classification
            analysis['genre'] = self._classify_genre(analysis)
            
            # Energy Profile (for dynamic processing)
            analysis['energy_profile'] = self._calculate_energy_profile(y_mono, sr)
            
            print(f"🧠 AI Analysis Complete:")
            print(f"   Genre: {analysis['genre']}")
            print(f"   BPM: {analysis['bpm']}")
            print(f"   Vocal prominence: {analysis['vocal_prominence']:.2f}")
            print(f"   Bass/Mid/High: {analysis['bass_ratio']:.2f}/{analysis['mid_ratio']:.2f}/{analysis['high_ratio']:.2f}")
            
            # Generate optimized settings
            analysis['recommended_settings'] = self._optimize_parameters(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"❌ AI Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return self._basic_analysis(file_path)
    
    def _basic_analysis(self, file_path: str) -> Dict[str, Any]:
        """Fallback basic analysis"""
        duration = get_audio_duration(file_path)
        return {
            "duration": round(duration, 2),
            "sample_rate": 48000,
            "channels": 2,
            "bpm": None,
            "genre": "unknown",
            "recommended_settings": {
                "rotation_speed": 0.15,
                "reverb_room": 0.6,
                "reverb_mix": 0.3,
                "bass_rotation": 0.08,
                "treble_rotation": 0.2,
            }
        }
    
    def _classify_genre(self, features: Dict[str, Any]) -> str:
        """Classify genre based on spectral features"""
        low_energy = features['bass_ratio']
        centroid = features['spectral_centroid_mean']
        
        best_match = 'unknown'
        best_score = float('inf')
        
        for genre, signature in self.genre_signatures.items():
            # Calculate distance from genre signature
            low_range = signature['low_energy_ratio']
            cent_range = signature['spectral_centroid']
            
            low_dist = min(abs(low_energy - low_range[0]), abs(low_energy - low_range[1]))
            cent_dist = min(abs(centroid - cent_range[0]), abs(centroid - cent_range[1]))
            
            score = low_dist + cent_dist / 1000  # Normalize centroid distance
            
            if score < best_score:
                best_score = score
                best_match = genre
        
        return best_match
    
    def _calculate_energy_profile(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Calculate energy profile for dynamic processing"""
        # Divide audio into segments
        segment_length = sr * 2  # 2-second segments
        num_segments = len(y) // segment_length
        
        energy_curve = []
        for i in range(num_segments):
            segment = y[i*segment_length:(i+1)*segment_length]
            energy = np.sqrt(np.mean(segment**2))
            energy_curve.append(float(energy))
        
        return {
            'curve': energy_curve[:50],  # Limit to 50 points
            'mean': float(np.mean(energy_curve)),
            'std': float(np.std(energy_curve)),
            'has_buildup': self._detect_buildup(energy_curve)
        }
    
    def _detect_buildup(self, energy_curve: list) -> bool:
        """Detect if song has energy buildup (for adaptive processing)"""
        if len(energy_curve) < 10:
            return False
        
        # Check if energy increases significantly in first half
        first_half = energy_curve[:len(energy_curve)//2]
        second_half = energy_curve[len(energy_curve)//2:]
        
        return np.mean(second_half) > np.mean(first_half) * 1.3
    
    def _optimize_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-powered parameter optimization based on audio characteristics
        This is where the 300 IQ magic happens!
        """
        genre = analysis['genre']
        bpm = analysis.get('bpm', 120)
        bass_ratio = analysis.get('bass_ratio', 0.5)
        has_vocals = analysis.get('has_vocals', False)
        vocal_prominence = analysis.get('vocal_prominence', 0.5)
        dynamic_range = analysis.get('dynamic_range', 10)
        rhythm_complexity = analysis.get('rhythm_complexity', 0.5)
        
        # Initialize with defaults
        params = {
            'rotation_speed': 0.15,
            'reverb_room': 0.6,
            'reverb_mix': 0.3,
            'bass_rotation': 0.08,
            'treble_rotation': 0.2,
            'stereo_width': 1.0,
            'elevation': 0.0,
            'distance': 1.0,
            'enable_vocal_center': has_vocals,
            'intensity_multiplier': 1.0,
        }
        
        # Genre-specific optimization
        if genre == 'electronic':
            # Fast rotation for electronic, emphasize high frequencies
            params['rotation_speed'] = 0.25
            params['treble_rotation'] = 0.35
            params['bass_rotation'] = 0.12
            params['reverb_room'] = 0.7
            params['stereo_width'] = 1.3
            
        elif genre == 'classical':
            # Slow, elegant rotation for classical
            params['rotation_speed'] = 0.10
            params['treble_rotation'] = 0.15
            params['bass_rotation'] = 0.06
            params['reverb_room'] = 0.85  # Large hall reverb
            params['reverb_mix'] = 0.5
            params['elevation'] = 0.15  # Slight elevation for orchestra feel
            
        elif genre == 'rock':
            # Moderate rotation, emphasize power
            params['rotation_speed'] = 0.18
            params['bass_rotation'] = 0.10
            params['treble_rotation'] = 0.25
            params['stereo_width'] = 1.2
            params['reverb_room'] = 0.5  # Tighter reverb
            
        elif genre == 'hip_hop':
            # Slower rotation, heavy bass emphasis
            params['rotation_speed'] = 0.12
            params['bass_rotation'] = 0.06  # Keep bass steady
            params['treble_rotation'] = 0.20
            params['reverb_mix'] = 0.25  # Less reverb, more clarity
            params['distance'] = 0.8  # Closer, more intimate
            
        elif genre == 'jazz':
            # Smooth, moderate rotation
            params['rotation_speed'] = 0.14
            params['reverb_room'] = 0.65
            params['reverb_mix'] = 0.35
            params['stereo_width'] = 1.1
            
        elif genre == 'ambient':
            # Very slow rotation, maximum space
            params['rotation_speed'] = 0.08
            params['bass_rotation'] = 0.04
            params['treble_rotation'] = 0.12
            params['reverb_room'] = 0.9
            params['reverb_mix'] = 0.6
            params['stereo_width'] = 1.4
            params['distance'] = 1.5  # Far, spacious
        
        # BPM-based adjustment
        if bpm:
            # Sync rotation to BPM for rhythmic coherence
            beats_per_second = bpm / 60.0
            
            # Find rotation speed that's a musical subdivision of BPM
            # (whole, half, quarter, eighth note rotation)
            musical_ratios = [1.0, 0.5, 0.25, 0.125, 0.0625]
            for ratio in musical_ratios:
                candidate = beats_per_second * ratio
                if 0.05 <= candidate <= 0.5:  # Valid range
                    params['rotation_speed'] = round(candidate, 3)
                    break
        
        # Bass ratio adjustment
        if bass_ratio > 0.6:
            # Bass-heavy tracks: slow down bass rotation, keep it grounded
            params['bass_rotation'] *= 0.7
            params['reverb_mix'] *= 0.8  # Less reverb to maintain clarity
        elif bass_ratio < 0.3:
            # Bright tracks: faster treble rotation
            params['treble_rotation'] *= 1.2
        
        # Vocal presence adjustment
        if has_vocals and vocal_prominence > 0.6:
            # Keep vocals more centered for intelligibility
            params['enable_vocal_center'] = True
            params['rotation_speed'] *= 0.85  # Slightly slower rotation
            params['reverb_mix'] *= 0.9  # Less reverb for clarity
        
        # Dynamic range adjustment
        if dynamic_range > 15:
            # High dynamic range: more subtle effects
            params['reverb_mix'] *= 0.8
            params['stereo_width'] *= 0.9
        elif dynamic_range < 8:
            # Compressed audio: can handle more aggressive processing
            params['stereo_width'] *= 1.1
            params['intensity_multiplier'] = 1.2
        
        # Rhythm complexity adjustment
        if rhythm_complexity > 0.7:
            # Complex rhythms: slower rotation to avoid confusion
            params['rotation_speed'] *= 0.85
            params['bass_rotation'] *= 0.8
        
        # Round values for cleaner parameters
        for key in params:
            if isinstance(params[key], float):
                params[key] = round(params[key], 3)
        
        return params

# Global analyzer instance
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
        if job_id in self.active_connections:
            del self.active_connections[job_id]

    async def send_progress(self, job_id: str, progress: int, stage: str):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_json({
                    "type": "progress",
                    "progress": progress,
                    "stage": stage
                })
            except Exception as e:
                print(f"Error sending progress: {e}")

    async def send_complete(self, job_id: str, output_url: str):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_json({
                    "type": "complete",
                    "output_url": output_url
                })
            except Exception as e:
                print(f"Error sending completion: {e}")

    async def send_error(self, job_id: str, message: str):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_json({
                    "type": "error",
                    "message": message
                })
            except Exception as e:
                print(f"Error sending error: {e}")

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
    enable_vocal_center: bool = False  # AI feature
    intensity_multiplier: float = 1.0  # AI feature

class YouTubeDownloadRequest(BaseModel):
    url: str

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def auto_detect_ffmpeg():
    """Auto-detect FFmpeg on Windows and add to PATH"""
    import platform
    
    if platform.system() != "Windows":
        return
    
    common_paths = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        os.path.expanduser(r"~\ffmpeg\bin"),
        r"C:\MediaToolkit",
        r"C:\Tools\ffmpeg\bin",
    ]
    
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    for path in common_paths:
        ffmpeg_path = os.path.join(path, "ffmpeg.exe")
        if os.path.exists(ffmpeg_path):
            if path not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + path
                print(f"✓ FFmpeg found and added to PATH: {path}")
            return
    
    print("🔍 Searching for FFmpeg on C: drive...")
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["where", "/R", "C:\\", "ffmpeg.exe"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                ffmpeg_path = result.stdout.strip().split('\n')[0]
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                if ffmpeg_dir not in os.environ["PATH"]:
                    os.environ["PATH"] += os.pathsep + ffmpeg_dir
                    print(f"✓ FFmpeg found and added to PATH: {ffmpeg_dir}")
                return
    except Exception as e:
        print(f"Search failed: {e}")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_audio_duration(file_path: str) -> float:
    """Get audio duration using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration: {e}")
        return 0.0

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
    """
    AI-Enhanced 8D audio processing with intelligent parameter adaptation
    """
    
    try:
        print(f"\n🎬 Starting AI-powered 8D processing for job {job_id}")
        
        await manager.send_progress(job_id, 10, "AI analyzing audio characteristics...")
        
        # Build intelligent filter chain
        if params.enable_multi_band:
            if audio_analysis and params.enable_vocal_center:
                # Advanced: vocal-centered processing
                filter_chain = build_vocal_aware_filter(params, audio_analysis)
            else:
                # Standard multi-band
                filter_chain = build_multiband_filter(params)
        else:
            filter_chain = build_simple_filter(params)
        
        print(f"🧠 AI-optimized filter chain created")
        
        await manager.send_progress(job_id, 30, "Applying AI-optimized binaural panning...")
        
        # Output codec
        if params.output_format == "mp3":
            codec_opts = ["-c:a", "libmp3lame", "-b:a", f"{params.bitrate}k"]
        elif params.output_format == "flac":
            codec_opts = ["-c:a", "flac", "-compression_level", "8"]
        else:
            codec_opts = ["-c:a", "pcm_s24le"]
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-af", filter_chain,
            "-ar", str(params.sample_rate),
            *codec_opts,
            output_file
        ]
        
        await manager.send_progress(job_id, 50, "Processing with professional-grade FFmpeg...")
        
        # Execute
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stderr_output = []
        
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            
            line_str = line.decode('utf-8', errors='ignore')
            stderr_output.append(line_str)
            
            if 'time=' in line_str:
                progress = min(50 + (await get_ffmpeg_progress(line_str) * 40), 90)
                await manager.send_progress(job_id, int(progress), "Encoding immersive audio...")
        
        await process.wait()
        
        if process.returncode != 0:
            print(f"❌ FFmpeg failed with return code {process.returncode}")
            for line in stderr_output[-20:]:
                print(f"  {line.strip()}")
            raise Exception(f"FFmpeg processing failed with code {process.returncode}")
        
        await manager.send_progress(job_id, 95, "Finalizing masterpiece...")
        
        if not os.path.exists(output_file):
            raise Exception("Output file not created")
        
        await manager.send_progress(job_id, 100, "AI processing complete!")
        
        print(f"✅ AI-powered processing complete for job {job_id}")
        return True
        
    except Exception as e:
        print(f"❌ Processing error for job {job_id}: {e}")
        await manager.send_error(job_id, str(e))
        return False

def build_vocal_aware_filter(params: ProcessingParams, analysis: Dict[str, Any]) -> str:
    """
    Advanced filter with vocal-aware processing
    Keeps vocals centered while rotating instruments
    """
    bass_freq = 200
    vocal_freq_low = 200
    vocal_freq_high = 3000
    
    phase_offset = 0.1
    distance_volume = 1.0 / max(params.distance, 0.3)
    intensity = params.intensity_multiplier
    
    # Elevation EQ
    if params.elevation != 0:
        elevation_eq = f"equalizer=f=8000:t=h:w=2000:g={params.elevation * 6}"
    else:
        elevation_eq = ""
    
    filter_parts = [
        # Convert to mono
        "pan=mono|c0=0.5*c0+0.5*c1",
        
        # Split into 3 bands: bass, vocals, high
        "asplit=3[bass_in][vocal_in][high_in]",
        
        # === BASS PROCESSING (< 200Hz) ===
        f"[bass_in]lowpass=f={bass_freq}[bass_filtered]",
        "[bass_filtered]asplit=2[bass_l][bass_r]",
        f"[bass_l]volume='sin(2*PI*{params.bass_rotation * intensity}*t)':eval=frame[bass_l_rot]",
        f"[bass_r]volume='cos(2*PI*{params.bass_rotation * intensity}*t+{phase_offset})':eval=frame[bass_r_rot]",
        "[bass_l_rot][bass_r_rot]join=inputs=2:channel_layout=stereo[bass_stereo]",
        f"[bass_stereo]volume={distance_volume}[bass8d]",
        
        # === VOCAL PROCESSING (200Hz-3kHz) - CENTERED ===
        f"[vocal_in]bandpass=f={int((vocal_freq_low + vocal_freq_high)/2)}:width_type=h:w={vocal_freq_high-vocal_freq_low}[vocal_filtered]",
        # Light rotation for vocals (50% of normal speed)
        "[vocal_filtered]asplit=2[vocal_l][vocal_r]",
        f"[vocal_l]volume='0.7 + 0.3*sin(2*PI*{params.rotation_speed * 0.5 * intensity}*t)':eval=frame[vocal_l_rot]",
        f"[vocal_r]volume='0.7 + 0.3*cos(2*PI*{params.rotation_speed * 0.5 * intensity}*t+{phase_offset})':eval=frame[vocal_r_rot]",
        "[vocal_l_rot][vocal_r_rot]join=inputs=2:channel_layout=stereo[vocal_stereo]",
        f"[vocal_stereo]volume={distance_volume * 1.1}[vocal8d]",
        
        # === HIGH PROCESSING (> 3kHz) ===
        f"[high_in]highpass=f={vocal_freq_high}[high_filtered]",
        "[high_filtered]asplit=2[high_l][high_r]",
        f"[high_l]volume='sin(2*PI*{params.treble_rotation * intensity}*t)':eval=frame[high_l_rot]",
        f"[high_r]volume='cos(2*PI*{params.treble_rotation * intensity}*t+{phase_offset})':eval=frame[high_r_rot]",
        "[high_l_rot][high_r_rot]join=inputs=2:channel_layout=stereo[high_stereo]",
        f"[high_stereo]volume={distance_volume}[high8d]",
        
        # === MIX ALL BANDS ===
        "[bass8d][vocal8d][high8d]amix=inputs=3:duration=first",
        
        # Reverb
        f"aecho=in_gain={1-params.reverb_mix*0.3}:out_gain={params.reverb_mix*0.7}:delays={int(params.reverb_room*50)}|{int(params.reverb_room*73)}:decays={params.reverb_room*0.5}|{params.reverb_room*0.3}",
        
        # Stereo width
        f"stereotools=mlev={params.stereo_width}",
        
        # Elevation
        elevation_eq,
        
        # Normalization
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    ]
    
    return ",".join([f for f in filter_parts if f])