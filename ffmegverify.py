#!/usr/bin/env python3
"""
FFmpeg Version & Feature Verification Tool
==========================================
Checks FFmpeg version and required filter availability for 8D Audio Converter v5.0
"""

import subprocess
import sys
import json
from typing import Dict, List, Tuple

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def print_status(feature: str, status: bool, details: str = ""):
    icon = f"{Colors.GREEN}✓{Colors.RESET}" if status else f"{Colors.RED}✗{Colors.RESET}"
    color = Colors.GREEN if status else Colors.RED
    print(f"  {icon}  {feature:<40} {color}{details}{Colors.RESET}")

def get_ffmpeg_version() -> Tuple[str, str]:
    """Get FFmpeg version string and parse major.minor"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        first_line = result.stdout.split('\n')[0]
        version = first_line.split('version')[1].split()[0] if 'version' in first_line else 'unknown'
        
        # Parse major version
        try:
            major = version.split('.')[0]
            minor = version.split('.')[1] if len(version.split('.')) > 1 else '0'
        except:
            major, minor = '0', '0'
            
        return version, f"{major}.{minor}"
    except Exception as e:
        return 'not_found', '0.0'

def check_filter_available(filter_name: str) -> bool:
    """Check if a specific filter is available in FFmpeg"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-filters'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return filter_name in result.stdout
    except:
        return False

def get_available_filters() -> List[str]:
    """Get list of all available filters"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-filters'],
            capture_output=True,
            text=True,
            timeout=10
        )
        filters = []
        for line in result.stdout.split('\n'):
            if line.strip().startswith(' '):
                parts = line.split()
                if parts and len(parts) > 0:
                    filters.append(parts[0])
        return filters
    except:
        return []

def test_filter_syntax(filter_name: str, test_args: str = "") -> bool:
    """Test if filter works with null input (dry run)"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.1',
            '-af', f"{filter_name}{test_args}",
            '-f', 'null', '-'
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False

def check_codec_available(codec: str) -> bool:
    """Check if encoder/decoder is available"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return codec in result.stdout
    except:
        return False

def main():
    print_header("🎵 FFmpeg Version & Feature Verification")
    
    # 1. Check FFmpeg Installation
    print(f"{Colors.BOLD}1. FFmpeg Installation{Colors.RESET}")
    version_str, version_num = get_ffmpeg_version()
    
    if version_str == 'not_found':
        print(f"  {Colors.RED}✗{Colors.RESET}  FFmpeg is NOT installed or not in PATH")
        print(f"\n  {Colors.YELLOW}Installation Instructions:{Colors.RESET}")
        print(f"    • Windows: Download from https://ffmpeg.org/download.html")
        print(f"    • macOS:   brew install ffmpeg")
        print(f"    • Linux:   sudo apt install ffmpeg")
        sys.exit(1)
    else:
        print_status("FFmpeg Installed", True, f"v{version_str}")
    
    # Version requirements
    try:
        major, minor = map(int, version_num.split('.'))
        version_ok = (major > 4) or (major == 4 and minor >= 4)
    except:
        version_ok = False
    
    print_status("Version ≥ 4.4 Required", version_ok, 
                "Recommended: 5.0+ for best features" if version_ok else "Upgrade required!")
    
    # 2. Required Filters for 8D Engine v5.0
    print_header("2. Required Filters (Studio Mastering v5.0)")
    
    required_filters = {
        'equalizer': '12-band EQ processing',
        'acompressor': 'Per-band dynamics control',
        'loudnorm': 'EBU R128 loudness normalization',
        'stereotools': 'Stereo width & mid-side processing',
        'aecho': 'Reverb (fallback)',
        'reverberate': 'High-density reverb (v5.0 feature)',
        'aphase': 'HRTF phase shifting',
        'alimiter': 'True peak limiting',
        'pan': 'Mono downmix',
        'asplit': 'Multi-band splitting',
        'amix': 'Band mixing',
        'join': 'Stereo channel joining',
        'volume': 'LFO-based panning',
        'highpass': 'Frequency band separation',
        'lowpass': 'Frequency band separation',
        'bandpass': 'Vocal band isolation',
    }
    
    filter_results = {}
    for filter_name, description in required_filters.items():
        available = check_filter_available(filter_name)
        filter_results[filter_name] = available
        print_status(f"{filter_name}", available, description)
    
    # 3. Codec Support
    print_header("3. Audio Codec Support")
    
    codecs = {
        'libmp3lame': 'MP3 encoding (320kbps)',
        'flac': 'FLAC lossless encoding',
        'pcm_s24le': '24-bit WAV encoding',
        'aac': 'AAC encoding (alternative)',
    }
    
    for codec, description in codecs.items():
        available = check_codec_available(codec)
        print_status(f"{codec}", available, description)
    
    # 4. Feature Compatibility Matrix
    print_header("4. Feature Compatibility Matrix")
    
    features = [
        ("6-Band Multi-Band Processing", filter_results.get('asplit', False) and filter_results.get('amix', False)),
        ("12-Band Parametric EQ", filter_results.get('equalizer', False)),
        ("HRTF Phase Shifting", filter_results.get('aphase', False)),
        ("Studio Reverb (reverberate)", filter_results.get('reverberate', False)),
        ("True Peak Limiting", filter_results.get('alimiter', False)),
        ("Loudness Normalization", filter_results.get('loudnorm', False)),
        ("Stereo Width Control", filter_results.get('stereotools', False)),
        ("Per-Band Compression", filter_results.get('acompressor', False)),
    ]
    
    compatible_count = sum(1 for _, status in features if status)
    total_count = len(features)
    
    for feature, status in features:
        print_status(feature, status, "✅ Supported" if status else "⚠️ Fallback available")
    
    # 5. Recommendations
    print_header("5. Recommendations")
    
    issues = []
    warnings = []
    
    if not version_ok:
        issues.append("FFmpeg version too old. Upgrade to 4.4+ minimum, 5.0+ recommended.")
    
    if not filter_results.get('reverberate', False):
        warnings.append("'reverberate' filter not available. Will fallback to 'aecho' (lower quality reverb).")
    
    if not filter_results.get('alimiter', False):
        warnings.append("'alimiter' filter not available. True peak limiting disabled.")
    
    if not filter_results.get('loudnorm', False):
        issues.append("'loudnorm' filter not available. Loudness normalization will fail.")
    
    if not filter_results.get('equalizer', False):
        issues.append("'equalizer' filter not available. 12-band EQ will not work.")
    
    if issues:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  CRITICAL ISSUES:{Colors.RESET}\n")
        for issue in issues:
            print(f"  • {issue}")
    
    if warnings:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  WARNINGS:{Colors.RESET}\n")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not issues and not warnings:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ All features available! Ready for Studio Mastering v5.0{Colors.RESET}\n")
    
    # 6. Quick Test
    print_header("6. Quick Processing Test")
    
    test_success = test_filter_syntax('equalizer', '=f=1000:t=q:w=100:g=3')
    print_status("Filter Chain Test", test_success, "Basic EQ chain functional" if test_success else "Filter syntax error")
    
    # Summary
    print_header("Summary")
    print(f"  FFmpeg Version:     {version_str}")
    print(f"  Features Available: {compatible_count}/{total_count}")
    print(f"  Compatibility:      {Colors.GREEN if compatible_count == total_count else Colors.YELLOW if compatible_count > total_count * 0.7 else Colors.RED}")
    print(f"  {'✅ READY FOR PRODUCTION' if compatible_count == total_count and not issues else '⚠️  REVIEW WARNINGS ABOVE'}{Colors.RESET}")
    print()

if __name__ == "__main__":
    main()