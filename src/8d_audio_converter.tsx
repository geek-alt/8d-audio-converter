import React, { useState, useRef, useEffect } from 'react';
import { Upload, Music, Download, Play, Pause, Settings, Sliders, Zap, Waves, Server, Chrome, AlertCircle } from 'lucide-react';

// ─── Design System ────────────────────────────────────────────────────────────
// Palette: near-black panels · copper-amber accent · warm neutrals
// Aesthetic: industrial recording studio console — specific, not gradient-chased

const FONT_STYLE = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

  * { box-sizing: border-box; }

  .plex { font-family: 'IBM Plex Mono', 'Courier New', monospace; }

  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    cursor: pointer;
    width: 100%;
  }

  input[type=range]::-webkit-slider-runnable-track {
    background: #3a3530;
    height: 2px;
    border-radius: 1px;
  }

  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    background: #c87c3a;
    border-radius: 1px;
    margin-top: -5px;
    transition: background 0.15s;
  }

  input[type=range]::-webkit-slider-thumb:hover {
    background: #e09050;
  }

  input[type=range]::-moz-range-track {
    background: #3a3530;
    height: 2px;
    border-radius: 1px;
  }

  input[type=range]::-moz-range-thumb {
    width: 12px;
    height: 12px;
    background: #c87c3a;
    border: none;
    border-radius: 1px;
  }

  input[type=range]:focus {
    outline: none;
  }

  input[type=range]:focus::-webkit-slider-thumb {
    box-shadow: 0 0 0 2px rgba(200, 124, 58, 0.4);
  }

  .tab-btn {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 12px 24px;
    border: none;
    background: transparent;
    cursor: pointer;
    transition: color 0.15s;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
  }

  .tab-active {
    color: #e0d4c4;
    border-bottom-color: #c87c3a;
  }

  .tab-inactive {
    color: #665a4e;
  }

  .tab-inactive:hover {
    color: #998070;
  }

  .btn-primary {
    background: #c87c3a;
    color: #0d0d0d;
    border: none;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: background 0.15s;
    padding: 11px 20px;
    border-radius: 2px;
  }

  .btn-primary:hover {
    background: #e09050;
  }

  .btn-primary:disabled {
    background: #3a2e24;
    color: #665a4e;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: transparent;
    color: #998070;
    border: 1px solid #3a3530;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 12px;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.15s;
    padding: 8px 14px;
    border-radius: 2px;
  }

  .btn-secondary:hover {
    border-color: #444;
    color: #c8b8a8;
  }

  .btn-selected {
    background: rgba(200, 124, 58, 0.15);
    border-color: #c87c3a;
    color: #e0a060;
  }

  .toggle-btn {
    background: #282420;
    border: 1px solid #3a3530;
    color: #665a4e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
    padding: 6px 12px;
    border-radius: 2px;
    white-space: nowrap;
  }

  .toggle-btn:hover {
    border-color: #3a3a3a;
    color: #998070;
  }

  .toggle-on {
    background: rgba(200, 124, 58, 0.12);
    border-color: rgba(200, 124, 58, 0.5);
    color: #d4905a;
  }

  .eq-bar-container {
    display: flex;
    align-items: stretch;
    gap: 2px;
    height: 64px;
    padding: 0 4px;
    background: #1e1b18;
    border: 1px solid #353028;
    border-radius: 2px;
    margin-bottom: 16px;
    overflow: hidden;
  }

  .eq-bar-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
  }

  .eq-bar-pos {
    position: absolute;
    bottom: 50%;
    width: 100%;
    background: #c87c3a;
    opacity: 0.75;
    transition: height 0.2s;
    border-radius: 1px 1px 0 0;
  }

  .eq-bar-neg {
    position: absolute;
    top: 50%;
    width: 100%;
    background: #8a4858;
    opacity: 0.65;
    transition: height 0.2s;
    border-radius: 0 0 1px 1px;
  }

  .eq-zero-line {
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: #3a3530;
    pointer-events: none;
  }

  .analysis-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    background: #282420;
    border: 1px solid #3a3530;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #665a4e;
  }

  .analysis-tag span:last-child {
    color: #d4905a;
  }

  .led-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
  }

  .led-green  { background: #4a9a5a; box-shadow: 0 0 4px rgba(74,154,90,0.5); }
  .led-red    { background: #9a4a4a; }
  .led-amber  { background: #c87c3a; box-shadow: 0 0 4px rgba(200,124,58,0.5); }

  .section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #554a40;
  }

  .value-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #d4905a;
    background: rgba(200, 124, 58, 0.08);
    border: 1px solid rgba(200, 124, 58, 0.2);
    padding: 1px 6px;
    border-radius: 2px;
  }

  .pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 12px;
    background: #252220;
    border: 1px solid #353028;
    border-radius: 2px;
    transition: all 0.2s;
  }

  .pipeline-step.active {
    background: #1a1612;
    border-color: rgba(200, 124, 58, 0.3);
  }

  .pipeline-step.done {
    background: #121a14;
    border-color: rgba(74, 154, 90, 0.3);
  }

  .step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #2a2a2a;
    min-width: 20px;
    padding-top: 1px;
  }

  .step-num.active { color: #c87c3a; }
  .step-num.done   { color: #4a9a5a; }

  .progress-bar-track {
    width: 100%;
    height: 3px;
    background: #302c28;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 4px;
  }

  .progress-bar-fill {
    height: 100%;
    background: #c87c3a;
    border-radius: 2px;
    transition: width 0.4s ease;
  }

  input[type=text], input[type=url] {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    background: #252220;
    border: 1px solid #3a3530;
    border-radius: 2px;
    color: #c8b8a8;
    padding: 10px 12px;
    transition: border-color 0.15s;
    outline: none;
    width: 100%;
  }

  input[type=text]:focus, input[type=url]:focus {
    border-color: #554a40;
  }

  input[type=text]::placeholder, input[type=url]::placeholder {
    color: #3a3a3a;
  }

  .scroll-thin::-webkit-scrollbar { width: 4px; }
  .scroll-thin::-webkit-scrollbar-track { background: #252220; }
  .scroll-thin::-webkit-scrollbar-thumb { background: #3a3530; border-radius: 2px; }
`;

// ─── Sub-components ───────────────────────────────────────────────────────────

const SliderRow = ({ label, value, min, max, step, onChange, unit = '', hint = '' }) => {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 12, color: '#7a6a5a', letterSpacing: '0.02em' }}>
          {label}
        </span>
        <span className="value-badge plex">
          {typeof value === 'number' && value % 1 !== 0 ? value.toFixed(2) : value}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))} />
      {hint && (
        <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 11, color: '#6a6258', marginTop: 4, marginBottom: 0 }}>
          {hint}
        </p>
      )}
    </div>
  );
};

const SectionDivider = ({ label }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '20px 0 14px' }}>
    <span className="section-label">{label}</span>
    <div style={{ flex: 1, height: 1, background: '#302c28' }} />
  </div>
);

const EQVisualizer = ({ bands }) => {
  const maxGain = 14;
  return (
    <div className="eq-bar-container">
      <div className="eq-zero-line" />
      {bands.map(({ freq, gain }) => {
        const barH = Math.abs(gain / maxGain) * 28;
        return (
          <div key={freq} className="eq-bar-col">
            {gain >= 0 ? (
              <div className="eq-bar-pos" style={{ height: barH }} />
            ) : (
              <div className="eq-bar-neg" style={{ height: barH }} />
            )}
          </div>
        );
      })}
    </div>
  );
};


// ─── Main Component ───────────────────────────────────────────────────────────

const AudioConverter = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [audioFile, setAudioFile] = useState(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [originalAudio, setOriginalAudio] = useState(null);
  const [processedAudio, setProcessedAudio] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentMode, setCurrentMode] = useState('original');

  // Batch
  const [isBatchMode, setIsBatchMode] = useState(false);
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchId, setBatchId] = useState(null);
  const [batchStatus, setBatchStatus] = useState(null);

  // 8D Parameters
  const [rotationSpeed, setRotationSpeed] = useState(0.15);
  const [reverbRoom, setReverbRoom] = useState(0.6);
  const [reverbMix, setReverbMix] = useState(0.3);
  const [bassRotation, setBassRotation] = useState(0.08);
  const [trebleRotation, setTrebleRotation] = useState(0.2);
  const [stereoWidth, setStereoWidth] = useState(1.0);
  const [elevation, setElevation] = useState(0.0);
  const [distance, setDistance] = useState(1.0);
  const [outputFormat, setOutputFormat] = useState('mp3');
  const [bitrate, setBitrate] = useState(320);

  // Backend
  const [backendStatus, setBackendStatus] = useState('checking');
  const [wsConnection, setWsConnection] = useState(null);
  const [audioAnalysis, setAudioAnalysis] = useState(null);
  const [reverbEngine, setReverbEngine] = useState('aecho');

  // 12-band EQ
  const [eqSub30Gain, setEqSub30Gain]         = useState(3.0);
  const [eqSub60Gain, setEqSub60Gain]         = useState(4.0);
  const [eqBass100Gain, setEqBass100Gain]     = useState(3.0);
  const [eqUbass200Gain, setEqUbass200Gain]   = useState(1.5);
  const [eqLowmid350Gain, setEqLowmid350Gain] = useState(-2.5);
  const [eqMid700Gain, setEqMid700Gain]       = useState(-1.0);
  const [eqUmid1500Gain, setEqUmid1500Gain]   = useState(1.0);
  const [eqPresence3kGain, setEqPresence3kGain] = useState(2.0);
  const [eqDef5kGain, setEqDef5kGain]         = useState(1.5);
  const [eqBril8kGain, setEqBril8kGain]       = useState(2.0);
  const [eqAir12kGain, setEqAir12kGain]       = useState(2.0);
  const [eqUair16kGain, setEqUair16kGain]     = useState(1.0);

  // Enhancement toggles
  const [enableVocalCenter, setEnableVocalCenter] = useState(false);
  const [vocalSafeBass, setVocalSafeBass]         = useState(true);
  const [instrumentEnhance, setInstrumentEnhance] = useState(true);

  // Studio controls
  const [reverbDensity, setReverbDensity] = useState(0.7);
  const [hrtfIntensity, setHrtfIntensity] = useState(1.0);

  const fileInputRef      = useRef(null);
  const batchFileInputRef = useRef(null);
  const audioElementRef   = useRef(null);

  useEffect(() => {
    checkBackendHealth();
    const iv = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(iv);
  }, []);

  const checkBackendHealth = async () => {
    try {
      const r = await fetch('http://localhost:8000/health');
      if (r.ok) {
        const d = await r.json();
        setBackendStatus('connected');
        setReverbEngine(d.reverb_engine || 'aecho');
      } else {
        setBackendStatus('error');
      }
    } catch {
      setBackendStatus('error');
    }
  };

  const connectWebSocket = (jobId) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        setProgress(data.progress);
        setProcessingStage(data.stage);
      } else if (data.type === 'complete') {
        setProgress(100);
        setProcessedAudio(data.output_url);
        setActiveTab('output');
        setIsProcessing(false);
        ws.close();
      } else if (data.type === 'error') {
        alert(`Processing error: ${data.message}`);
        setIsProcessing(false);
        ws.close();
      }
    };
    ws.onerror = () => { setIsProcessing(false); };
    setWsConnection(ws);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAudioFile(file);
    setOriginalAudio(URL.createObjectURL(file));
    await analyzeAudio(file);
    setActiveTab('controls');
  };

  const handleBatchFileUpload = (e) => {
    const files = Array.from(e.target.files || []).filter(f =>
      f.type.startsWith('audio/') || /\.(mp3|wav|m4a|flac|ogg)$/i.test(f.name)
    );
    setBatchFiles(files);
    setIsBatchMode(true);
  };

  const buildParams = () => ({
    rotation_speed: rotationSpeed, reverb_room: reverbRoom,
    reverb_mix: reverbMix, bass_rotation: bassRotation,
    treble_rotation: trebleRotation, stereo_width: stereoWidth,
    elevation, distance, output_format: outputFormat, bitrate,
    enable_hrtf: true, enable_convolution_reverb: true,
    enable_multi_band: true, sample_rate: 48000, bit_depth: 24,
    enable_vocal_center: enableVocalCenter, vocal_safe_bass: vocalSafeBass,
    instrument_enhance: instrumentEnhance,
    eq_sub30_gain: eqSub30Gain, eq_sub60_gain: eqSub60Gain,
    eq_bass100_gain: eqBass100Gain, eq_ubass200_gain: eqUbass200Gain,
    eq_lowmid350_gain: eqLowmid350Gain, eq_mid700_gain: eqMid700Gain,
    eq_umid1500_gain: eqUmid1500Gain, eq_presence3k_gain: eqPresence3kGain,
    eq_def5k_gain: eqDef5kGain, eq_bril8k_gain: eqBril8kGain,
    eq_air12k_gain: eqAir12kGain, eq_uair16k_gain: eqUair16kGain,
    eq_sub_bass_gain: 0, eq_bass_gain: 0,
    eq_low_mid_gain: 0, eq_presence_gain: 0, eq_air_gain: 0,
    reverb_density: reverbDensity, hrtf_intensity: hrtfIntensity,
    enable_limiter: true,
  });

  const processBatch = async () => {
    if (!batchFiles.length) return;
    setIsProcessing(true);
    const formData = new FormData();
    batchFiles.forEach(f => formData.append('files', f));
    formData.append('params', JSON.stringify(buildParams()));
    try {
      const r = await fetch('http://localhost:8000/batch/process', { method: 'POST', body: formData });
      if (r.ok) {
        const d = await r.json();
        setBatchId(d.batch_id);
        pollBatchStatus(d.batch_id);
      } else {
        alert('Batch processing failed.');
        setIsProcessing(false);
      }
    } catch (e) {
      alert('Failed to start batch: ' + e.message);
      setIsProcessing(false);
    }
  };

  const pollBatchStatus = (id) => {
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`http://localhost:8000/batch/${id}/status`);
        if (r.ok) {
          const s = await r.json();
          setBatchStatus(s);
          if (s.completed === s.total || s.failed === s.total) {
            clearInterval(iv);
            setIsProcessing(false);
          }
        }
      } catch {}
    }, 1000);
  };

  const downloadBatchResults = () => {
    if (!batchStatus?.jobs) return;
    batchStatus.jobs.forEach(job => {
      if (job.status === 'completed' && job.output_url) {
        const a = Object.assign(document.createElement('a'), {
          href: `http://localhost:8000${job.output_url}`,
          download: job.filename.replace(/\.[^/.]+$/, '_8d.' + outputFormat)
        });
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
      }
    });
  };

  const analyzeAudio = async (file) => {
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('http://localhost:8000/analyze', { method: 'POST', body: fd });
      if (r.ok) {
        const d = await r.json();
        setAudioAnalysis(d);
        if (d.recommended_settings) applyRecommendedSettings(d.recommended_settings);
      }
    } catch {}
  };

  const handleYoutubeDownload = async () => {
    if (!youtubeUrl) return;
    setIsProcessing(true);
    setProgress(5);
    setProcessingStage('Downloading from YouTube…');
    try {
      const r = await fetch('http://localhost:8000/youtube/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: youtubeUrl }),
      });
      if (r.ok) {
        const d = await r.json();
        setAudioFile({ name: d.title + '.mp3' });
        setOriginalAudio(d.audio_url);
        setProgress(15);
        setActiveTab('controls');
        await analyzeAudio(d.audio_url);
      } else {
        alert('Download failed. Check the URL.');
      }
    } catch (e) {
      alert('Backend error: ' + e.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const process8DAudio = async () => {
    if (!audioFile && !originalAudio) return;
    setIsProcessing(true);
    setProgress(0);
    setProcessingStage('Preparing pipeline…');

    const fd = new FormData();
    if (audioFile instanceof File) fd.append('audio_file', audioFile);
    else fd.append('audio_url', originalAudio);
    fd.append('params', JSON.stringify(buildParams()));

    try {
      const r = await fetch('http://localhost:8000/process', { method: 'POST', body: fd });
      if (r.ok) {
        const d = await r.json();
        connectWebSocket(d.job_id);
      } else {
        alert('Processing failed. Check backend logs.');
        setIsProcessing(false);
      }
    } catch (e) {
      alert('Backend error: ' + e.message);
      setIsProcessing(false);
    }
  };

  const applyPreset = (preset) => {
    const presets = {
      subtle:       { rotation: 0.08, reverb: 0.3, mix: 0.2, bass: 0.05, treble: 0.12, width: 0.8, elevation: 0, distance: 1.0, density: 0.6, hrtf: 0.7 },
      classic:      { rotation: 0.15, reverb: 0.6, mix: 0.3, bass: 0.08, treble: 0.2, width: 1.0, elevation: 0, distance: 1.0, density: 0.7, hrtf: 1.0 },
      intense:      { rotation: 0.3, reverb: 0.8, mix: 0.4, bass: 0.15, treble: 0.4, width: 1.2, elevation: 0.1, distance: 0.8, density: 0.85, hrtf: 1.2 },
      cinematic:    { rotation: 0.12, reverb: 0.85, mix: 0.45, bass: 0.06, treble: 0.18, width: 1.3, elevation: 0.15, distance: 1.2, density: 0.9, hrtf: 1.1 },
      experimental: { rotation: 0.5, reverb: 0.9, mix: 0.5, bass: 0.25, treble: 0.6, width: 1.5, elevation: 0.2, distance: 0.6, density: 0.95, hrtf: 1.5 },
    };
    const p = presets[preset];
    if (!p) return;
    setRotationSpeed(p.rotation); setReverbRoom(p.reverb); setReverbMix(p.mix);
    setBassRotation(p.bass); setTrebleRotation(p.treble); setStereoWidth(p.width);
    setElevation(p.elevation); setDistance(p.distance);
    setReverbDensity(p.density); setHrtfIntensity(p.hrtf);
  };

  const applyRecommendedSettings = (s) => {
    if (s.rotation_speed != null) setRotationSpeed(s.rotation_speed);
    if (s.reverb_room    != null) setReverbRoom(s.reverb_room);
    if (s.reverb_mix     != null) setReverbMix(s.reverb_mix);
    if (s.bass_rotation  != null) setBassRotation(s.bass_rotation);
    if (s.treble_rotation!= null) setTrebleRotation(s.treble_rotation);
    if (s.stereo_width   != null) setStereoWidth(s.stereo_width);
    if (s.elevation      != null) setElevation(s.elevation);
    if (s.distance       != null) setDistance(s.distance);
    if (s.enable_vocal_center != null) setEnableVocalCenter(s.enable_vocal_center);
    if (s.vocal_safe_bass!= null) setVocalSafeBass(s.vocal_safe_bass);
    if (s.instrument_enhance!= null) setInstrumentEnhance(s.instrument_enhance);
    if (s.eq_sub30_gain      != null) setEqSub30Gain(s.eq_sub30_gain);
    if (s.eq_sub60_gain      != null) setEqSub60Gain(s.eq_sub60_gain);
    if (s.eq_bass100_gain    != null) setEqBass100Gain(s.eq_bass100_gain);
    if (s.eq_ubass200_gain   != null) setEqUbass200Gain(s.eq_ubass200_gain);
    if (s.eq_lowmid350_gain  != null) setEqLowmid350Gain(s.eq_lowmid350_gain);
    if (s.eq_mid700_gain     != null) setEqMid700Gain(s.eq_mid700_gain);
    if (s.eq_umid1500_gain   != null) setEqUmid1500Gain(s.eq_umid1500_gain);
    if (s.eq_presence3k_gain != null) setEqPresence3kGain(s.eq_presence3k_gain);
    if (s.eq_def5k_gain      != null) setEqDef5kGain(s.eq_def5k_gain);
    if (s.eq_bril8k_gain     != null) setEqBril8kGain(s.eq_bril8k_gain);
    if (s.eq_air12k_gain     != null) setEqAir12kGain(s.eq_air12k_gain);
    if (s.eq_uair16k_gain    != null) setEqUair16kGain(s.eq_uair16k_gain);
    if (s.reverb_density     != null) setReverbDensity(s.reverb_density);
    if (s.hrtf_intensity     != null) setHrtfIntensity(s.hrtf_intensity);
  };

  const togglePlayback = () => {
    const audio = audioElementRef.current;
    if (!audio) return;
    isPlaying ? audio.pause() : audio.play();
    setIsPlaying(!isPlaying);
  };

  const switchMode = (mode) => {
    setCurrentMode(mode);
    setIsPlaying(false);
    if (audioElementRef.current) audioElementRef.current.pause();
  };

  const downloadAudio = async () => {
    if (!processedAudio) return;
    const blob = await (await fetch(processedAudio)).blob();
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement('a'), {
      href: url,
      download: `${audioFile?.name?.replace(/\.[^/.]+$/, '') ?? 'output'}_8D.${outputFormat}`
    });
    a.click();
    URL.revokeObjectURL(url);
  };

  const eqBands = [
    { freq: '30', gain: eqSub30Gain },     { freq: '60', gain: eqSub60Gain },
    { freq: '100', gain: eqBass100Gain },  { freq: '200', gain: eqUbass200Gain },
    { freq: '350', gain: eqLowmid350Gain },{ freq: '700', gain: eqMid700Gain },
    { freq: '1.5k', gain: eqUmid1500Gain },{ freq: '3k', gain: eqPresence3kGain },
    { freq: '5k', gain: eqDef5kGain },    { freq: '8k', gain: eqBril8kGain },
    { freq: '12k', gain: eqAir12kGain },  { freq: '16k', gain: eqUair16kGain },
  ];

  const pipelineStages = [
    { label: 'Deep Analysis', desc: 'MFCC · key detection · crest factor · stereo correlation', threshold: 15 },
    { label: 'Frequency Splitting', desc: '8-band HRTF separation', threshold: 30 },
    { label: 'Spatial Panning', desc: 'ITD inter-ear delay · LFO volume automation', threshold: 50 },
    { label: 'Pinna & Head Shadow', desc: 'Notch EQ at 8.5/10.5/13 kHz · head shadowing', threshold: 65 },
    { label: 'Reverb Engine', desc: 'Allpass diffusion → pre-delay → room reverb', threshold: 80 },
    { label: 'Mastering Chain', desc: 'Diffuse-field EQ · equal-loudness · 12-band EQ · limiter', threshold: 93 },
  ];

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <>
      <style>{FONT_STYLE}</style>
      <div style={{
        minHeight: '100vh',
        background: '#1a1714',
        fontFamily: '"DM Sans", sans-serif',
        color: '#c8b8a8',
        padding: '0 0 60px',
      }}>

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div style={{
          borderBottom: '1px solid #302c28',
          padding: '18px 28px',
          display: 'flex',
          alignItems: 'center',
          gap: 20,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 6, height: 6, background: '#c87c3a', borderRadius: '50%',
              boxShadow: '0 0 6px rgba(200,124,58,0.6)'
            }} />
            <span className="plex" style={{
              fontSize: 13, fontWeight: 500, letterSpacing: '0.08em',
              color: '#c8b8a8', textTransform: 'uppercase'
            }}>
              8D Audio Converter
            </span>
          </div>

          <div style={{ flex: 1 }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className={`led-dot ${
                backendStatus === 'connected' ? 'led-green' :
                backendStatus === 'checking'  ? 'led-amber' : 'led-red'
              }`} />
              <span className="plex" style={{ fontSize: 10, color: '#554a40', letterSpacing: '0.06em' }}>
                {backendStatus === 'connected' ? 'BACKEND ONLINE' :
                 backendStatus === 'checking'  ? 'CONNECTING' : 'OFFLINE'}
              </span>
            </div>
            {backendStatus === 'connected' && (
              <span className="plex" style={{ fontSize: 10, color: '#6a6258', letterSpacing: '0.04em' }}>
                {reverbEngine === 'reverberate' ? 'HIGH-DENSITY REV' : 'AECHO REV'} · v6.0
              </span>
            )}
          </div>
        </div>

        {/* ── Backend offline banner ──────────────────────────────────────── */}
        {backendStatus !== 'connected' && backendStatus !== 'checking' && (
          <div style={{
            margin: '16px 28px 0',
            padding: '12px 16px',
            background: '#252220',
            border: '1px solid #3a2820',
            borderRadius: 2,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
          }}>
            <div style={{ marginTop: 2 }}>
              <span className="led-dot led-red" />
            </div>
            <div>
              <p style={{ fontSize: 13, fontWeight: 500, color: '#c86060', margin: '0 0 4px' }}>
                Backend not running
              </p>
              <p style={{ fontSize: 11, color: '#5a4a40', margin: 0 }}>
                Start the Python backend to enable processing →{' '}
                <code className="plex" style={{ fontSize: 11, color: '#887060' }}>python backend.py</code>
              </p>
            </div>
          </div>
        )}

        {/* ── Main Panel ─────────────────────────────────────────────────── */}
        <div style={{ maxWidth: 960, margin: '24px auto', padding: '0 28px' }}>

          {/* Tab bar */}
          <div style={{ display: 'flex', borderBottom: '1px solid #302c28', marginBottom: 28 }}>
            {[
              { id: 'upload', label: 'Source' },
              { id: 'controls', label: 'Parameters' },
              { id: 'output', label: 'Output' },
            ].map(tab => (
              <button
                key={tab.id}
                className={`tab-btn ${activeTab === tab.id ? 'tab-active' : 'tab-inactive'}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* ─── UPLOAD TAB ────────────────────────────────────────────── */}
          {activeTab === 'upload' && (
            <div style={{ maxWidth: 640 }}>

              {/* Mode toggle */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
                <button
                  className={`btn-secondary ${!isBatchMode ? 'btn-selected' : ''}`}
                  onClick={() => { setIsBatchMode(false); setBatchFiles([]); }}
                >
                  Single file
                </button>
                <button
                  className={`btn-secondary ${isBatchMode ? 'btn-selected' : ''}`}
                  onClick={() => setIsBatchMode(true)}
                >
                  Batch
                </button>
              </div>

              {!isBatchMode ? (
                <>
                  {/* Upload zone */}
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                      border: '1px dashed #2a2a2a',
                      borderRadius: 2,
                      padding: '48px 24px',
                      textAlign: 'center',
                      cursor: 'pointer',
                      transition: 'border-color 0.15s',
                      background: '#1e1b18',
                    }}
                    onMouseEnter={e => e.currentTarget.style.borderColor = '#3a3a3a'}
                    onMouseLeave={e => e.currentTarget.style.borderColor = '#2a2a2a'}
                  >
                    <p style={{ fontSize: 14, color: '#7a6a5a', margin: '0 0 6px', fontWeight: 500 }}>
                      Drop audio file here
                    </p>
                    <p className="plex" style={{ fontSize: 11, color: '#3a3a3a', margin: 0, letterSpacing: '0.04em' }}>
                      MP3 · WAV · M4A · FLAC · OGG
                    </p>
                    <input ref={fileInputRef} type="file" accept="audio/*"
                      onChange={handleFileUpload} style={{ display: 'none' }} />
                  </div>

                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    margin: '20px 0', color: '#5a5248'
                  }}>
                    <div style={{ flex: 1, height: 1, background: '#282420' }} />
                    <span className="plex" style={{ fontSize: 10, color: '#5a5248', letterSpacing: '0.1em' }}>OR</span>
                    <div style={{ flex: 1, height: 1, background: '#282420' }} />
                  </div>

                  {/* YouTube */}
                  <div>
                    <p style={{ fontSize: 11, color: '#554a40', margin: '0 0 8px', letterSpacing: '0.04em' }}>
                      YOUTUBE URL
                    </p>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <input
                        type="text"
                        placeholder="https://youtube.com/watch?v=..."
                        value={youtubeUrl}
                        onChange={e => setYoutubeUrl(e.target.value)}
                        style={{ flex: 1 }}
                      />
                      <button
                        className="btn-primary"
                        onClick={handleYoutubeDownload}
                        disabled={!youtubeUrl || backendStatus !== 'connected'}
                      >
                        Fetch
                      </button>
                    </div>
                    <p style={{ fontSize: 11, color: '#5a5248', margin: '6px 0 0', fontStyle: 'italic' }}>
                      Highest quality audio extracted automatically
                    </p>
                  </div>
                </>
              ) : (
                <>
                  {/* Batch upload zone */}
                  <div
                    onClick={() => batchFileInputRef.current?.click()}
                    style={{
                      border: '1px dashed #2a2a2a',
                      borderRadius: 2,
                      padding: '40px 24px',
                      textAlign: 'center',
                      cursor: 'pointer',
                      background: '#1e1b18',
                    }}
                  >
                    <p style={{ fontSize: 14, color: '#7a6a5a', margin: '0 0 6px', fontWeight: 500 }}>
                      Select multiple audio files
                    </p>
                    <p className="plex" style={{ fontSize: 11, color: '#3a3a3a', margin: 0 }}>
                      All files processed with shared settings
                    </p>
                    <input ref={batchFileInputRef} type="file" accept="audio/*" multiple
                      onChange={handleBatchFileUpload} style={{ display: 'none' }} />
                  </div>

                  {batchFiles.length > 0 && (
                    <div style={{ marginTop: 16 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                        <span style={{ fontSize: 12, color: '#7a6a5a' }}>
                          {batchFiles.length} file{batchFiles.length !== 1 ? 's' : ''} selected
                        </span>
                        <button className="btn-secondary" style={{ fontSize: 11 }}
                          onClick={() => { setBatchFiles([]); setBatchStatus(null); }}>
                          Clear
                        </button>
                      </div>
                      <div className="scroll-thin" style={{ maxHeight: 200, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {batchFiles.map((f, i) => (
                          <div key={i} style={{
                            display: 'flex', justifyContent: 'space-between',
                            padding: '7px 10px', background: '#252220',
                            border: '1px solid #353028', borderRadius: 2,
                          }}>
                            <span style={{ fontSize: 12, color: '#7a6a5a', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                              {f.name}
                            </span>
                            <span className="plex" style={{ fontSize: 10, color: '#6a6258', marginLeft: 12, flexShrink: 0 }}>
                              {(f.size / 1024 / 1024).toFixed(1)} MB
                            </span>
                          </div>
                        ))}
                      </div>
                      <button className="btn-primary" disabled={isProcessing}
                        onClick={processBatch}
                        style={{ width: '100%', marginTop: 12, padding: '12px' }}>
                        {isProcessing ? 'Processing…' : `Convert ${batchFiles.length} files`}
                      </button>
                    </div>
                  )}

                  {batchStatus && (
                    <div style={{ marginTop: 16, background: '#252220', border: '1px solid #353028', borderRadius: 2, padding: 16 }}>
                      <div style={{ display: 'flex', gap: 16, marginBottom: 10 }}>
                        {[
                          { label: 'Total', val: batchStatus.total, color: '#7a6a5a' },
                          { label: 'Done', val: batchStatus.completed, color: '#4a9a5a' },
                          { label: 'Running', val: batchStatus.processing, color: '#c87c3a' },
                          { label: 'Failed', val: batchStatus.failed, color: '#9a4a4a' },
                        ].map(({ label, val, color }) => (
                          <div key={label} style={{ textAlign: 'center' }}>
                            <span className="plex" style={{ fontSize: 18, color, display: 'block', lineHeight: 1 }}>{val}</span>
                            <span style={{ fontSize: 10, color: '#6a6258' }}>{label}</span>
                          </div>
                        ))}
                      </div>
                      <div className="progress-bar-track">
                        <div className="progress-bar-fill" style={{
                          width: `${(batchStatus.completed / batchStatus.total) * 100}%`,
                        }} />
                      </div>
                      {batchStatus.completed > 0 && (
                        <button className="btn-primary" style={{ width: '100%', marginTop: 12 }}
                          onClick={downloadBatchResults}>
                          Download {batchStatus.completed} completed
                        </button>
                      )}
                    </div>
                  )}
                </>
              )}

              {/* File info + analysis */}
              {audioFile && (
                <div style={{
                  marginTop: 20, padding: '14px 16px',
                  background: '#252220', border: '1px solid #353028', borderRadius: 2,
                }}>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
                    <div>
                      <p style={{ fontSize: 13, fontWeight: 500, color: '#c8b8a8', margin: '0 0 2px' }}>
                        {audioFile.name}
                      </p>
                      {audioFile.size && (
                        <span className="plex" style={{ fontSize: 10, color: '#554a40' }}>
                          {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                      )}
                    </div>
                    <button className="btn-primary" onClick={() => setActiveTab('controls')}>
                      Configure →
                    </button>
                  </div>

                  {audioAnalysis && (
                    <div style={{ marginTop: 14, paddingTop: 14, borderTop: '1px solid #353028' }}>
                      <p className="section-label" style={{ marginBottom: 10 }}>Analysis</p>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {[
                          { label: 'BPM', val: audioAnalysis.bpm || '—' },
                          { label: 'KEY', val: audioAnalysis.key ? `${audioAnalysis.key} ${audioAnalysis.mode || ''}` : '—' },
                          { label: 'GENRE', val: audioAnalysis.genre || '—' },
                          { label: 'DURATION', val: audioAnalysis.duration ? `${audioAnalysis.duration}s` : '—' },
                          { label: 'CREST', val: audioAnalysis.crest_factor_db != null ? `${audioAnalysis.crest_factor_db} dB` : '—' },
                          { label: 'STEREO', val: audioAnalysis.is_fake_stereo ? 'fake' : 'true' },
                          { label: 'TRANSIENTS', val: audioAnalysis.transient_density || '—' },
                          { label: 'HNR', val: audioAnalysis.hnr_db != null ? `${audioAnalysis.hnr_db} dB` : '—' },
                        ].map(({ label, val }) => (
                          <div key={label} className="analysis-tag">
                            <span>{label}</span>
                            <span>{String(val).toUpperCase()}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ─── CONTROLS TAB ──────────────────────────────────────────── */}
          {activeTab === 'controls' && (
            <div style={{ maxWidth: 760 }}>

              {/* Presets */}
              <SectionDivider label="Presets" />
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {['subtle', 'classic', 'intense', 'cinematic', 'experimental'].map(p => (
                  <button key={p} className="btn-secondary"
                    style={{ textTransform: 'capitalize' }}
                    onClick={() => applyPreset(p)}>
                    {p}
                  </button>
                ))}
              </div>

              {/* Core spatial parameters */}
              <SectionDivider label="Spatial Rotation" />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 24px' }}>
                <SliderRow label="Rotation Speed" value={rotationSpeed} min={0.05} max={1.0} step={0.01} onChange={setRotationSpeed} unit=" Hz" />
                <SliderRow label="Bass Rotation" value={bassRotation} min={0.02} max={0.5} step={0.01} onChange={setBassRotation} unit=" Hz" />
                <SliderRow label="Treble Rotation" value={trebleRotation} min={0.05} max={1.0} step={0.01} onChange={setTrebleRotation} unit=" Hz" />
                <SliderRow label="Stereo Width" value={stereoWidth} min={0.5} max={2.0} step={0.1} onChange={setStereoWidth} unit="×" />
              </div>

              <SectionDivider label="Reverb" />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 24px' }}>
                <SliderRow label="Room Size" value={reverbRoom} min={0} max={1} step={0.01} onChange={setReverbRoom} unit="%" hint="" />
                <SliderRow label="Wet Mix" value={reverbMix} min={0} max={1} step={0.01} onChange={setReverbMix} unit="%" />
                <SliderRow label="Density" value={reverbDensity} min={0.3} max={1.0} step={0.05} onChange={setReverbDensity} unit=""
                  hint="Higher = lusher tail + more diffusion" />
                <SliderRow label="HRTF Shadowing" value={hrtfIntensity} min={0} max={1.5} step={0.1} onChange={setHrtfIntensity} unit=""
                  hint="Head blocking of high frequencies" />
              </div>

              <SectionDivider label="3D Positioning" />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 24px' }}>
                <SliderRow label="Elevation" value={elevation} min={-1} max={1} step={0.05} onChange={setElevation}
                  hint="Positive = above, negative = below" />
                <SliderRow label="Distance" value={distance} min={0.3} max={2.0} step={0.1} onChange={setDistance} unit="×"
                  hint="Closer = louder and drier" />
              </div>

              {/* Studio v6.0 info tags */}
              <div style={{ marginTop: 16, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {[
                  'ITD Inter-ear Delay',
                  'Pinna Notch EQ',
                  'Diffuse-Field EQ',
                  'Equal Loudness',
                  'Allpass Diffusion',
                  reverbEngine === 'reverberate' ? 'High-Density Reverb' : '4-Tap Aecho',
                ].map(tag => (
                  <div key={tag} className="analysis-tag" style={{ color: '#665a4e' }}>
                    {tag}
                  </div>
                ))}
              </div>

              {/* Enhancement toggles */}
              <SectionDivider label="Enhancement" />
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {[
                  { label: 'Vocal Center', val: enableVocalCenter, set: setEnableVocalCenter },
                  { label: 'Vocal-Safe Bass', val: vocalSafeBass, set: setVocalSafeBass },
                  { label: 'Instrument Enhance', val: instrumentEnhance, set: setInstrumentEnhance },
                ].map(({ label, val, set }) => (
                  <button key={label} className={`toggle-btn ${val ? 'toggle-on' : ''}`}
                    onClick={() => set(!val)}>
                    {val ? '◈' : '◇'} {label}
                  </button>
                ))}
              </div>

              {/* Advanced frequency controls */}
              <SectionDivider label="Frequency Controls" />
              <button
                className="btn-secondary"
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{ marginBottom: showAdvanced ? 16 : 0, display: 'flex', alignItems: 'center', gap: 8 }}
              >
                <span className="plex" style={{ fontSize: 10 }}>{showAdvanced ? '▼' : '▶'}</span>
                12-Band EQ & Rotation
              </button>

              {showAdvanced && (
                <div style={{
                  background: '#1e1b18',
                  border: '1px solid #353028',
                  borderRadius: 2,
                  padding: 20,
                }}>
                  {/* EQ visualizer */}
                  <p className="section-label" style={{ marginBottom: 10 }}>Frequency Response</p>
                  <EQVisualizer bands={eqBands} />

                  {/* EQ sliders */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 24px' }}>
                    {[
                      { label: '30 Hz · Sub Rumble',   val: eqSub30Gain,      set: setEqSub30Gain,      min: -12, max: 12 },
                      { label: '60 Hz · Sub Punch',    val: eqSub60Gain,      set: setEqSub60Gain,      min: -12, max: 12 },
                      { label: '100 Hz · Bass Warmth', val: eqBass100Gain,    set: setEqBass100Gain,    min: -12, max: 12 },
                      { label: '200 Hz · Upper Bass',  val: eqUbass200Gain,   set: setEqUbass200Gain,   min: -12, max:  6 },
                      { label: '350 Hz · Mud Cut',     val: eqLowmid350Gain,  set: setEqLowmid350Gain,  min: -14, max:  6 },
                      { label: '700 Hz · Nasal Cut',   val: eqMid700Gain,     set: setEqMid700Gain,     min: -12, max:  6 },
                      { label: '1.5 kHz · Instrument', val: eqUmid1500Gain,   set: setEqUmid1500Gain,   min: -12, max: 12 },
                      { label: '3 kHz · Presence',     val: eqPresence3kGain, set: setEqPresence3kGain, min: -12, max: 12 },
                      { label: '5 kHz · Definition',   val: eqDef5kGain,      set: setEqDef5kGain,      min: -12, max: 12 },
                      { label: '8 kHz · Brilliance',   val: eqBril8kGain,     set: setEqBril8kGain,     min: -12, max: 12 },
                      { label: '12 kHz · Air Shimmer', val: eqAir12kGain,     set: setEqAir12kGain,     min: -12, max: 12 },
                      { label: '16 kHz · Ultra Air',   val: eqUair16kGain,    set: setEqUair16kGain,    min: -12, max: 12 },
                    ].map(({ label, val, set, min, max }) => (
                      <div key={label}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <span style={{ fontSize: 11, color: '#5a4a3a', fontFamily: '"IBM Plex Mono",monospace' }}>
                            {label}
                          </span>
                          <span style={{
                            fontSize: 11, fontFamily: '"IBM Plex Mono",monospace',
                            color: val >= 0 ? '#c87c3a' : '#9a4a6a',
                          }}>
                            {val > 0 ? '+' : ''}{val.toFixed(1)} dB
                          </span>
                        </div>
                        <input type="range" min={min} max={max} step={0.5} value={val}
                          onChange={e => set(parseFloat(e.target.value))} />
                      </div>
                    ))}
                  </div>

                  {/* Per-band rotation */}
                  <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid #302c28' }}>
                    <p className="section-label" style={{ marginBottom: 14 }}>Per-Band Rotation</p>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 24px' }}>
                      <SliderRow label="Bass Band" value={bassRotation} min={0.02} max={0.5} step={0.01} onChange={setBassRotation} unit=" Hz" />
                      <SliderRow label="Treble Band" value={trebleRotation} min={0.05} max={1.0} step={0.01} onChange={setTrebleRotation} unit=" Hz" />
                    </div>
                  </div>
                </div>
              )}

              {/* Output format */}
              <SectionDivider label="Output" />
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                {['mp3', 'wav', 'flac'].map(fmt => (
                  <button key={fmt} className={`btn-secondary ${outputFormat === fmt ? 'btn-selected' : ''}`}
                    style={{ textTransform: 'uppercase', fontFamily: '"IBM Plex Mono",monospace', fontSize: 11 }}
                    onClick={() => setOutputFormat(fmt)}>
                    {fmt}
                  </button>
                ))}
              </div>
              {outputFormat === 'mp3' && (
                <SliderRow label="Bitrate" value={bitrate} min={128} max={320} step={32} onChange={v => setBitrate(parseInt(v))} unit=" kbps" />
              )}

              {/* Process button */}
              <div style={{ marginTop: 28 }}>
                <button
                  className="btn-primary"
                  disabled={isProcessing || !audioFile || backendStatus !== 'connected'}
                  onClick={process8DAudio}
                  style={{ width: '100%', padding: '14px', fontSize: 14, letterSpacing: '0.05em' }}
                >
                  {isProcessing ? (
                    <span className="plex">
                      {processingStage || 'Processing…'} — {progress.toFixed(0)}%
                    </span>
                  ) : (
                    'Generate 8D Audio'
                  )}
                </button>

                {/* Progress area */}
                {isProcessing && (
                  <div style={{ marginTop: 20 }}>
                    <div className="progress-bar-track" style={{ height: 4 }}>
                      <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
                    </div>
                    <p className="plex" style={{ fontSize: 10, color: '#554a40', marginTop: 6, letterSpacing: '0.04em' }}>
                      {processingStage}
                    </p>

                    {/* Pipeline stages */}
                    <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 4 }}>
                      {pipelineStages.map((stage, idx) => {
                        const isDone   = progress >= stage.threshold + 15;
                        const isActive = progress >= stage.threshold && !isDone;
                        return (
                          <div key={stage.label} className={`pipeline-step ${isDone ? 'done' : isActive ? 'active' : ''}`}>
                            <span className={`step-num ${isDone ? 'done' : isActive ? 'active' : ''}`}>
                              {isDone ? '✓' : `${idx + 1}.`}
                            </span>
                            <div>
                              <p style={{ fontSize: 12, fontWeight: 500, color: isDone ? '#4a9a5a' : isActive ? '#c8b8a8' : '#3a3530', margin: 0 }}>
                                {stage.label}
                              </p>
                              <p style={{ fontSize: 10, color: isDone ? '#2a5a34' : isActive ? '#554a3a' : '#252520', margin: '2px 0 0', fontFamily: '"IBM Plex Mono",monospace' }}>
                                {stage.desc}
                              </p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ─── OUTPUT TAB ────────────────────────────────────────────── */}
          {activeTab === 'output' && (
            <div style={{ maxWidth: 540 }}>
              {processedAudio ? (
                <>
                  <div style={{
                    padding: '12px 16px',
                    background: '#121a14',
                    border: '1px solid rgba(74,154,90,0.25)',
                    borderRadius: 2,
                    marginBottom: 24,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span className="led-dot led-green" />
                      <p style={{ fontSize: 13, fontWeight: 500, color: '#5a9a6a', margin: 0 }}>
                        Processing complete
                      </p>
                    </div>
                    <p style={{ fontSize: 11, color: '#3a8050', margin: '4px 0 0' }}>
                      8D audio with ITD · pinna EQ · diffuse-field correction — use headphones
                    </p>
                  </div>

                  {/* Compare modes */}
                  <SectionDivider label="Compare" />
                  <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
                    {['original', 'processed'].map(mode => (
                      <button key={mode} className={`btn-secondary ${currentMode === mode ? 'btn-selected' : ''}`}
                        style={{ textTransform: 'capitalize', flex: 1 }}
                        onClick={() => switchMode(mode)}>
                        {mode === 'original' ? 'Original' : '8D Processed'}
                      </button>
                    ))}
                  </div>

                  {/* Player */}
                  <div style={{
                    padding: '20px', background: '#252220',
                    border: '1px solid #353028', borderRadius: 2,
                    textAlign: 'center', marginBottom: 16,
                  }}>
                    <audio
                      ref={audioElementRef}
                      src={currentMode === 'original' ? originalAudio : processedAudio}
                      onEnded={() => setIsPlaying(false)}
                      style={{ display: 'none' }}
                    />
                    <button
                      onClick={togglePlayback}
                      style={{
                        width: 48, height: 48, borderRadius: '50%',
                        background: '#302c28', border: '1px solid #3a3530',
                        cursor: 'pointer', color: '#c87c3a',
                        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 16, transition: 'all 0.15s',
                      }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = '#c87c3a'; e.currentTarget.style.background = '#1e1612'; }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = '#3a3530'; e.currentTarget.style.background = '#2a2520'; }}
                    >
                      {isPlaying ? '⏸' : '▶'}
                    </button>
                    <p className="plex" style={{ fontSize: 10, color: '#6a6258', marginTop: 8, marginBottom: 0 }}>
                      {currentMode === 'original' ? 'ORIGINAL' : '8D PROCESSED'}
                    </p>
                  </div>

                  <button className="btn-primary" onClick={downloadAudio}
                    style={{ width: '100%', padding: '13px', fontSize: 13 }}>
                    Download {outputFormat.toUpperCase()} ({bitrate > 0 ? `${bitrate} kbps` : 'lossless'})
                  </button>
                </>
              ) : (
                <div style={{
                  padding: '60px 24px', textAlign: 'center',
                  border: '1px dashed #3a3530', borderRadius: 2,
                }}>
                  <p style={{ fontSize: 14, color: '#6a6258', marginBottom: 8, fontWeight: 500 }}>
                    No output yet
                  </p>
                  <p style={{ fontSize: 12, color: '#5a5248', marginBottom: 16 }}>
                    Upload a file and process it to see the result here
                  </p>
                  <button className="btn-secondary" onClick={() => setActiveTab('upload')}>
                    Go to source →
                  </button>
                </div>
              )}
            </div>
          )}

        </div>
      </div>
    </>
  );
};

export default AudioConverter;