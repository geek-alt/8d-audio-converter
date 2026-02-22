import React, { useState, useRef, useEffect } from 'react';
import { Upload, Music, Download, Play, Pause, Settings, Sliders, Zap, Waves, Server, Chrome, AlertCircle } from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════════════════
// MERIDIAN DESIGN SYSTEM — Editorial Instrument Aesthetic
// ═══════════════════════════════════════════════════════════════════════════════
// Inspired by Dieter Rams & vintage scientific equipment (1975→2024)
// 
// Color Palette:
//   Background: #F4EFE6 (warm linen/cream)
//   Text primary: #1C1208 (warm near-black ink)
//   Accent/signal: #C13318 (vermillion - like a vintage meter needle)
//   Navy: #1B3858 (deep navy for secondary states)
//   Warm gray: #7A7060 (muted labels/borders)
//   Border: #C8BCA8 (warm parchment border)
//
// Typography:
//   Display: DM Serif Display (italic for headers - editorial tension)
//   Data/Technical: Fragment Mono (precise values)
//   Body: Syne (clean sans-serif)
//
// Layout:
//   Two-panel split: Left (orbital visualizer), Right (tabbed controls)
// ═══════════════════════════════════════════════════════════════════════════════

const MERIDIAN_FONT_STYLE = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;500;600;700&display=swap');
  @import url('https://fonts.cdnfonts.com/css/fragment-mono');

  * { 
    box-sizing: border-box;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  .meridian-display {
    font-family: 'DM Serif Display', serif;
    font-style: italic;
  }

  .meridian-mono {
    font-family: 'Fragment Mono', 'Courier New', monospace;
    font-variant-numeric: tabular-nums;
  }

  .meridian-body {
    font-family: 'Syne', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  /* ── Range Inputs (Vintage Precision Knob Aesthetic) ──────────────── */
  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    cursor: pointer;
    width: 100%;
  }

  input[type=range]::-webkit-slider-runnable-track {
    background: #C8BCA8;
    height: 1px;
    border-radius: 0;
  }

  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    background: #C13318;
    border: 2px solid #F4EFE6;
    box-shadow: 0 0 0 1px #7A7060;
    border-radius: 0;
    margin-top: -6px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }

  input[type=range]::-webkit-slider-thumb:hover {
    background: #E04520;
    transform: scale(1.15);
  }

  input[type=range]::-webkit-slider-thumb:active {
    background: #A02810;
  }

  input[type=range]::-moz-range-track {
    background: #C8BCA8;
    height: 1px;
    border-radius: 0;
  }

  input[type=range]::-moz-range-thumb {
    width: 14px;
    height: 14px;
    background: #C13318;
    border: 2px solid #F4EFE6;
    box-shadow: 0 0 0 1px #7A7060;
    border-radius: 0;
  }

  input[type=range]:focus {
    outline: none;
  }

  input[type=range]:focus::-webkit-slider-thumb {
    box-shadow: 0 0 0 3px rgba(193, 51, 24, 0.25);
  }

  /* ── Tab Buttons (Editorial Precision) ──────────────────────────── */
  .meridian-tab {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 14px 22px;
    border: none;
    background: transparent;
    cursor: pointer;
    transition: all 0.2s;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    position: relative;
  }

  .meridian-tab-active {
    color: #1C1208;
    border-bottom-color: #C13318;
    background: rgba(193, 51, 24, 0.04);
  }

  .meridian-tab-inactive {
    color: #7A7060;
    border-bottom-color: transparent;
  }

  .meridian-tab-inactive:hover {
    color: #1C1208;
    background: rgba(122, 112, 96, 0.06);
  }

  /* ── Primary Button (Vermillion Action) ──────────────────────────── */
  .meridian-btn-primary {
    background: #C13318;
    color: #F4EFE6;
    border: none;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    padding: 12px 24px;
    border-radius: 0;
    text-transform: uppercase;
    box-shadow: 0 2px 8px rgba(193, 51, 24, 0.20);
  }

  .meridian-btn-primary:hover {
    background: #E04520;
    box-shadow: 0 4px 12px rgba(193, 51, 24, 0.30);
    transform: translateY(-1px);
  }

  .meridian-btn-primary:active {
    transform: translateY(0);
    box-shadow: 0 1px 4px rgba(193, 51, 24, 0.25);
  }

  .meridian-btn-primary:disabled {
    background: #C8BCA8;
    color: #7A7060;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
  }

  /* ── Secondary Button (Navy Outline) ──────────────────────────── */
  .meridian-btn-secondary {
    background: transparent;
    color: #1B3858;
    border: 1px solid #C8BCA8;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.2s;
    padding: 9px 18px;
    border-radius: 0;
    text-transform: uppercase;
  }

  .meridian-btn-secondary:hover {
    border-color: #1B3858;
    background: rgba(27, 56, 88, 0.04);
  }

  .meridian-btn-selected {
    background: rgba(193, 51, 24, 0.08);
    border-color: #C13318;
    color: #C13318;
  }

  /* ── Toggle Button (Instrument Switch) ──────────────────────────── */
  .meridian-toggle {
    background: #FEFDFB;
    border: 1px solid #C8BCA8;
    color: #7A7060;
    font-family: 'Fragment Mono', monospace;
    font-size: 10px;
    cursor: pointer;
    transition: all 0.2s;
    padding: 7px 14px;
    border-radius: 0;
    white-space: nowrap;
    letter-spacing: 0.02em;
  }

  .meridian-toggle:hover {
    border-color: #1B3858;
    color: #1B3858;
  }

  .meridian-toggle-on {
    background: rgba(193, 51, 24, 0.10);
    border-color: #C13318;
    color: #C13318;
    font-weight: 500;
  }

  /* ── EQ Visualizer (Vintage Meter) ──────────────────────────── */
  .meridian-eq-container {
    display: flex;
    align-items: stretch;
    gap: 3px;
    height: 72px;
    padding: 6px 8px;
    background: #FEFDFB;
    border: 1px solid #C8BCA8;
    border-radius: 0;
    margin-bottom: 20px;
    overflow: hidden;
    position: relative;
  }

  .meridian-eq-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
  }

  .meridian-eq-bar-pos {
    position: absolute;
    bottom: 50%;
    width: 100%;
    background: linear-gradient(to top, #C13318, #E04520);
    opacity: 0.85;
    transition: height 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    border-radius: 0;
  }

  .meridian-eq-bar-neg {
    position: absolute;
    top: 50%;
    width: 100%;
    background: linear-gradient(to bottom, #1B3858, #2A4A6A);
    opacity: 0.70;
    transition: height 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    border-radius: 0;
  }

  .meridian-eq-zero {
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: #7A7060;
    pointer-events: none;
  }

  /* ── Analysis Tag (Telemetry Badge) ──────────────────────────── */
  .meridian-analysis-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    background: #FEFDFB;
    border: 1px solid #C8BCA8;
    border-radius: 0;
    font-family: 'Fragment Mono', monospace;
    font-size: 10px;
    color: #7A7060;
  }

  .meridian-analysis-tag span:last-child {
    color: #C13318;
    font-weight: 500;
  }

  /* ── LED Indicator (Signal Dot) ──────────────────────────── */
  .meridian-led {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.08);
  }

  .meridian-led-green  { background: #2A7A3A; box-shadow: 0 0 6px rgba(42,122,58,0.6); }
  .meridian-led-red    { background: #C13318; box-shadow: 0 0 6px rgba(193,51,24,0.5); }
  .meridian-led-amber  { background: #D4905A; box-shadow: 0 0 6px rgba(212,144,90,0.5); }
  .meridian-led-gold   { background: #D4A650; box-shadow: 0 0 8px rgba(212,166,80,0.7); }

  /* ── Section Label (Engineering Notation) ──────────────────────────── */
  .meridian-section-label {
    font-family: 'Syne', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7A7060;
  }

  /* ── Value Badge (Measurement Display) ──────────────────────────── */
  .meridian-value {
    font-family: 'Fragment Mono', monospace;
    font-size: 11px;
    color: #C13318;
    background: rgba(193, 51, 24, 0.06);
    border: 1px solid rgba(193, 51, 24, 0.15);
    padding: 2px 8px;
    border-radius: 0;
    letter-spacing: 0.03em;
  }

  /* ── Pipeline Step (Processing Stage Indicator) ──────────────────────────── */
  .meridian-pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 12px 14px;
    background: #FEFDFB;
    border: 1px solid #C8BCA8;
    border-radius: 0;
    transition: all 0.3s;
  }

  .meridian-pipeline-step.active {
    background: rgba(193, 51, 24, 0.04);
    border-color: #C13318;
    box-shadow: 0 0 0 1px rgba(193, 51, 24, 0.1);
  }

  .meridian-pipeline-step.done {
    background: rgba(42, 122, 58, 0.03);
    border-color: #2A7A3A;
  }

  .meridian-step-num {
    font-family: 'Fragment Mono', monospace;
    font-size: 11px;
    color: #C8BCA8;
    min-width: 22px;
    padding-top: 1px;
  }

  .meridian-step-num.active { color: #C13318; font-weight: 600; }
  .meridian-step-num.done   { color: #2A7A3A; font-weight: 500; }

  /* ── Progress Bar (Linear Meter) ──────────────────────────── */
  .meridian-progress-track {
    width: 100%;
    height: 4px;
    background: #E8E0D2;
    border-radius: 0;
    overflow: hidden;
    margin-top: 6px;
    border: 1px solid #C8BCA8;
  }

  .meridian-progress-fill {
    height: 100%;
    background: linear-gradient(to right, #C13318, #E04520);
    border-radius: 0;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* ── Text Inputs (Precision Field) ──────────────────────────── */
  input[type=text], input[type=url] {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    background: #FEFDFB;
    border: 1px solid #C8BCA8;
    border-radius: 0;
    color: #1C1208;
    padding: 11px 14px;
    transition: all 0.2s;
    outline: none;
    width: 100%;
  }

  input[type=text]:focus, input[type=url]:focus {
    border-color: #1B3858;
    box-shadow: 0 0 0 3px rgba(27, 56, 88, 0.08);
  }

  input[type=text]::placeholder, input[type=url]::placeholder {
    color: #C8BCA8;
  }

  /* ── Scroll Styling ──────────────────────────── */
  .meridian-scroll::-webkit-scrollbar { width: 5px; }
  .meridian-scroll::-webkit-scrollbar-track { background: #F4EFE6; }
  .meridian-scroll::-webkit-scrollbar-thumb { background: #C8BCA8; border-radius: 0; }
  .meridian-scroll::-webkit-scrollbar-thumb:hover { background: #7A7060; }
`;

// ═══════════════════════════════════════════════════════════════════════════════
// SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const SliderRow = ({ label, value, min, max, step, onChange, unit = '', hint = '' }) => {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span className="meridian-body" style={{ fontSize: 12, color: '#7A7060', letterSpacing: '0.03em', fontWeight: 500 }}>
          {label}
        </span>
        <span className="meridian-value meridian-mono">
          {typeof value === 'number' && value % 1 !== 0 ? value.toFixed(2) : value}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))} />
      {hint && (
        <p className="meridian-body" style={{ fontSize: 10, color: '#7A7060', marginTop: 6, marginBottom: 0, fontStyle: 'italic' }}>
          {hint}
        </p>
      )}
    </div>
  );
};

const SectionDivider = ({ label }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '24px 0 16px' }}>
    <span className="meridian-section-label">{label}</span>
    <div style={{ flex: 1, height: 1, background: '#C8BCA8' }} />
  </div>
);

const EQVisualizer = ({ bands }) => {
  const maxGain = 14;
  return (
    <div className="meridian-eq-container">
      <div className="meridian-eq-zero" />
      {bands.map(({ freq, gain }) => {
        const barH = Math.abs(gain / maxGain) * 32;
        return (
          <div key={freq} className="meridian-eq-col">
            {gain >= 0 ? (
              <div className="meridian-eq-bar-pos" style={{ height: barH }} />
            ) : (
              <div className="meridian-eq-bar-neg" style={{ height: barH }} />
            )}
          </div>
        );
      })}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════════
// ORBITAL RING VISUALIZER — Real-time animation at rotationSpeed Hz
// ═══════════════════════════════════════════════════════════════════════════════

interface OrbitalVisualizerProps {
  rotationSpeed: number;    // Hz
  isProcessing: boolean;
  isComplete: boolean;
  progress: number;         // 0-100
}

const OrbitalVisualizer: React.FC<OrbitalVisualizerProps> = ({
  rotationSpeed,
  isProcessing,
  isComplete,
  progress
}) => {
  const [rotation, setRotation] = useState(0);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(Date.now());

  useEffect(() => {
    const animate = () => {
      const now = Date.now();
      const delta = (now - lastTimeRef.current) / 1000; // seconds
      lastTimeRef.current = now;

      // Rotate at rotationSpeed Hz (full rotation = 360 degrees)
      const degreesPerSecond = rotationSpeed * 360;
      setRotation(prev => (prev + degreesPerSecond * delta) % 360);

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [rotationSpeed]);

  const numDots = 8;
  const radius = 110;
  const centerX = 180;
  const centerY = 180;

  const dotColor = isComplete ? '#D4A650' : (isProcessing ? '#C13318' : '#7A7060');
  const dotGlow = isComplete ? 'rgba(212, 166, 80, 0.5)' : (isProcessing ? 'rgba(193, 51, 24, 0.5)' : 'rgba(122, 112, 96, 0.2)');

  return (
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#FEFDFB',
      borderRight: '1px solid #C8BCA8',
      position: 'relative',
    }}>
      {/* Title */}
      <div style={{ position: 'absolute', top: 32, left: 32, right: 32, textAlign: 'center' }}>
        <h2 className="meridian-display" style={{ fontSize: 28, color: '#1C1208', margin: '0 0 8px', lineHeight: 1.1 }}>
          Spatial Field Monitor
        </h2>
        <p className="meridian-mono" style={{ fontSize: 11, color: '#7A7060', margin: 0, letterSpacing: '0.08em' }}>
          ROTATION: {rotationSpeed.toFixed(2)} Hz · {(rotationSpeed * 60).toFixed(1)} RPM
        </p>
      </div>

      {/* SVG Orbital Ring */}
      <svg width="360" height="360" viewBox="0 0 360 360" style={{ maxWidth: '100%', height: 'auto' }}>
        {/* Outer frame ring */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius + 20}
          fill="none"
          stroke="#E8E0D2"
          strokeWidth="1"
        />
        
        {/* Inner orbital path */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius}
          fill="none"
          stroke="#C8BCA8"
          strokeWidth="1"
          strokeDasharray="4 4"
        />

        {/* Center crosshair */}
        <line x1={centerX - 8} y1={centerY} x2={centerX + 8} y2={centerY} stroke="#7A7060" strokeWidth="1" />
        <line x1={centerX} y1={centerY - 8} x2={centerX} y2={centerY + 8} stroke="#7A7060" strokeWidth="1" />
        <circle cx={centerX} cy={centerY} r="3" fill="#7A7060" />

        {/* Orbiting dots */}
        {Array.from({ length: numDots }).map((_, i) => {
          const angle = (rotation + (i * 360 / numDots)) * (Math.PI / 180);
          const x = centerX + radius * Math.cos(angle);
          const y = centerY + radius * Math.sin(angle);

          return (
            <g key={i}>
              {/* Glow effect (only when processing or complete) */}
              {(isProcessing || isComplete) && (
                <circle
                  cx={x}
                  cy={y}
                  r="12"
                  fill={dotGlow}
                  opacity="0.4"
                />
              )}
              
              {/* Dot */}
              <circle
                cx={x}
                cy={y}
                r="5"
                fill={dotColor}
                stroke="#F4EFE6"
                strokeWidth="1.5"
              />

              {/* Motion trail (only when processing) */}
              {isProcessing && (
                <circle
                  cx={x}
                  cy={y}
                  r="8"
                  fill="none"
                  stroke={dotColor}
                  strokeWidth="1"
                  opacity="0.3"
                />
              )}
            </g>
          );
        })}

        {/* Progress arc (when processing) */}
        {isProcessing && progress > 0 && (
          <path
            d={describeArc(centerX, centerY, radius + 30, 0, (progress / 100) * 360)}
            fill="none"
            stroke="#C13318"
            strokeWidth="2"
            strokeLinecap="round"
          />
        )}
      </svg>

      {/* Status indicator */}
      <div style={{ position: 'absolute', bottom: 32, textAlign: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center' }}>
          <span className={`meridian-led ${
            isComplete ? 'meridian-led-gold' : 
            isProcessing ? 'meridian-led-red' : 
            'meridian-led-amber'
          }`} />
          <span className="meridian-mono" style={{ fontSize: 10, color: '#7A7060', letterSpacing: '0.06em' }}>
            {isComplete ? 'COMPLETE' : isProcessing ? 'PROCESSING' : 'STANDBY'}
          </span>
        </div>
        {isProcessing && (
          <p className="meridian-mono" style={{ fontSize: 11, color: '#C13318', margin: '8px 0 0', fontWeight: 500 }}>
            {progress}%
          </p>
        )}
      </div>
    </div>
  );
};

// Helper function to draw SVG arc
function describeArc(x: number, y: number, radius: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(x, y, radius, endAngle);
  const end = polarToCartesian(x, y, radius, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return [
    "M", start.x, start.y,
    "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y
  ].join(" ");
}

function polarToCartesian(centerX: number, centerY: number, radius: number, angleInDegrees: number) {
  const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
  return {
    x: centerX + (radius * Math.cos(angleInRadians)),
    y: centerY + (radius * Math.sin(angleInRadians))
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

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
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
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

  // Spatial Output
  const [spatialFormat, setSpatialFormat] = useState('stereo');

  // Stem Separation
  const [enableStemSeparation, setEnableStemSeparation] = useState(false);
  const [stemVocalsRotation, setStemVocalsRotation] = useState(null);
  const [stemDrumsRotation, setStemDrumsRotation] = useState(null);
  const [stemBassRotation, setStemBassRotation] = useState(null);
  const [stemOtherRotation, setStemOtherRotation] = useState(null);
  const [stemPreset, setStemPreset] = useState('auto');
  // Per-stem width overrides
  const [stemVocalsWidth, setStemVocalsWidth] = useState(null);
  const [stemDrumsWidth, setStemDrumsWidth] = useState(null);
  const [stemBassWidth, setStemBassWidth] = useState(null);
  const [stemOtherWidth, setStemOtherWidth] = useState(null);
  // Per-stem elevation overrides
  const [stemVocalsElevation, setStemVocalsElevation] = useState(null);
  const [stemDrumsElevation, setStemDrumsElevation] = useState(null);
  const [stemBassElevation, setStemBassElevation] = useState(null);
  const [stemOtherElevation, setStemOtherElevation] = useState(null);
  // Per-stem reverb overrides
  const [stemVocalsReverb, setStemVocalsReverb] = useState(null);
  const [stemDrumsReverb, setStemDrumsReverb] = useState(null);
  const [stemBassReverb, setStemBassReverb] = useState(null);
  const [stemOtherReverb, setStemOtherReverb] = useState(null);
  // Guitar/Piano (6-stem model)
  const [stemGuitarRotation, setStemGuitarRotation] = useState(null);
  const [stemGuitarWidth, setStemGuitarWidth] = useState(null);
  const [stemPianoRotation, setStemPianoRotation] = useState(null);
  const [stemPianoWidth, setStemPianoWidth] = useState(null);
  const [stemModel, setStemModel] = useState('htdemucs');

  // Video Visualizer
  const [generateVideo, setGenerateVideo] = useState(false);
  const [videoStyle, setVideoStyle] = useState('waveform');
  const [videoResolution, setVideoResolution] = useState('1280x720');

  // Backend
  const [backendStatus, setBackendStatus] = useState('checking');
  const wsRef = useRef<WebSocket | null>(null);
  const [audioAnalysis, setAudioAnalysis] = useState(null);
  const [reverbEngine, setReverbEngine] = useState('aecho');
  const [backendCaps, setBackendCaps] = useState({
    stem_separation: false,
    stem_engine: 'none',
    ambisonics_foa: false,
    atmos_71_4: false,
    video_visualizer: false
  });

  // 12-band EQ
  const [eqSub30Gain, setEqSub30Gain] = useState(3.0);
  const [eqSub60Gain, setEqSub60Gain] = useState(4.0);
  const [eqBass100Gain, setEqBass100Gain] = useState(3.0);
  const [eqUbass200Gain, setEqUbass200Gain] = useState(1.5);
  const [eqLowmid350Gain, setEqLowmid350Gain] = useState(-2.5);
  const [eqMid700Gain, setEqMid700Gain] = useState(-1.0);
  const [eqUmid1500Gain, setEqUmid1500Gain] = useState(1.0);
  const [eqPresence3kGain, setEqPresence3kGain] = useState(2.0);
  const [eqDef5kGain, setEqDef5kGain] = useState(1.5);
  const [eqBril8kGain, setEqBril8kGain] = useState(2.0);
  const [eqAir12kGain, setEqAir12kGain] = useState(2.0);
  const [eqUair16kGain, setEqUair16kGain] = useState(1.0);

  // Enhancement toggles
  const [enableVocalCenter, setEnableVocalCenter] = useState(false);
  const [vocalSafeBass, setVocalSafeBass] = useState(true);
  const [instrumentEnhance, setInstrumentEnhance] = useState(true);

  // Studio controls
  const [reverbDensity, setReverbDensity] = useState(0.7);
  const [hrtfIntensity, setHrtfIntensity] = useState(1.0);

  const fileInputRef = useRef(null);
  const batchFileInputRef = useRef(null);
  const audioElementRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    checkBackendHealth();
    const iv = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(iv);
  }, []);

  // BUG FIX: wsRef was never cleaned up on unmount. If the user navigated away
  // mid-processing, the WebSocket stayed open and its onmessage callback fired
  // state setters on an unmounted component (React warning + potential memory leak).
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  // BUG FIX (smaller): object URLs created by URL.createObjectURL are never
  // revoked, leaking memory for every file the user loads. Track the current
  // object URL and revoke it when a new file is loaded or the component unmounts.
  const objectUrlRef = useRef<string | null>(null);
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  const checkBackendHealth = async () => {
    try {
      const r = await fetch('http://localhost:8000/health');
      if (r.ok) {
        const d = await r.json();
        setBackendStatus('connected');
        setReverbEngine(d.reverb_engine || 'aecho');
        setBackendCaps({
          stem_separation: d.stem_separation || false,
          stem_engine: d.stem_engine || 'none',
          ambisonics_foa: d.ambisonics_foa || false,
          atmos_71_4: d.atmos_71_4 || false,
          video_visualizer: d.video_visualizer || false,
        });
      } else {
        setBackendStatus('error');
      }
    } catch {
      setBackendStatus('error');
    }
  };

  const connectWebSocket = (jobId) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);
    wsRef.current = ws;
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        setProgress(data.progress);
        setProcessingStage(data.stage);
      } else if (data.type === 'complete') {
        setProgress(100);
        setProcessedAudio(data.output_url);
        if (data.video_url) setProcessedVideoUrl(data.video_url);
        setActiveTab('output');
        setIsProcessing(false);
        ws.close();
        wsRef.current = null;
      } else if (data.type === 'error') {
        alert(`Processing error: ${data.message}`);
        setIsProcessing(false);
        ws.close();
        wsRef.current = null;
      }
    };
    ws.onerror = () => { setIsProcessing(false); wsRef.current = null; };
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAudioFile(file);
    // Revoke the previous object URL to free memory before creating a new one
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
    }
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setOriginalAudio(url);
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

  const buildParams = () => {
    let actualOutputFormat = outputFormat;
    if (spatialFormat === 'ambisonics_foa') actualOutputFormat = 'ambisonics_foa';
    else if (spatialFormat === 'atmos_71_4') actualOutputFormat = 'atmos_71_4';

    let svr = stemVocalsRotation, sdr = stemDrumsRotation;
    let sbr = stemBassRotation, sor = stemOtherRotation;
    if (stemPreset === 'vocals_front') {
      svr = rotationSpeed * 0.5; sdr = rotationSpeed * 1.3;
      sbr = bassRotation; sor = rotationSpeed * 0.9;
    } else if (stemPreset === 'drums_wide') {
      svr = rotationSpeed * 0.6; sdr = trebleRotation * 1.5;
      sbr = bassRotation * 0.5; sor = rotationSpeed;
    }

    return {
      rotation_speed: rotationSpeed,
      reverb_room: reverbRoom,
      reverb_mix: reverbMix,
      bass_rotation: bassRotation,
      treble_rotation: trebleRotation,
      stereo_width: stereoWidth,
      elevation,
      distance,
      output_format: actualOutputFormat,
      bitrate,
      enable_hrtf: true,
      enable_convolution_reverb: true,
      enable_multi_band: true,
      sample_rate: 48000,
      bit_depth: 24,
      enable_vocal_center: enableVocalCenter,
      vocal_safe_bass: vocalSafeBass,
      instrument_enhance: instrumentEnhance,
      eq_sub30_gain: eqSub30Gain,
      eq_sub60_gain: eqSub60Gain,
      eq_bass100_gain: eqBass100Gain,
      eq_ubass200_gain: eqUbass200Gain,
      eq_lowmid350_gain: eqLowmid350Gain,
      eq_mid700_gain: eqMid700Gain,
      eq_umid1500_gain: eqUmid1500Gain,
      eq_presence3k_gain: eqPresence3kGain,
      eq_def5k_gain: eqDef5kGain,
      eq_bril8k_gain: eqBril8kGain,
      eq_air12k_gain: eqAir12kGain,
      eq_uair16k_gain: eqUair16kGain,
      eq_sub_bass_gain: 0,
      eq_bass_gain: 0,
      eq_low_mid_gain: 0,
      eq_presence_gain: 0,
      eq_air_gain: 0,
      reverb_density: reverbDensity,
      hrtf_intensity: hrtfIntensity,
      enable_limiter: true,
      enable_stem_separation: enableStemSeparation,
      stem_engine_model: stemModel,  // BUG FIX: was "stem_model" — backend field is "stem_engine_model";
                                     // Pydantic silently dropped the unknown key so htdemucs_6s was never used
      stem_vocals_rotation: svr,
      stem_drums_rotation: sdr,
      stem_bass_rotation_override: sbr,
      stem_other_rotation: sor,
      // Per-stem width overrides
      stem_vocals_width: stemVocalsWidth,
      stem_drums_width: stemDrumsWidth,
      stem_bass_width: stemBassWidth,
      stem_other_width: stemOtherWidth,
      // Per-stem elevation overrides
      stem_vocals_elevation: stemVocalsElevation,
      stem_drums_elevation: stemDrumsElevation,
      stem_bass_elevation: stemBassElevation,
      stem_other_elevation: stemOtherElevation,
      // Per-stem reverb overrides
      stem_vocals_reverb: stemVocalsReverb,
      stem_drums_reverb: stemDrumsReverb,
      stem_bass_reverb: stemBassReverb,
      stem_other_reverb: stemOtherReverb,
      // Guitar/Piano (6-stem model)
      stem_guitar_rotation: stemGuitarRotation,
      stem_guitar_width: stemGuitarWidth,
      stem_piano_rotation: stemPianoRotation,
      stem_piano_width: stemPianoWidth,
      generate_video: generateVideo,
      video_style: videoStyle,
      video_resolution: videoResolution,
      video_fps: 25,
    };
  };

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
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
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
        // BUG FIX: analyzeAudio expects a File/Blob — passing the raw URL string
        // caused fd.append('file', urlString) to create a text FormData field,
        // not a file upload, so /analyze received an empty or malformed file and
        // analysis tags never populated for YouTube tracks.
        // Solution: fetch the audio, wrap it as a Blob, then pass to analyzeAudio.
        try {
          const audioResp = await fetch(d.audio_url);
          if (audioResp.ok) {
            const audioBlob = await audioResp.blob();
            const audioFile = new File([audioBlob], d.title + '.mp3', { type: 'audio/mpeg' });
            await analyzeAudio(audioFile);
          }
        } catch {
          // Analysis is best-effort; non-fatal if the fetch fails
        }
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
      subtle: { rotation: 0.08, reverb: 0.3, mix: 0.2, bass: 0.05, treble: 0.12, width: 0.8, elevation: 0, distance: 1.0, density: 0.6, hrtf: 0.7 },
      classic: { rotation: 0.15, reverb: 0.6, mix: 0.3, bass: 0.08, treble: 0.2, width: 1.0, elevation: 0, distance: 1.0, density: 0.7, hrtf: 1.0 },
      intense: { rotation: 0.3, reverb: 0.8, mix: 0.4, bass: 0.15, treble: 0.4, width: 1.2, elevation: 0.1, distance: 0.8, density: 0.85, hrtf: 1.2 },
      cinematic: { rotation: 0.12, reverb: 0.85, mix: 0.45, bass: 0.06, treble: 0.18, width: 1.3, elevation: 0.15, distance: 1.2, density: 0.9, hrtf: 1.1 },
      experimental: { rotation: 0.5, reverb: 0.9, mix: 0.5, bass: 0.25, treble: 0.6, width: 1.5, elevation: 0.2, distance: 0.6, density: 0.95, hrtf: 1.5 },
    };
    const p = presets[preset];
    if (!p) return;
    setRotationSpeed(p.rotation);
    setReverbRoom(p.reverb);
    setReverbMix(p.mix);
    setBassRotation(p.bass);
    setTrebleRotation(p.treble);
    setStereoWidth(p.width);
    setElevation(p.elevation);
    setDistance(p.distance);
    setReverbDensity(p.density);
    setHrtfIntensity(p.hrtf);
  };

  const applyRecommendedSettings = (s) => {
    if (s.rotation_speed != null) setRotationSpeed(s.rotation_speed);
    if (s.reverb_room != null) setReverbRoom(s.reverb_room);
    if (s.reverb_mix != null) setReverbMix(s.reverb_mix);
    if (s.bass_rotation != null) setBassRotation(s.bass_rotation);
    if (s.treble_rotation != null) setTrebleRotation(s.treble_rotation);
    if (s.stereo_width != null) setStereoWidth(s.stereo_width);
    if (s.elevation != null) setElevation(s.elevation);
    if (s.distance != null) setDistance(s.distance);
    if (s.enable_vocal_center != null) setEnableVocalCenter(s.enable_vocal_center);
    if (s.vocal_safe_bass != null) setVocalSafeBass(s.vocal_safe_bass);
    if (s.instrument_enhance != null) setInstrumentEnhance(s.instrument_enhance);
    if (s.eq_sub30_gain != null) setEqSub30Gain(s.eq_sub30_gain);
    if (s.eq_sub60_gain != null) setEqSub60Gain(s.eq_sub60_gain);
    if (s.eq_bass100_gain != null) setEqBass100Gain(s.eq_bass100_gain);
    if (s.eq_ubass200_gain != null) setEqUbass200Gain(s.eq_ubass200_gain);
    if (s.eq_lowmid350_gain != null) setEqLowmid350Gain(s.eq_lowmid350_gain);
    if (s.eq_mid700_gain != null) setEqMid700Gain(s.eq_mid700_gain);
    if (s.eq_umid1500_gain != null) setEqUmid1500Gain(s.eq_umid1500_gain);
    if (s.eq_presence3k_gain != null) setEqPresence3kGain(s.eq_presence3k_gain);
    if (s.eq_def5k_gain != null) setEqDef5kGain(s.eq_def5k_gain);
    if (s.eq_bril8k_gain != null) setEqBril8kGain(s.eq_bril8k_gain);
    if (s.eq_air12k_gain != null) setEqAir12kGain(s.eq_air12k_gain);
    if (s.eq_uair16k_gain != null) setEqUair16kGain(s.eq_uair16k_gain);
    if (s.reverb_density != null) setReverbDensity(s.reverb_density);
    if (s.hrtf_intensity != null) setHrtfIntensity(s.hrtf_intensity);
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
    { freq: '30', gain: eqSub30Gain },
    { freq: '60', gain: eqSub60Gain },
    { freq: '100', gain: eqBass100Gain },
    { freq: '200', gain: eqUbass200Gain },
    { freq: '350', gain: eqLowmid350Gain },
    { freq: '700', gain: eqMid700Gain },
    { freq: '1.5k', gain: eqUmid1500Gain },
    { freq: '3k', gain: eqPresence3kGain },
    { freq: '5k', gain: eqDef5kGain },
    { freq: '8k', gain: eqBril8kGain },
    { freq: '12k', gain: eqAir12kGain },
    { freq: '16k', gain: eqUair16kGain },
  ];

  const pipelineStages = [
    { label: 'Deep Analysis', desc: 'MFCC · key detection · crest factor · stereo correlation', threshold: 12 },
    ...(enableStemSeparation ? [{ label: 'Stem Separation', desc: 'Demucs: vocals · drums · bass · other', threshold: 18 }] : []),
    { label: 'Frequency Splitting', desc: spatialFormat !== 'stereo' ? `${spatialFormat === 'ambisonics_foa' ? 'Ambisonics FOA encoder' : 'Atmos 7.1.4 bed encoder'}` : '8-band HRTF separation', threshold: enableStemSeparation ? 30 : 28 },
    { label: 'Spatial Panning', desc: 'ITD inter-ear delay · LFO volume automation', threshold: enableStemSeparation ? 52 : 48 },
    { label: 'Pinna & Head Shadow', desc: 'Notch EQ at 8.5/10.5/13 kHz · head shadowing', threshold: enableStemSeparation ? 64 : 62 },
    { label: 'Reverb Engine', desc: 'Allpass diffusion → pre-delay → room reverb', threshold: enableStemSeparation ? 76 : 78 },
    { label: 'Mastering Chain', desc: 'Diffuse-field EQ · equal-loudness · 12-band EQ · limiter', threshold: enableStemSeparation ? 88 : 90 },
    ...(generateVideo ? [{ label: 'Video Render', desc: `${videoStyle} visualizer · ${videoResolution}`, threshold: 94 }] : []),
  ];

  // ═══════════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════════

  return (
    <>
      <style>{MERIDIAN_FONT_STYLE}</style>
      <div className="meridian-body" style={{
        minHeight: '100vh',
        background: '#F4EFE6',
        color: '#1C1208',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {/* ═══ HEADER ═══ */}
        <div style={{
          borderBottom: '1px solid #C8BCA8',
          padding: '20px 32px',
          display: 'flex',
          alignItems: 'center',
          gap: 24,
          background: '#FEFDFB',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 7,
              height: 7,
              background: '#C13318',
              borderRadius: '50%',
              boxShadow: '0 0 8px rgba(193,51,24,0.6)'
            }} />
            <span className="meridian-display" style={{
              fontSize: 18,
              color: '#1C1208',
            }}>
              8D Audio Converter
            </span>
            <span className="meridian-mono" style={{
              fontSize: 9,
              color: '#7A7060',
              letterSpacing: '0.1em',
              marginLeft: 8,
            }}>
              MERIDIAN
            </span>
          </div>

          <div style={{ flex: 1 }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span className={`meridian-led ${
                backendStatus === 'connected' ? 'meridian-led-green' :
                backendStatus === 'checking' ? 'meridian-led-amber' : 'meridian-led-red'
              }`} />
              <span className="meridian-mono" style={{ fontSize: 9, color: '#7A7060', letterSpacing: '0.08em' }}>
                {backendStatus === 'connected' ? 'ONLINE' :
                 backendStatus === 'checking' ? 'CONNECTING' : 'OFFLINE'}
              </span>
            </div>
            {backendStatus === 'connected' && (
              <span className="meridian-mono" style={{ fontSize: 9, color: '#7A7060', letterSpacing: '0.05em' }}>
                {reverbEngine === 'reverberate' ? 'HD-REV' : 'AECHO'} · v7.0
                {backendCaps.stem_separation && ' · STEMS'}
                {spatialFormat === 'ambisonics_foa' && ' · AMBI'}
                {spatialFormat === 'atmos_71_4' && ' · ATMOS'}
              </span>
            )}
          </div>
        </div>

        {/* ═══ BACKEND OFFLINE BANNER ═══ */}
        {backendStatus !== 'connected' && backendStatus !== 'checking' && (
          <div style={{
            margin: '20px 32px 0',
            padding: '14px 18px',
            background: 'rgba(193, 51, 24, 0.08)',
            border: '1px solid rgba(193, 51, 24, 0.3)',
            borderRadius: 0,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 12,
          }}>
            <div style={{ marginTop: 2 }}>
              <span className="meridian-led meridian-led-red" />
            </div>
            <div>
              <p className="meridian-body" style={{ fontSize: 13, fontWeight: 600, color: '#C13318', margin: '0 0 6px' }}>
                Backend not running
              </p>
              <p className="meridian-mono" style={{ fontSize: 10, color: '#7A7060', margin: 0 }}>
                Start the Python backend to enable processing →{' '}
                <code style={{ fontSize: 10, color: '#1B3858' }}>python backend.py</code>
              </p>
            </div>
          </div>
        )}

        {/* ═══ TWO-PANEL LAYOUT ═══ */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* LEFT PANEL: Orbital Visualizer */}
          <div style={{ width: '42%', minWidth: 420, position: 'relative' }}>
            <OrbitalVisualizer
              rotationSpeed={rotationSpeed}
              isProcessing={isProcessing}
              isComplete={processedAudio !== null && !isProcessing}
              progress={progress}
            />
          </div>

          {/* RIGHT PANEL: Tabbed Controls */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#FEFDFB' }}>
            {/* Tab Bar */}
            <div style={{
              display: 'flex',
              borderBottom: '1px solid #C8BCA8',
              overflowX: 'auto',
              background: '#F4EFE6',
            }}>
              {[
                { id: 'upload', label: 'Source' },
                { id: 'controls', label: 'Parameters' },
                { id: 'spatial', label: 'Spatial' },
                { id: 'stems', label: 'Stems' },
                { id: 'visualizer', label: 'Visualizer' },
                { id: 'output', label: 'Output' },
              ].map(tab => (
                <button
                  key={tab.id}
                  className={`meridian-tab ${activeTab === tab.id ? 'meridian-tab-active' : 'meridian-tab-inactive'}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="meridian-scroll" style={{
              flex: 1,
              overflowY: 'auto',
              padding: '32px',
            }}>
              {/* ═══ UPLOAD TAB ═══ */}
              {activeTab === 'upload' && (
                <div style={{ maxWidth: 600 }}>
                  <h3 className="meridian-display" style={{ fontSize: 24, margin: '0 0 24px', color: '#1C1208' }}>
                    Audio Source
                  </h3>

                  {/* Mode toggle */}
                  <div style={{ display: 'flex', gap: 10, marginBottom: 28 }}>
                    <button
                      className={`meridian-btn-secondary ${!isBatchMode ? 'meridian-btn-selected' : ''}`}
                      onClick={() => { setIsBatchMode(false); setBatchFiles([]); }}
                    >
                      Single file
                    </button>
                    <button
                      className={`meridian-btn-secondary ${isBatchMode ? 'meridian-btn-selected' : ''}`}
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
                        onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                        onDragLeave={() => setIsDragging(false)}
                        onDrop={e => {
                          e.preventDefault();
                          setIsDragging(false);
                          const file = e.dataTransfer.files?.[0];
                          if (file && (file.type.startsWith('audio/') || /\.(mp3|wav|m4a|flac|ogg)$/i.test(file.name))) {
                            setAudioFile(file);
                            setOriginalAudio(URL.createObjectURL(file));
                            analyzeAudio(file);
                            setActiveTab('controls');
                          }
                        }}
                        style={{
                          border: `2px dashed ${isDragging ? '#C13318' : '#C8BCA8'}`,
                          borderRadius: 0,
                          padding: '56px 32px',
                          textAlign: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                          background: isDragging ? 'rgba(193, 51, 24, 0.04)' : '#F9F6F0',
                        }}
                        onMouseEnter={e => {
                          if (!isDragging) {
                            e.currentTarget.style.borderColor = '#C13318';
                            e.currentTarget.style.background = 'rgba(193, 51, 24, 0.02)';
                          }
                        }}
                        onMouseLeave={e => {
                          if (!isDragging) {
                            e.currentTarget.style.borderColor = '#C8BCA8';
                            e.currentTarget.style.background = '#F9F6F0';
                          }
                        }}
                      >
                        <p className="meridian-body" style={{ fontSize: 15, color: isDragging ? '#C13318' : '#1C1208', margin: '0 0 8px', fontWeight: 600 }}>
                          {isDragging ? 'Drop to load' : 'Drop audio file here'}
                        </p>
                        <p className="meridian-mono" style={{ fontSize: 10, color: '#7A7060', margin: 0, letterSpacing: '0.06em' }}>
                          MP3 · WAV · M4A · FLAC · OGG
                        </p>
                        <input ref={fileInputRef} type="file" accept="audio/*"
                          onChange={handleFileUpload} style={{ display: 'none' }} />
                      </div>

                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 14,
                        margin: '24px 0',
                        color: '#7A7060'
                      }}>
                        <div style={{ flex: 1, height: 1, background: '#C8BCA8' }} />
                        <span className="meridian-mono" style={{ fontSize: 9, color: '#7A7060', letterSpacing: '0.12em' }}>OR</span>
                        <div style={{ flex: 1, height: 1, background: '#C8BCA8' }} />
                      </div>

                      {/* YouTube */}
                      <div>
                        <p className="meridian-section-label" style={{ margin: '0 0 10px' }}>
                          YouTube URL
                        </p>
                        <div style={{ display: 'flex', gap: 10 }}>
                          <input
                            type="text"
                            placeholder="https://youtube.com/watch?v=..."
                            value={youtubeUrl}
                            onChange={e => setYoutubeUrl(e.target.value)}
                            style={{ flex: 1 }}
                          />
                          <button
                            className="meridian-btn-primary"
                            onClick={handleYoutubeDownload}
                            disabled={!youtubeUrl || backendStatus !== 'connected'}
                          >
                            Fetch
                          </button>
                        </div>
                        <p className="meridian-body" style={{ fontSize: 11, color: '#7A7060', margin: '8px 0 0', fontStyle: 'italic' }}>
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
                          border: '2px dashed #C8BCA8',
                          borderRadius: 0,
                          padding: '48px 32px',
                          textAlign: 'center',
                          cursor: 'pointer',
                          background: '#F9F6F0',
                        }}
                      >
                        <p className="meridian-body" style={{ fontSize: 15, color: '#1C1208', margin: '0 0 8px', fontWeight: 600 }}>
                          Select multiple audio files
                        </p>
                        <p className="meridian-mono" style={{ fontSize: 10, color: '#7A7060', margin: 0 }}>
                          All files processed with shared settings
                        </p>
                        <input ref={batchFileInputRef} type="file" accept="audio/*" multiple
                          onChange={handleBatchFileUpload} style={{ display: 'none' }} />
                      </div>

                      {batchFiles.length > 0 && (
                        <div style={{ marginTop: 20 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                            <span className="meridian-body" style={{ fontSize: 13, color: '#1C1208', fontWeight: 500 }}>
                              {batchFiles.length} file{batchFiles.length !== 1 ? 's' : ''} selected
                            </span>
                            <button className="meridian-btn-secondary" style={{ fontSize: 11 }}
                              onClick={() => { setBatchFiles([]); setBatchStatus(null); }}>
                              Clear
                            </button>
                          </div>
                          <div className="meridian-scroll" style={{ maxHeight: 220, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {batchFiles.map((f, i) => (
                              <div key={i} style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                padding: '10px 12px',
                                background: '#F9F6F0',
                                border: '1px solid #C8BCA8',
                                borderRadius: 0,
                              }}>
                                <span className="meridian-body" style={{ fontSize: 12, color: '#1C1208', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                                  {f.name}
                                </span>
                                <span className="meridian-mono" style={{ fontSize: 10, color: '#7A7060', marginLeft: 14, flexShrink: 0 }}>
                                  {(f.size / 1024 / 1024).toFixed(1)} MB
                                </span>
                              </div>
                            ))}
                          </div>
                          <button
                            className="meridian-btn-primary"
                            disabled={isProcessing}
                            onClick={processBatch}
                            style={{ width: '100%', marginTop: 20 }}
                          >
                            {isProcessing ? 'Processing...' : `Process ${batchFiles.length} Files`}
                          </button>

                          {batchStatus && (
                            <div style={{ marginTop: 24 }}>
                              <div style={{ marginBottom: 12 }}>
                                <span className="meridian-body" style={{ fontSize: 13, fontWeight: 600, color: '#1C1208' }}>
                                  Batch Progress: {batchStatus.completed}/{batchStatus.total}
                                </span>
                              </div>
                              {batchStatus.completed === batchStatus.total && (
                                <button className="meridian-btn-primary" style={{ width: '100%' }} onClick={downloadBatchResults}>
                                  Download All
                                </button>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}

              {/* ═══ CONTROLS TAB ═══ */}
              {activeTab === 'controls' && (
                <div style={{ maxWidth: 680 }}>
                  <h3 className="meridian-display" style={{ fontSize: 24, margin: '0 0 28px', color: '#1C1208' }}>
                    8D Spatial Parameters
                  </h3>

                  {/* File Info */}
                  {audioFile && (
                    <div style={{ marginBottom: 28, padding: '16px', background: '#F9F6F0', border: '1px solid #C8BCA8', borderRadius: 0 }}>
                      <p className="meridian-body" style={{ fontSize: 14, color: '#1C1208', margin: '0 0 8px', fontWeight: 500 }}>
                        Loaded File
                      </p>
                      <p className="meridian-mono" style={{ fontSize: 12, color: '#7A7060', margin: 0 }}>
                        {audioFile.name || 'Unknown file'}
                      </p>
                    </div>
                  )}

                  {/* Presets */}
                  <div style={{ marginBottom: 28 }}>
                    <p className="meridian-section-label" style={{ marginBottom: 12 }}>
                      Presets
                    </p>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {['subtle', 'classic', 'intense', 'cinematic', 'experimental'].map(preset => (
                        <button
                          key={preset}
                          className="meridian-btn-secondary"
                          onClick={() => applyPreset(preset)}
                        >
                          {preset}
                        </button>
                      ))}
                    </div>
                  </div>

                  <SectionDivider label="Rotation" />

                  <SliderRow
                    label="Rotation Speed"
                    value={rotationSpeed}
                    min={0.01}
                    max={0.6}
                    step={0.01}
                    onChange={setRotationSpeed}
                    unit=" Hz"
                    hint="Primary circular panning frequency"
                  />

                  <SliderRow
                    label="Bass Rotation"
                    value={bassRotation}
                    min={0.01}
                    max={0.3}
                    step={0.01}
                    onChange={setBassRotation}
                    unit=" Hz"
                    hint="Low-frequency band rotation (slower movement)"
                  />

                  <SliderRow
                    label="Treble Rotation"
                    value={trebleRotation}
                    min={0.02}
                    max={0.8}
                    step={0.01}
                    onChange={setTrebleRotation}
                    unit=" Hz"
                    hint="High-frequency band rotation (faster shimmer)"
                  />

                  <SectionDivider label="Reverb & Space" />

                  <SliderRow
                    label="Reverb Room Size"
                    value={reverbRoom}
                    min={0.1}
                    max={1.0}
                    step={0.01}
                    onChange={setReverbRoom}
                    hint="Virtual room dimensions (0.1 = booth, 1.0 = cathedral)"
                  />

                  <SliderRow
                    label="Reverb Mix"
                    value={reverbMix}
                    min={0.0}
                    max={0.7}
                    step={0.01}
                    onChange={setReverbMix}
                    hint="Wet/dry balance for spatial depth"
                  />

                  <SliderRow
                    label="Reverb Density"
                    value={reverbDensity}
                    min={0.3}
                    max={1.0}
                    step={0.01}
                    onChange={setReverbDensity}
                    hint="Diffusion network density (early reflections)"
                  />

                  <SectionDivider label="Spatial Geometry" />

                  <SliderRow
                    label="Stereo Width"
                    value={stereoWidth}
                    min={0.5}
                    max={2.0}
                    step={0.05}
                    onChange={setStereoWidth}
                    hint="Frequency-dependent stereo spread"
                  />

                  <SliderRow
                    label="Elevation"
                    value={elevation}
                    min={-0.3}
                    max={0.3}
                    step={0.01}
                    onChange={setElevation}
                    hint="Vertical positioning (-0.3 = below, +0.3 = above)"
                  />

                  <SliderRow
                    label="Distance"
                    value={distance}
                    min={0.5}
                    max={2.0}
                    step={0.05}
                    onChange={setDistance}
                    hint="Perceived source distance (affects reverb pre-delay)"
                  />

                  <SliderRow
                    label="HRTF Intensity"
                    value={hrtfIntensity}
                    min={0.5}
                    max={1.5}
                    step={0.05}
                    onChange={setHrtfIntensity}
                    hint="Head-related transfer function strength"
                  />

                  <SectionDivider label="Enhancement" />

                  <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
                    <button
                      className={`meridian-toggle ${enableVocalCenter ? 'meridian-toggle-on' : ''}`}
                      onClick={() => setEnableVocalCenter(!enableVocalCenter)}
                    >
                      Vocal Center
                    </button>
                    <button
                      className={`meridian-toggle ${vocalSafeBass ? 'meridian-toggle-on' : ''}`}
                      onClick={() => setVocalSafeBass(!vocalSafeBass)}
                    >
                      Vocal-Safe Bass
                    </button>
                    <button
                      className={`meridian-toggle ${instrumentEnhance ? 'meridian-toggle-on' : ''}`}
                      onClick={() => setInstrumentEnhance(!instrumentEnhance)}
                    >
                      Instrument Enhance
                    </button>
                  </div>

                  <SectionDivider label="12-Band EQ" />

                  <EQVisualizer bands={eqBands} />

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 24px' }}>
                    <SliderRow label="30 Hz Sub" value={eqSub30Gain} min={-14} max={14} step={0.5} onChange={setEqSub30Gain} unit=" dB" />
                    <SliderRow label="60 Hz Sub" value={eqSub60Gain} min={-14} max={14} step={0.5} onChange={setEqSub60Gain} unit=" dB" />
                    <SliderRow label="100 Hz Bass" value={eqBass100Gain} min={-14} max={14} step={0.5} onChange={setEqBass100Gain} unit=" dB" />
                    <SliderRow label="200 Hz Bass" value={eqUbass200Gain} min={-14} max={14} step={0.5} onChange={setEqUbass200Gain} unit=" dB" />
                    <SliderRow label="350 Hz Low-Mid" value={eqLowmid350Gain} min={-14} max={14} step={0.5} onChange={setEqLowmid350Gain} unit=" dB" />
                    <SliderRow label="700 Hz Mid" value={eqMid700Gain} min={-14} max={14} step={0.5} onChange={setEqMid700Gain} unit=" dB" />
                    <SliderRow label="1.5k Mid" value={eqUmid1500Gain} min={-14} max={14} step={0.5} onChange={setEqUmid1500Gain} unit=" dB" />
                    <SliderRow label="3k Presence" value={eqPresence3kGain} min={-14} max={14} step={0.5} onChange={setEqPresence3kGain} unit=" dB" />
                    <SliderRow label="5k Definition" value={eqDef5kGain} min={-14} max={14} step={0.5} onChange={setEqDef5kGain} unit=" dB" />
                    <SliderRow label="8k Brilliance" value={eqBril8kGain} min={-14} max={14} step={0.5} onChange={setEqBril8kGain} unit=" dB" />
                    <SliderRow label="12k Air" value={eqAir12kGain} min={-14} max={14} step={0.5} onChange={setEqAir12kGain} unit=" dB" />
                    <SliderRow label="16k Ultra-Air" value={eqUair16kGain} min={-14} max={14} step={0.5} onChange={setEqUair16kGain} unit=" dB" />
                  </div>

                  <div style={{ marginTop: 32 }}>
                    <button
                      className="meridian-btn-primary"
                      style={{ width: '100%', padding: '16px' }}
                      onClick={process8DAudio}
                      disabled={!audioFile && !originalAudio || isProcessing || backendStatus !== 'connected'}
                    >
                      {isProcessing ? 'Processing...' : 'Process Audio'}
                    </button>
                  </div>
                </div>
              )}

              {/* ═══ SPATIAL TAB ═══ */}
              {activeTab === 'spatial' && (
                <div style={{ maxWidth: 600 }}>
                  <h3 className="meridian-display" style={{ fontSize: 24, margin: '0 0 28px', color: '#1C1208' }}>
                    Spatial Output Format
                  </h3>

                  <p className="meridian-body" style={{ fontSize: 13, color: '#7A7060', marginBottom: 20, lineHeight: 1.6 }}>
                    Choose the output spatial format. Advanced formats require compatible playback systems.
                  </p>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {[
                      {
                        id: 'stereo',
                        name: 'Stereo (Standard)',
                        desc: 'Standard left/right for headphones & speakers',
                        availability: 'Universal — works everywhere',
                        note: null,
                        badge: 'RECOMMENDED',
                      },
                      {
                        id: 'ambisonics_foa',
                        name: 'Ambisonics FOA (B-Format)',
                        desc: '4-channel first-order ambisonics (W, X, Y, Z)',
                        availability: backendCaps.ambisonics_foa ? 'Available' : 'Requires Ambisonics encoder',
                        note: 'Outputs a 4-channel 24-bit WAV (B-format). Use a binaural decoder like SSA or IEM Binaural for headphone playback. Compatible with VR/360° environments and spatial audio DAW workflows.',
                        badge: '4CH · 24-BIT WAV',
                      },
                      {
                        id: 'atmos_71_4',
                        name: 'Dolby Atmos 7.1.4',
                        desc: '12-channel immersive bed layout',
                        availability: backendCaps.atmos_71_4 ? 'Available' : 'Requires Atmos encoder',
                        note: 'Outputs a 12-channel 24-bit WAV (L/R/C/LFE/Ls/Rs/Lss/Rss/Ltf/Rtf/Ltb/Rtb). Requires a Dolby Atmos rendering environment or compatible speakers/processor for proper playback.',
                        badge: '12CH · 24-BIT WAV',
                      },
                    ].map(format => {
                      const isSelected = spatialFormat === format.id;
                      const isDisabled = (format.id === 'ambisonics_foa' && !backendCaps.ambisonics_foa) ||
                                        (format.id === 'atmos_71_4' && !backendCaps.atmos_71_4);
                      return (
                        <div
                          key={format.id}
                          onClick={() => !isDisabled && setSpatialFormat(format.id)}
                          style={{
                            padding: '18px 20px',
                            border: isSelected ? '2px solid #C13318' : '1px solid #C8BCA8',
                            borderRadius: 0,
                            background: isSelected ? 'rgba(193, 51, 24, 0.04)' : '#F9F6F0',
                            cursor: isDisabled ? 'not-allowed' : 'pointer',
                            opacity: isDisabled ? 0.5 : 1,
                            transition: 'all 0.2s',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 6 }}>
                            <div style={{
                              width: 12, height: 12, borderRadius: '50%', flexShrink: 0, marginTop: 3,
                              border: '2px solid ' + (isSelected ? '#C13318' : '#C8BCA8'),
                              background: isSelected ? '#C13318' : 'transparent',
                            }} />
                            <div style={{ flex: 1 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                                <span className="meridian-body" style={{ fontSize: 14, fontWeight: 600, color: '#1C1208' }}>
                                  {format.name}
                                </span>
                                {format.badge && (
                                  <span className="meridian-mono" style={{
                                    fontSize: 9, letterSpacing: '0.1em', padding: '2px 7px',
                                    background: isSelected ? 'rgba(193,51,24,0.1)' : '#EDE8DF',
                                    color: isSelected ? '#C13318' : '#7A7060',
                                    border: '1px solid ' + (isSelected ? 'rgba(193,51,24,0.25)' : '#C8BCA8'),
                                  }}>
                                    {format.badge}
                                  </span>
                                )}
                              </div>
                              <p className="meridian-body" style={{ fontSize: 12, color: '#7A7060', margin: '4px 0 0' }}>
                                {format.desc}
                              </p>
                              <p className="meridian-mono" style={{ fontSize: 10, color: isDisabled ? '#C13318' : '#2A7A3A', margin: '4px 0 0', letterSpacing: '0.04em' }}>
                                {format.availability}
                              </p>
                            </div>
                          </div>
                          {/* Expanded notes panel for advanced formats */}
                          {isSelected && format.note && (
                            <div style={{
                              marginTop: 12, marginLeft: 22,
                              padding: '12px 14px',
                              background: 'rgba(27, 56, 88, 0.04)',
                              border: '1px solid rgba(27, 56, 88, 0.15)',
                              borderRadius: 0,
                            }}>
                              <p className="meridian-body" style={{ fontSize: 11, color: '#1B3858', margin: 0, lineHeight: 1.7 }}>
                                {format.note}
                              </p>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  <SectionDivider label="Output Settings" />

                  <div style={{ marginBottom: 18 }}>
                    <p className="meridian-section-label" style={{ marginBottom: 10 }}>
                      Format
                    </p>
                    <div style={{ display: 'flex', gap: 8 }}>
                      {['mp3', 'wav', 'flac'].map(fmt => (
                        <button
                          key={fmt}
                          className={`meridian-btn-secondary ${outputFormat === fmt && spatialFormat === 'stereo' ? 'meridian-btn-selected' : ''}`}
                          onClick={() => setOutputFormat(fmt)}
                          disabled={spatialFormat !== 'stereo'}
                        >
                          {fmt.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>

                  {outputFormat === 'mp3' && spatialFormat === 'stereo' && (
                    <SliderRow
                      label="Bitrate"
                      value={bitrate}
                      min={128}
                      max={320}
                      step={32}
                      onChange={setBitrate}
                      unit=" kbps"
                    />
                  )}
                </div>
              )}

              {/* ═══ STEMS TAB ═══ */}
              {activeTab === 'stems' && (
                <div style={{ maxWidth: 680 }}>
                  <h3 className="meridian-display" style={{ fontSize: 24, margin: '0 0 28px', color: '#1C1208' }}>
                    Stem Separation
                  </h3>

                  {!backendCaps.stem_separation ? (
                    <div style={{
                      padding: '20px 24px',
                      background: 'rgba(193, 51, 24, 0.08)',
                      border: '1px solid rgba(193, 51, 24, 0.3)',
                      borderRadius: 0,
                    }}>
                      <p className="meridian-body" style={{ fontSize: 13, color: '#C13318', fontWeight: 600, margin: '0 0 8px' }}>
                        Stem separation unavailable
                      </p>
                      <p className="meridian-mono" style={{ fontSize: 10, color: '#7A7060', margin: 0 }}>
                        Install Demucs to enable: <code style={{ color: '#1B3858' }}>pip install demucs</code>
                      </p>
                    </div>
                  ) : (
                    <>
                      <div style={{ marginBottom: 24 }}>
                        <button
                          className={`meridian-toggle ${enableStemSeparation ? 'meridian-toggle-on' : ''}`}
                          onClick={() => setEnableStemSeparation(!enableStemSeparation)}
                          style={{ fontSize: 12, padding: '10px 18px' }}
                        >
                          {enableStemSeparation ? 'Enabled' : 'Disabled'}
                        </button>
                      </div>

                      {enableStemSeparation && (
                        <>
                          {/* GPU timing warning */}
                          <div style={{
                            marginBottom: 20,
                            padding: '12px 16px',
                            background: 'rgba(212, 144, 90, 0.08)',
                            border: '1px solid rgba(212, 144, 90, 0.35)',
                            borderRadius: 0,
                            display: 'flex', alignItems: 'flex-start', gap: 10,
                          }}>
                            <span className="meridian-led meridian-led-amber" style={{ marginTop: 3 }} />
                            <p className="meridian-body" style={{ fontSize: 12, color: '#7A7060', margin: 0, lineHeight: 1.6 }}>
                              Stem separation adds <strong style={{ color: '#1C1208' }}>1–3 minutes</strong> per track. GPU recommended — CPU may take 5–10 min per track.
                            </p>
                          </div>

                          <p className="meridian-body" style={{ fontSize: 13, color: '#7A7060', marginBottom: 20, lineHeight: 1.6 }}>
                            Separate audio into stems and apply individual spatial processing to each.
                          </p>

                          {/* Stem Model */}
                          <div style={{ marginBottom: 24 }}>
                            <p className="meridian-section-label" style={{ marginBottom: 10 }}>
                              Demucs Model
                            </p>
                            <div style={{ display: 'flex', gap: 8 }}>
                              {[
                                { id: 'htdemucs', label: 'htdemucs', desc: '4 stems' },
                                { id: 'htdemucs_6s', label: 'htdemucs_6s', desc: '6 stems' },
                              ].map(m => (
                                <button
                                  key={m.id}
                                  className={`meridian-btn-secondary ${stemModel === m.id ? 'meridian-btn-selected' : ''}`}
                                  onClick={() => setStemModel(m.id)}
                                  style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2, padding: '10px 14px' }}
                                >
                                  <span style={{ fontWeight: 700 }}>{m.label}</span>
                                  <span style={{ fontSize: 10, opacity: 0.7, textTransform: 'none', letterSpacing: 0 }}>{m.desc}</span>
                                </button>
                              ))}
                            </div>
                          </div>

                          <div style={{ marginBottom: 24 }}>
                            <p className="meridian-section-label" style={{ marginBottom: 12 }}>
                              Stem Preset
                            </p>
                            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                              {[
                                { id: 'auto', label: 'Auto' },
                                { id: 'vocals_front', label: 'Vocals Front' },
                                { id: 'drums_wide', label: 'Drums Wide' },
                                { id: 'custom', label: 'Custom' },
                              ].map(preset => (
                                <button
                                  key={preset.id}
                                  className={`meridian-btn-secondary ${stemPreset === preset.id ? 'meridian-btn-selected' : ''}`}
                                  onClick={() => setStemPreset(preset.id)}
                                >
                                  {preset.label}
                                </button>
                              ))}
                            </div>
                          </div>

                          {/* Preset summary grid for non-custom presets */}
                          {stemPreset !== 'custom' && (
                            <div style={{ marginBottom: 24 }}>
                              <p className="meridian-section-label" style={{ marginBottom: 10 }}>
                                Effective Rotation Per Stem
                              </p>
                              <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(4, 1fr)',
                                gap: 1,
                                background: '#C8BCA8',
                                border: '1px solid #C8BCA8',
                              }}>
                                {[
                                  { label: '🎤 Vocals', rot: stemPreset === 'vocals_front' ? rotationSpeed * 0.5 : stemPreset === 'drums_wide' ? rotationSpeed * 0.6 : rotationSpeed },
                                  { label: '🥁 Drums', rot: stemPreset === 'vocals_front' ? rotationSpeed * 1.3 : stemPreset === 'drums_wide' ? trebleRotation * 1.5 : trebleRotation },
                                  { label: '🎸 Bass', rot: stemPreset === 'vocals_front' ? bassRotation : stemPreset === 'drums_wide' ? bassRotation * 0.5 : bassRotation },
                                  { label: '🎹 Other', rot: stemPreset === 'vocals_front' ? rotationSpeed * 0.9 : stemPreset === 'drums_wide' ? rotationSpeed : rotationSpeed },
                                ].map(stem => (
                                  <div key={stem.label} style={{ background: '#FEFDFB', padding: '12px 10px', textAlign: 'center' }}>
                                    <p className="meridian-body" style={{ fontSize: 11, color: '#7A7060', margin: '0 0 6px' }}>{stem.label}</p>
                                    <p className="meridian-mono" style={{ fontSize: 13, color: '#C13318', margin: 0, fontWeight: 600 }}>
                                      {stem.rot.toFixed(2)} Hz
                                    </p>
                                    <p className="meridian-mono" style={{ fontSize: 9, color: '#7A7060', margin: '3px 0 0' }}>
                                      {(stem.rot * 60).toFixed(1)} RPM
                                    </p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {stemPreset === 'custom' && (
                            <>
                              <SectionDivider label="Per-Stem Controls" />
                              {/* 5-column header */}
                              <div style={{
                                display: 'grid',
                                gridTemplateColumns: '80px 1fr 1fr 1fr 1fr',
                                gap: 8,
                                marginBottom: 8,
                                paddingBottom: 8,
                                borderBottom: '1px solid #C8BCA8',
                              }}>
                                {['Stem', 'Rotation', 'Width', 'Elevation', 'Reverb Mix'].map(h => (
                                  <span key={h} className="meridian-section-label" style={{ fontSize: 9 }}>{h}</span>
                                ))}
                              </div>
                              {/* Stem rows */}
                              {[
                                {
                                  emoji: '🎤', label: 'Vocals',
                                  rot: stemVocalsRotation ?? rotationSpeed, setRot: setStemVocalsRotation, rotMax: 0.6,
                                  wid: stemVocalsWidth ?? stereoWidth, setWid: setStemVocalsWidth,
                                  elev: stemVocalsElevation ?? elevation, setElev: setStemVocalsElevation,
                                  rev: stemVocalsReverb ?? reverbMix, setRev: setStemVocalsReverb,
                                },
                                {
                                  emoji: '🥁', label: 'Drums',
                                  rot: stemDrumsRotation ?? trebleRotation, setRot: setStemDrumsRotation, rotMax: 0.8,
                                  wid: stemDrumsWidth ?? stereoWidth, setWid: setStemDrumsWidth,
                                  elev: stemDrumsElevation ?? elevation, setElev: setStemDrumsElevation,
                                  rev: stemDrumsReverb ?? reverbMix, setRev: setStemDrumsReverb,
                                },
                                {
                                  emoji: '🎸', label: 'Bass',
                                  rot: stemBassRotation ?? bassRotation, setRot: setStemBassRotation, rotMax: 0.3,
                                  wid: stemBassWidth ?? stereoWidth, setWid: setStemBassWidth,
                                  elev: stemBassElevation ?? elevation, setElev: setStemBassElevation,
                                  rev: stemBassReverb ?? reverbMix, setRev: setStemBassReverb,
                                },
                                {
                                  emoji: '🎹', label: 'Other',
                                  rot: stemOtherRotation ?? rotationSpeed, setRot: setStemOtherRotation, rotMax: 0.6,
                                  wid: stemOtherWidth ?? stereoWidth, setWid: setStemOtherWidth,
                                  elev: stemOtherElevation ?? elevation, setElev: setStemOtherElevation,
                                  rev: stemOtherReverb ?? reverbMix, setRev: setStemOtherReverb,
                                },
                                ...(stemModel === 'htdemucs_6s' ? [
                                  {
                                    emoji: '🎺', label: 'Guitar',
                                    rot: stemGuitarRotation ?? rotationSpeed, setRot: setStemGuitarRotation, rotMax: 0.6,
                                    wid: stemGuitarWidth ?? stereoWidth, setWid: setStemGuitarWidth,
                                    elev: null, setElev: null,
                                    rev: null, setRev: null,
                                  },
                                  {
                                    emoji: '🎵', label: 'Piano',
                                    rot: stemPianoRotation ?? rotationSpeed, setRot: setStemPianoRotation, rotMax: 0.6,
                                    wid: stemPianoWidth ?? stereoWidth, setWid: setStemPianoWidth,
                                    elev: null, setElev: null,
                                    rev: null, setRev: null,
                                  },
                                ] : []),
                              ].map(stem => (
                                <div key={stem.label} style={{
                                  display: 'grid',
                                  gridTemplateColumns: '80px 1fr 1fr 1fr 1fr',
                                  gap: 8,
                                  alignItems: 'center',
                                  padding: '10px 0',
                                  borderBottom: '1px solid #E8E0D2',
                                }}>
                                  <span className="meridian-body" style={{ fontSize: 12, color: '#1C1208', fontWeight: 600 }}>
                                    {stem.emoji} {stem.label}
                                  </span>
                                  {/* Rotation */}
                                  <div>
                                    <input type="range" min={0.01} max={stem.rotMax} step={0.01} value={stem.rot}
                                      onChange={e => stem.setRot(parseFloat(e.target.value))} />
                                    <span className="meridian-value meridian-mono" style={{ fontSize: 9 }}>{stem.rot.toFixed(2)} Hz</span>
                                  </div>
                                  {/* Width */}
                                  <div>
                                    <input type="range" min={0.5} max={2.0} step={0.05} value={stem.wid}
                                      onChange={e => stem.setWid(parseFloat(e.target.value))} />
                                    <span className="meridian-value meridian-mono" style={{ fontSize: 9 }}>{stem.wid.toFixed(2)}</span>
                                  </div>
                                  {/* Elevation */}
                                  <div>
                                    {stem.elev !== null && stem.setElev ? (
                                      <>
                                        <input type="range" min={-0.3} max={0.3} step={0.01} value={stem.elev}
                                          onChange={e => stem.setElev(parseFloat(e.target.value))} />
                                        <span className="meridian-value meridian-mono" style={{ fontSize: 9 }}>{stem.elev >= 0 ? '+' : ''}{stem.elev.toFixed(2)}</span>
                                      </>
                                    ) : <span className="meridian-mono" style={{ fontSize: 9, color: '#C8BCA8' }}>—</span>}
                                  </div>
                                  {/* Reverb Mix */}
                                  <div>
                                    {stem.rev !== null && stem.setRev ? (
                                      <>
                                        <input type="range" min={0.0} max={0.7} step={0.01} value={stem.rev}
                                          onChange={e => stem.setRev(parseFloat(e.target.value))} />
                                        <span className="meridian-value meridian-mono" style={{ fontSize: 9 }}>{stem.rev.toFixed(2)}</span>
                                      </>
                                    ) : <span className="meridian-mono" style={{ fontSize: 9, color: '#C8BCA8' }}>—</span>}
                                  </div>
                                </div>
                              ))}
                            </>
                          )}
                        </>
                      )}
                    </>
                  )}
                </div>
              )}

              {/* ═══ VISUALIZER TAB ═══ */}
              {activeTab === 'visualizer' && (
                <div style={{ maxWidth: 600 }}>
                  <h3 className="meridian-display" style={{ fontSize: 24, margin: '0 0 28px', color: '#1C1208' }}>
                    Video Visualizer
                  </h3>

                  <div style={{ marginBottom: 24 }}>
                    <button
                      className={`meridian-toggle ${generateVideo ? 'meridian-toggle-on' : ''}`}
                      onClick={() => setGenerateVideo(!generateVideo)}
                      style={{ fontSize: 12, padding: '10px 18px' }}
                    >
                      {generateVideo ? 'Enabled' : 'Disabled'}
                    </button>
                  </div>

                  {generateVideo && (
                    <>
                      <p className="meridian-body" style={{ fontSize: 13, color: '#7A7060', marginBottom: 20, lineHeight: 1.6 }}>
                        Generate a synchronized video visualizer for your 8D audio.
                      </p>

                      <div style={{ marginBottom: 24 }}>
                        <p className="meridian-section-label" style={{ marginBottom: 12 }}>
                          Style
                        </p>
                        <div style={{ display: 'flex', gap: 8 }}>
                          {['waveform', 'spectrum', 'vectorscope'].map(style => (
                            <button
                              key={style}
                              className={`meridian-btn-secondary ${videoStyle === style ? 'meridian-btn-selected' : ''}`}
                              onClick={() => setVideoStyle(style)}
                            >
                              {style}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div style={{ marginBottom: 24 }}>
                        <p className="meridian-section-label" style={{ marginBottom: 12 }}>
                          Resolution
                        </p>
                        <div style={{ display: 'flex', gap: 8 }}>
                          {['854x480', '1280x720', '1920x1080', '3840x2160'].map(res => (
                            <button
                              key={res}
                              className={`meridian-btn-secondary ${videoResolution === res ? 'meridian-btn-selected' : ''}`}
                              onClick={() => setVideoResolution(res)}
                            >
                              {res}
                            </button>
                          ))}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* ═══ OUTPUT TAB ═══ */}
              {activeTab === 'output' && (
                <div style={{ maxWidth: 600 }}>
                  <h3 className="meridian-display" style={{ fontSize: 24, margin: '0 0 28px', color: '#1C1208' }}>
                    Output & Playback
                  </h3>

                  {isProcessing && (
                    <div style={{ marginBottom: 32 }}>
                      <p className="meridian-body" style={{ fontSize: 14, fontWeight: 600, color: '#1C1208', marginBottom: 12 }}>
                        Processing Pipeline
                      </p>
                      {pipelineStages.map((stage, idx) => {
                        const isActive = progress >= stage.threshold && progress < (pipelineStages[idx + 1]?.threshold || 100);
                        const isDone = progress > stage.threshold;

                        return (
                          <div key={idx} className={`meridian-pipeline-step ${isActive ? 'active' : isDone ? 'done' : ''}`} style={{ marginBottom: 8 }}>
                            <span className={`meridian-step-num ${isActive ? 'active' : isDone ? 'done' : ''}`}>
                              {String(idx + 1).padStart(2, '0')}
                            </span>
                            <div style={{ flex: 1 }}>
                              <p className="meridian-body" style={{ fontSize: 13, fontWeight: 600, color: '#1C1208', margin: '0 0 4px' }}>
                                {stage.label}
                              </p>
                              <p className="meridian-mono" style={{ fontSize: 10, color: '#7A7060', margin: 0 }}>
                                {stage.desc}
                              </p>
                            </div>
                          </div>
                        );
                      })}

                      <div className="meridian-progress-track" style={{ marginTop: 16 }}>
                        <div className="meridian-progress-fill" style={{ width: `${progress}%` }} />
                      </div>
                      <p className="meridian-mono" style={{ fontSize: 11, color: '#7A7060', margin: '8px 0 0', textAlign: 'center' }}>
                        {processingStage || 'Processing...'}
                      </p>
                    </div>
                  )}

                  {processedAudio && !isProcessing && (
                    <>
                      <div style={{ marginBottom: 28 }}>
                        <p className="meridian-section-label" style={{ marginBottom: 12 }}>
                          Playback Mode
                        </p>
                        <div style={{ display: 'flex', gap: 8 }}>
                          <button
                            className={`meridian-btn-secondary ${currentMode === 'original' ? 'meridian-btn-selected' : ''}`}
                            onClick={() => switchMode('original')}
                          >
                            Original
                          </button>
                          <button
                            className={`meridian-btn-secondary ${currentMode === 'processed' ? 'meridian-btn-selected' : ''}`}
                            onClick={() => switchMode('processed')}
                          >
                            8D Processed
                          </button>
                        </div>
                      </div>

                      <div style={{
                        padding: '24px',
                        background: '#F9F6F0',
                        border: '1px solid #C8BCA8',
                        borderRadius: 0,
                        marginBottom: 24,
                      }}>
                        <audio
                          ref={audioElementRef}
                          src={currentMode === 'original' ? originalAudio : processedAudio}
                          onEnded={() => setIsPlaying(false)}
                          style={{ display: 'none' }}
                        />
                        <button
                          className="meridian-btn-primary"
                          onClick={togglePlayback}
                          style={{ width: '100%', marginBottom: 16 }}
                        >
                          {isPlaying ? 'Pause' : 'Play'}
                        </button>
                        <p className="meridian-mono" style={{ fontSize: 11, color: '#7A7060', margin: 0, textAlign: 'center' }}>
                          Now playing: {currentMode === 'original' ? 'Original' : '8D Processed'}
                        </p>
                      </div>

                      <button
                        className="meridian-btn-primary"
                        onClick={downloadAudio}
                        style={{ width: '100%' }}
                      >
                        Download 8D Audio
                      </button>

                      {processedVideoUrl && (
                        <div style={{ marginTop: 24 }}>
                          <p className="meridian-section-label" style={{ marginBottom: 12 }}>
                            Video Visualizer
                          </p>
                          <video
                            src={processedVideoUrl}
                            controls
                            style={{
                              width: '100%',
                              border: '1px solid #C8BCA8',
                              borderRadius: 0,
                            }}
                          />
                        </div>
                      )}
                    </>
                  )}

                  {!isProcessing && !processedAudio && (
                    <div style={{
                      padding: '32px 24px',
                      background: '#F9F6F0',
                      border: '1px solid #C8BCA8',
                      borderRadius: 0,
                      textAlign: 'center',
                    }}>
                      <p className="meridian-body" style={{ fontSize: 13, color: '#7A7060', margin: 0 }}>
                        No processed audio yet. Upload a file and configure parameters to begin.
                      </p>
                    </div>
                  )}

                  {/* Audio Analysis */}
                  {audioAnalysis && (
                    <div style={{ marginTop: 32 }}>
                      <SectionDivider label="Audio Analysis" />
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        {audioAnalysis.genre && (
                          <span className="meridian-analysis-tag">
                            <span>Genre:</span>
                            <span>{audioAnalysis.genre}</span>
                          </span>
                        )}
                        {audioAnalysis.bpm && (
                          <span className="meridian-analysis-tag">
                            <span>BPM:</span>
                            <span>{audioAnalysis.bpm}</span>
                          </span>
                        )}
                        {audioAnalysis.key && (
                          <span className="meridian-analysis-tag">
                            <span>Key:</span>
                            <span>{audioAnalysis.key} {audioAnalysis.mode}</span>
                          </span>
                        )}
                        {audioAnalysis.crest_factor_db != null && (
                          <span className="meridian-analysis-tag">
                            <span>Crest:</span>
                            <span>{audioAnalysis.crest_factor_db.toFixed(1)} dB</span>
                          </span>
                        )}
                        {audioAnalysis.has_vocals != null && (
                          <span className="meridian-analysis-tag">
                            <span>Vocals:</span>
                            <span>{audioAnalysis.has_vocals ? 'Yes' : 'No'}</span>
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default AudioConverter;
