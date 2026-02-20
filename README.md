# 8D Audio Converter

A web application to convert audio files into immersive 8D audio experiences.

## Description

The 8D Audio Converter is a modern web-based tool that transforms regular audio files into 8D audio format, creating an immersive listening experience where the sound appears to move around your head in a three-dimensional space. This effect is achieved through advanced audio panning and spatial manipulation techniques using FFmpeg's powerful audio processing capabilities.

### What is 8D Audio?

8D audio is a sound technique that creates the illusion of music moving in a circular pattern around the listener's head, producing an immersive, three-dimensional soundscape. When listened to with headphones, it creates the sensation that the sound is coming from different directions and distances, offering a unique and engaging audio experience.

### How It Works

The application processes audio files by applying dynamic panning effects that move the sound between left and right channels while adjusting phase relationships and spatial cues. The result is an audio file that, when played through headphones, creates the perception of sound sources moving around the listener in three-dimensional space.

## Features

- **Audio Conversion**: Convert standard audio files to 8D format
- **Batch Processing**: Process multiple songs efficiently
- **Web Interface**: User-friendly UI built with React and TypeScript
- **FFmpeg Integration**: Leverages FFmpeg for audio processing

## Project Structure

```
├── src/                          # Frontend source code
│   ├── 8d_audio_converter.tsx   # Main React component
│   ├── index.css                 # Styles
│   └── main.tsx                  # Entry point
├── backend.py                    # Main backend server
├── backend_v2.py                 # Backend v2 implementation
├── backend_8d_server_v2_incomplete.py  # WIP backend version
├── requirements.txt              # Python dependencies
├── package.json                  # Node dependencies
├── vite.config.ts               # Vite configuration
├── tsconfig.json                # TypeScript configuration
├── tailwind.config.js           # Tailwind CSS configuration
├── ffmpeg-master-latest-win64-gpl/  # FFmpeg binaries
└── uploads/                      # User uploads directory
```

## Setup

### Backend Requirements

Install Python requirements:

```bash
pip install -r requirements.txt
```

### Frontend Requirements

Install Node dependencies:

```bash
npm install
```

## Usage

### Start Backend

```bash
python backend.py
```

### Start Frontend (Development)

```bash
npm run dev
```

The application will be available at `http://localhost:5173` (or the port shown by Vite).

## Technologies

- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **Backend**: Python (Flask/FastAPI)
- **Audio Processing**: FFmpeg
- **Build Tools**: Vite, PostCSS, TypeScript

## Author

@geek-alt
