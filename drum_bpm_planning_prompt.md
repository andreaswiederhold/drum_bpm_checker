
# Drum BPM Timing Trainer - Windows Real-Time Application

## Implementation Status: ✅ COMPLETE

---

## Architecture Overview

### Chosen Onset Detection Method: Energy Envelope

I selected **Energy Envelope** detection over Spectral Flux for these reasons:
- **Lower computational cost**: Simple rectification + moving average vs. FFT per block
- **Lower latency**: No need to accumulate frames for spectral analysis
- **Good transient response**: Works well for drum hits which have sharp attack
- **Easier tuning**: Single sensitivity parameter vs. multiple spectral parameters

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────────────┐   │
│  │ Microphone   │───▶│ Audio Callback  │───▶│ Thread-Safe Ring      │   │
│  │ (PortAudio)  │    │ (512 samples)   │    │ Buffer (deque)        │   │
│  └──────────────┘    └─────────────────┘    └───────────┬───────────┘   │
│                                                          │               │
│                                                          ▼               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   PROCESSING THREAD                               │   │
│  │  ┌─────────────┐   ┌────────────┐   ┌─────────────┐   ┌────────┐ │   │
│  │  │ High-Pass   │──▶│ Energy     │──▶│ Adaptive    │──▶│ BPM    │ │   │
│  │  │ Filter      │   │ Envelope   │   │ Threshold   │   │ Compute│ │   │
│  │  └─────────────┘   └────────────┘   └─────────────┘   └────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                          │               │
│                                                          ▼               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      GUI THREAD (QTimer @ 30Hz)                   │   │
│  │  ┌─────────────┐   ┌────────────┐   ┌─────────────────────────┐  │   │
│  │  │ BPM Display │   │ Waveform   │   │ Spectrum Plot           │  │   │
│  │  │ (72pt font) │   │ Plot       │   │ (FFT + Hann window)     │  │   │
│  │  └─────────────┘   └────────────┘   └─────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Latency Analysis

| Component | Latency |
|-----------|---------|
| Audio block size (512 samples @ 44.1kHz) | ~11.6 ms |
| Processing thread cycle | ~5-10 ms |
| GUI update interval | 33 ms |
| **Total end-to-end** | **~50-55 ms** |

### Buffer Sizes

- **Audio ring buffer**: 3 seconds (132,300 samples @ 44.1kHz)
- **Energy history**: 500 ms for adaptive threshold
- **Onset history**: Last 11 onsets for BPM calculation

---

## Original Planning Prompt

# Prompt for GitHub Copilot (Planning Mode) — Windows Real‑Time Drum BPM GUI

**Role:**  
Act as a senior Python engineer specialized in **real‑time audio processing** and **Qt GUI** applications on **Windows**.

**Goal:**  
Build a small, self‑contained Python app with a GUI that listens to **microphone input** (Windows audio input), detects **drum stick hits** (onsets), computes and displays **current BPM**, and shows both a **live waveform** and a **frequency spectrum**. The tool is for improving drum timing.

**Tech Stack (Windows‑friendly):**  
- Audio I/O: `sounddevice` (PortAudio)  
- Numerics: `numpy` (+ optional `scipy` if needed)  
- GUI: `PySide6`  
- Real‑time plotting: `pyqtgraph` (fast)  
- Optional (only if clearly beneficial): `librosa` for onset features

**Functional Requirements:**  
1. **Live audio capture** from the system microphone (mono, 44.1 kHz or 48 kHz).  
2. **Onset detection** robust to noise:  
   - Pre‑processing: mono, **high‑pass filter ~30–60 Hz** to reduce rumble; optional band‑pass for snare transient focus (e.g., 700–4 kHz).  
   - Detection method: **energy envelope** OR **spectral flux** (choose one and explain).  
   - **Adaptive threshold** (mean + k·std or EMA‑based) and **refractory/lockout** of ~80–120 ms to avoid double hits.  
3. **BPM computation** from Inter‑Onset Intervals (IOI):  
   - Maintain last N onsets (e.g., 5–10), compute **robust BPM** (median‑based), reject outliers (IQR or MAD).  
   - Show **big numeric BPM** + a small **sparkline** or short history trend.  
4. **Visualizations:**  
   - **Waveform**: last 1–3 seconds, smooth scrolling.  
   - **Spectrum**: FFT with Hann window; magnitude in dB; sensible frequency axis.  
5. **Controls in the GUI:**  
   - Sensitivity/threshold (slider).  
   - Min interval/refractory (ms) (slider).  
   - High‑pass on/off.  
   - Sample rate selector (44.1/48 kHz).  
   - **Audio device selector** (list Windows input devices via `sounddevice.query_devices()`; allow selection).  
6. **Performance:**  
   - End‑to‑end latency target **< 50–100 ms**.  
   - Use an **audio callback → ring buffer/queue → processing thread → GUI updates via QTimer (30–60 Hz)**.  
   - No heavy work in the audio callback.  
7. **Stability:**  
   - Start/stop capture cleanly; handle device errors/dropouts without freezing.  
   - Thread‑safe communication, no race conditions.  
8. **Export (optional):**  
   - Save onset timestamps and rolling BPM to **CSV**.

**Non‑Functional / Architecture:**  
- Producer‑Consumer design: audio callback fills a **thread‑safe ring buffer** (e.g., `collections.deque`), processing thread computes envelope/onsets/BPM, GUI thread renders with a QTimer.  
- Separate modules/sections for: audio I/O, signal processing, GUI.  
- Clear docstrings and inline comments for core parts (callback, onset detection, BPM aggregation).  
- Cross‑platform code where feasible, but **optimize for Windows** (WASAPI host if applicable).

**Windows specifics to consider:**  
- Default to the active microphone; provide a **device dropdown** (show name and index).  
- Sample formats: use `dtype='float32'`.  
- Reasonable default `blocksize` (e.g., 512 or 1024); allow override.  
- Mention Windows privacy setting: microphone access must be enabled.

**Deliverables (in this order):**  
1. **Planning section:**  
   - Short architecture overview and reasoning for the chosen onset method.  
   - Exact data flow (callback → buffer → processing → GUI).  
   - Latency considerations and chosen buffer sizes.  
2. **Setup instructions for Windows:**  
   - `pip install` commands.  
   - How to select a device, how to run.  
3. **A single, runnable Python file** (`drum_bpm_gui.py`) that implements everything above, with comments.  
4. **Brief README block** at the top of the file: features, controls, tips for calibration.  
5. **Testing notes:** how to verify detection (e.g., tap near mic), how to check BPM stability, how to tweak thresholds.

**Acceptance Criteria:**  
- Clean launch on Windows with PySide6 GUI.  
- Real‑time waveform and spectrum rendering at ~30–60 FPS without GUI stutter.  
- Reliable hit detection on typical pad/snare taps; double hits suppressed via refractory interval.  
- BPM readout stabilizes with steady playing and doesn’t jump erratically.  
- Device selection works and the app doesn’t crash on device errors.  
- CPU usage stays reasonable on a typical laptop.

**Nice‑to‑haves (only if time permits):**  
- **Metronome mode**: play a click at a selected BPM and show **±ms deviation** per hit (off‑grid error).  
- **Jitter metrics**: IOI histogram, median ± MAD, std‑dev in ms.  
- **Dark mode** toggle.  
- **Pack to EXE** instructions with PyInstaller (optional).

**Key Implementation Hints (follow):**  
- Use `sounddevice.InputStream` with a callback to push mono frames into a `deque` with a fixed max length (couple seconds).  
- For onset detection, compute either:  
  - **Energy envelope**: rectify → moving average / low‑pass; adaptive z‑score threshold; or  
  - **Spectral flux**: windowed FFT per block; sum positive spectral differences; adaptive threshold.  
- For spectrum, use `numpy.fft.rfft` on a recent window of samples; apply Hann window; convert to dB scale.  
- Use `QTimer` (e.g., 30 ms) to update GUI; never do plotting in the audio callback.  
- Make min‑interval and threshold user‑tunable; expose on the GUI.  
- Use robust statistics (median) to compute BPM from the last N IOIs and ignore outliers.

**Now produce:**  
1) The plan (bulleted, concise).  
2) The full Windows setup instructions.  
3) The complete single‑file implementation with clear comments.  
4) A short “How to calibrate” guide within the README block.  
