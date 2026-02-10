# Drum BPM Timing Trainer

A real-time drum hit detection and BPM display application for Windows, designed to help drummers improve their timing accuracy.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- 🎤 **Real-time microphone capture** with selectable audio device
- 🥁 **Onset detection** using energy envelope with adaptive threshold
- 📊 **BPM computation** from inter-onset intervals with outlier rejection (median-based)
- 📈 **Live waveform display** (last 2 seconds of audio)
- 🎵 **Live frequency spectrum** (FFT with Hann window)
- ⚙️ **Adjustable parameters**: sensitivity, refractory period, high-pass filter
- 💾 **CSV export** of onset timestamps and BPM readings
- 🌙 **Dark theme** UI

## Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  Controls: [Device ▼] [44100 Hz ▼] [Sens ━━●━] [Start]     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                     127.5  BPM    ●                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Waveform                                                   │
│  ▁▂▃▅█▅▃▂▁▁▁▁▁▁▂▃▅█▅▃▂▁▁▁▁▁▁▂▃▅█▅▃▂▁▁▁▁▁▁                  │
├─────────────────────────────────────────────────────────────┤
│  Spectrum                                                   │
│  ▅▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                                       │
│  0Hz                                              8000Hz    │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.9 or higher
- Windows 10/11 (optimized for Windows, may work on other platforms)
- A microphone or audio input device
- Windows microphone privacy settings must allow access

## Installation

### 1. Clone the repository

```powershell
git clone <repository-url>
cd bpm_checker
```

### 2. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Or install individually:

```powershell
pip install sounddevice numpy scipy PySide6 pyqtgraph
```

### 4. Run the application

```powershell
python drum_bpm_gui.py
```

## Usage

### Basic Operation

1. **Select your audio device** from the dropdown (your microphone)
2. **Choose sample rate** (44100 Hz is standard, 48000 Hz for pro audio)
3. **Click "Start"** to begin listening
4. **Play your drums** - the app will detect hits and compute BPM
5. Watch the **big BPM number** update as you play
6. The **green dot** flashes on each detected hit

### Controls Explained

| Control | Description |
|---------|-------------|
| **Audio Device** | Select your microphone or audio input |
| **Sample Rate** | 44100 Hz (CD quality) or 48000 Hz (pro audio) |
| **Sensitivity** | Detection threshold (1.0-10.0). Lower = more sensitive |
| **Min Interval** | Minimum ms between detected hits (prevents double-triggers) |
| **High-Pass Filter** | Removes low-frequency rumble (recommended: ON) |
| **Start/Stop** | Begin or end audio capture |
| **Export CSV** | Save onset timestamps and BPM data |

## Calibration Tips

### Getting Good Detection

1. **Start with Sensitivity around 2.0-3.0**
2. **If missing hits**: Lower the sensitivity value
3. **If detecting false hits** (noise): Raise the sensitivity value
4. **If getting double-triggers**: Increase "Min Interval" (try 120-150 ms)
5. **Enable High-Pass** filter if room noise causes false triggers

### Optimal Setup

- Position the microphone **close to your drum/pad** (within 1-2 feet)
- Use a **directional microphone** if available
- Reduce background noise in your environment
- For electronic drums, consider using a direct audio output if available

### Testing Detection

1. Start the app and click "Start"
2. Tap near the microphone with your hand or stick
3. Watch for the **green indicator** to flash on each tap
4. Check that the waveform shows clear peaks for each hit
5. Verify BPM stabilizes when tapping at a steady tempo

## Troubleshooting

### No audio devices found

- Check Windows Sound Settings → Input devices
- Ensure microphone privacy settings allow app access:
  - Settings → Privacy → Microphone → Allow apps to access your microphone

### No hits detected

- Lower the Sensitivity value
- Check the waveform - you should see amplitude spikes when hitting
- Move the microphone closer to the drum

### Too many false hits

- Increase the Sensitivity value
- Enable the High-Pass filter
- Increase the Min Interval value
- Reduce ambient noise

### Audio errors on start

- Try a different audio device
- Try a different sample rate
- Ensure no other application is using the microphone exclusively

## Technical Details

### Algorithm

1. **Audio Capture**: `sounddevice` InputStream with callback (512-sample blocks)
2. **Pre-processing**: Butterworth high-pass filter (50 Hz cutoff, 4th order)
3. **Envelope Detection**: Rectification + moving average (64-sample window)
4. **Adaptive Threshold**: `mean + k × std` of recent energy history
5. **Onset Detection**: Energy exceeds threshold with refractory lockout
6. **BPM Calculation**: Median of inter-onset intervals, IQR outlier rejection

### Performance

- **End-to-end latency**: ~50-55 ms
- **GUI update rate**: 30 FPS
- **CPU usage**: Typically < 5% on modern hardware

## CSV Export Format

The exported CSV contains:

| Column | Description |
|--------|-------------|
| Timestamp | ISO 8601 timestamp of the hit |
| Onset Time (s) | Seconds since first hit |
| BPM | Computed BPM at that moment |

## Building an Executable (Optional)

To create a standalone `.exe` file:

```powershell
pip install pyinstaller
pyinstaller --onefile --windowed --name "DrumBPM" drum_bpm_gui.py
```

The executable will be in the `dist` folder.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio I/O
- [PySide6](https://doc.qt.io/qtforpython/) - Qt GUI framework
- [pyqtgraph](https://www.pyqtgraph.org/) - Fast scientific plotting
