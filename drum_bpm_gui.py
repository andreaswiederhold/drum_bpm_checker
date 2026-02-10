#!/usr/bin/env python3
"""
================================================================================
DRUM BPM TIMING TRAINER - Real-Time Drum Hit Detection & BPM Display
================================================================================

FEATURES:
---------
- Real-time microphone capture with selectable audio device
- Onset detection using energy envelope with adaptive threshold
- BPM computation from inter-onset intervals (IOI) with outlier rejection
- Live waveform display (last 2 seconds)
- Live frequency spectrum (FFT with Hann window)
- Adjustable sensitivity, refractory period, and high-pass filter
- CSV export of onset timestamps and BPM readings

CONTROLS:
---------
- Audio Device: Select your microphone from the dropdown
- Sample Rate: Choose 44100 Hz or 48000 Hz
- Sensitivity: Adjust detection threshold (lower = more sensitive)
- Min Interval: Minimum time between detected hits (prevents double triggers)
- High-Pass Filter: Toggle to reduce low-frequency rumble
- Start/Stop: Begin or end audio capture
- Export CSV: Save onset data to file

CALIBRATION TIPS:
-----------------
1. Start with Sensitivity around 2.0-3.0
2. If missing hits, lower Sensitivity
3. If detecting false hits (noise), raise Sensitivity
4. If getting double-triggers on single hits, increase Min Interval
5. Enable High-Pass filter if room noise/rumble causes false triggers
6. For best results, position the mic close to your drum/pad

REQUIREMENTS:
-------------
pip install sounddevice numpy scipy PySide6 pyqtgraph

================================================================================
"""

import sys
import time
import csv
from collections import deque
from datetime import datetime
from threading import Thread, Lock, Event
import numpy as np
from scipy.signal import butter, lfilter

import sounddevice as sd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QCheckBox, QGroupBox,
    QFileDialog, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import pyqtgraph as pg


# -----------------------------------------------------------------------------
# CONSTANTS & DEFAULTS
# -----------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 44100
BLOCK_SIZE = 512  # Audio callback block size
BUFFER_DURATION = 3.0  # Seconds of audio to keep in buffer
WAVEFORM_DISPLAY_DURATION = 2.0  # Seconds of waveform to display
GUI_UPDATE_INTERVAL_MS = 33  # ~30 FPS GUI updates
DEFAULT_SENSITIVITY = 2.5  # Threshold multiplier (mean + k*std)
DEFAULT_REFRACTORY_MS = 100  # Minimum ms between onsets
DEFAULT_HIGHPASS_CUTOFF = 50  # Hz
BPM_HISTORY_LENGTH = 10  # Number of IOIs to keep for BPM calculation
MIN_BPM = 30
MAX_BPM = 300


# -----------------------------------------------------------------------------
# SIGNAL PROCESSING UTILITIES
# -----------------------------------------------------------------------------
def design_highpass_filter(cutoff, fs, order=4):
    """Design a Butterworth high-pass filter."""
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    # Clamp to valid range
    normalized_cutoff = max(0.001, min(0.999, normalized_cutoff))
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)
    return b, a


def compute_energy_envelope(signal, window_size=64):
    """
    Compute energy envelope using rectification and moving average.
    This is efficient and works well for transient detection.
    """
    # Rectify (absolute value)
    rectified = np.abs(signal)
    # Moving average using convolution
    if len(rectified) < window_size:
        return rectified
    kernel = np.ones(window_size) / window_size
    envelope = np.convolve(rectified, kernel, mode='same')
    return envelope


def compute_spectrum(signal, fs, window_size=2048):
    """
    Compute frequency spectrum using FFT with Hann window.
    Returns frequencies and magnitude in dB.
    """
    if len(signal) < window_size:
        # Pad with zeros if signal is too short
        padded = np.zeros(window_size)
        padded[:len(signal)] = signal
        signal = padded
    else:
        signal = signal[-window_size:]
    
    # Apply Hann window
    window = np.hanning(window_size)
    windowed = signal * window
    
    # Compute FFT
    fft_result = np.fft.rfft(windowed)
    magnitude = np.abs(fft_result)
    
    # Convert to dB (with floor to avoid log(0))
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Frequency axis
    frequencies = np.fft.rfftfreq(window_size, 1.0 / fs)
    
    return frequencies, magnitude_db


# -----------------------------------------------------------------------------
# AUDIO PROCESSOR CLASS
# -----------------------------------------------------------------------------
class AudioProcessor:
    """
    Handles audio capture, onset detection, and BPM computation.
    Uses a producer-consumer pattern with thread-safe buffers.
    """
    
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, device=None):
        self.sample_rate = sample_rate
        self.device = device
        self.block_size = BLOCK_SIZE
        
        # Thread-safe audio buffer (ring buffer using deque)
        buffer_samples = int(BUFFER_DURATION * sample_rate)
        self.audio_buffer = deque(maxlen=buffer_samples)
        self.buffer_lock = Lock()
        
        # Onset detection state
        self.sensitivity = DEFAULT_SENSITIVITY
        self.refractory_samples = int(DEFAULT_REFRACTORY_MS * sample_rate / 1000)
        self.highpass_enabled = True
        self.highpass_b, self.highpass_a = design_highpass_filter(
            DEFAULT_HIGHPASS_CUTOFF, sample_rate
        )
        self.filter_state = None
        
        # Onset tracking
        self.onset_times = deque(maxlen=BPM_HISTORY_LENGTH + 1)
        self.last_onset_sample = -self.refractory_samples
        self.total_samples_processed = 0
        self.onset_timestamps = []  # For CSV export
        self.bpm_history = []  # For CSV export
        
        # Energy envelope history for adaptive threshold
        self.energy_history = deque(maxlen=int(sample_rate * 0.5))  # 500ms of history
        
        # Current state (thread-safe access)
        self.current_bpm = 0.0
        self.current_bpm_lock = Lock()
        self.last_onset_detected = False
        
        # Stream control
        self.stream = None
        self.running = False
        self.processing_thread = None
        self.stop_event = Event()
    
    def set_sensitivity(self, value):
        """Set detection sensitivity (threshold multiplier)."""
        self.sensitivity = value
    
    def set_refractory_ms(self, ms):
        """Set minimum interval between onsets."""
        self.refractory_samples = int(ms * self.sample_rate / 1000)
    
    def set_highpass_enabled(self, enabled):
        """Enable/disable high-pass filter."""
        self.highpass_enabled = enabled
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Audio stream callback - runs in audio thread.
        Only does minimal work: copies data to buffer.
        """
        if status:
            print(f"Audio callback status: {status}")
        
        # Get mono channel
        mono = indata[:, 0].copy()
        
        # Thread-safe buffer append
        with self.buffer_lock:
            self.audio_buffer.extend(mono)
    
    def process_audio(self):
        """
        Processing thread - detects onsets and computes BPM.
        Runs in a separate thread to avoid blocking audio callback.
        """
        process_chunk_size = self.block_size * 2
        
        while not self.stop_event.is_set():
            # Get data from buffer
            with self.buffer_lock:
                if len(self.audio_buffer) < process_chunk_size:
                    time.sleep(0.005)
                    continue
                chunk = np.array(list(self.audio_buffer)[-process_chunk_size:])
            
            # Apply high-pass filter if enabled
            if self.highpass_enabled:
                if self.filter_state is None:
                    self.filter_state = np.zeros(max(len(self.highpass_a), len(self.highpass_b)) - 1)
                try:
                    filtered, self.filter_state = lfilter(
                        self.highpass_b, self.highpass_a, chunk, zi=self.filter_state
                    )
                except ValueError:
                    # Reset filter state if dimensions don't match
                    self.filter_state = np.zeros(max(len(self.highpass_a), len(self.highpass_b)) - 1)
                    filtered = chunk
            else:
                filtered = chunk
            
            # Compute energy envelope
            envelope = compute_energy_envelope(filtered)
            
            # Update energy history
            self.energy_history.extend(envelope)
            
            # Compute adaptive threshold
            if len(self.energy_history) > 100:
                energy_arr = np.array(self.energy_history)
                mean_energy = np.mean(energy_arr)
                std_energy = np.std(energy_arr)
                threshold = mean_energy + self.sensitivity * std_energy
            else:
                threshold = 0.01 * self.sensitivity
            
            # Detect onset
            current_energy = np.max(envelope)
            samples_since_last = self.total_samples_processed - self.last_onset_sample
            
            self.last_onset_detected = False
            if current_energy > threshold and samples_since_last > self.refractory_samples:
                # Onset detected!
                self.last_onset_detected = True
                current_time = time.time()
                self.onset_times.append(current_time)
                self.last_onset_sample = self.total_samples_processed
                
                # Record for export
                self.onset_timestamps.append(current_time)
                
                # Compute BPM from IOIs
                self._update_bpm()
            
            self.total_samples_processed += len(chunk) // 2
            time.sleep(0.005)
    
    def _update_bpm(self):
        """Compute BPM from recent inter-onset intervals."""
        if len(self.onset_times) < 2:
            return
        
        times = list(self.onset_times)
        iois = np.diff(times)  # Inter-onset intervals in seconds
        
        if len(iois) == 0:
            return
        
        # Filter out outliers using IQR
        if len(iois) >= 4:
            q1 = np.percentile(iois, 25)
            q3 = np.percentile(iois, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            valid_iois = iois[(iois >= lower) & (iois <= upper)]
        else:
            valid_iois = iois
        
        if len(valid_iois) == 0:
            valid_iois = iois
        
        # Use median for robustness
        median_ioi = np.median(valid_iois)
        
        if median_ioi > 0:
            bpm = 60.0 / median_ioi
            # Clamp to reasonable range
            bpm = max(MIN_BPM, min(MAX_BPM, bpm))
            
            with self.current_bpm_lock:
                self.current_bpm = bpm
                self.bpm_history.append((time.time(), bpm))
    
    def get_current_bpm(self):
        """Thread-safe getter for current BPM."""
        with self.current_bpm_lock:
            return self.current_bpm
    
    def get_waveform_data(self, duration=WAVEFORM_DISPLAY_DURATION):
        """Get recent audio data for waveform display."""
        samples = int(duration * self.sample_rate)
        with self.buffer_lock:
            if len(self.audio_buffer) < samples:
                data = np.array(list(self.audio_buffer))
            else:
                data = np.array(list(self.audio_buffer)[-samples:])
        return data
    
    def get_spectrum_data(self):
        """Get current frequency spectrum."""
        with self.buffer_lock:
            if len(self.audio_buffer) < 2048:
                return np.array([]), np.array([])
            data = np.array(list(self.audio_buffer)[-4096:])
        return compute_spectrum(data, self.sample_rate)
    
    def start(self):
        """Start audio capture and processing."""
        if self.running:
            return
        
        self.stop_event.clear()
        self.filter_state = None
        
        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype='float32',
                callback=self.audio_callback
            )
            self.stream.start()
            
            # Start processing thread
            self.processing_thread = Thread(target=self.process_audio, daemon=True)
            self.processing_thread.start()
            
            self.running = True
        except Exception as e:
            raise RuntimeError(f"Failed to start audio stream: {e}")
    
    def stop(self):
        """Stop audio capture and processing."""
        if not self.running:
            return
        
        self.stop_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.running = False
    
    def reset(self):
        """Reset all state for new session."""
        with self.buffer_lock:
            self.audio_buffer.clear()
        self.onset_times.clear()
        self.onset_timestamps.clear()
        self.bpm_history.clear()
        self.energy_history.clear()
        self.last_onset_sample = -self.refractory_samples
        self.total_samples_processed = 0
        with self.current_bpm_lock:
            self.current_bpm = 0.0
    
    def export_to_csv(self, filepath):
        """Export onset data to CSV file."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Onset Time (s)', 'BPM'])
            
            start_time = self.onset_timestamps[0] if self.onset_timestamps else 0
            bpm_dict = dict(self.bpm_history)
            
            for i, onset_time in enumerate(self.onset_timestamps):
                relative_time = onset_time - start_time
                # Find closest BPM reading
                bpm = 0
                for t, b in self.bpm_history:
                    if t <= onset_time:
                        bpm = b
                    else:
                        break
                writer.writerow([
                    datetime.fromtimestamp(onset_time).isoformat(),
                    f"{relative_time:.3f}",
                    f"{bpm:.1f}"
                ])


# -----------------------------------------------------------------------------
# MAIN GUI WINDOW
# -----------------------------------------------------------------------------
class DrumBPMWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drum BPM Timing Trainer")
        self.setMinimumSize(900, 700)
        
        # Initialize audio processor
        self.processor = None
        self.current_device = None
        self.current_sample_rate = DEFAULT_SAMPLE_RATE
        
        # Build UI
        self._build_ui()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        
        # Onset flash state
        self.onset_flash_counter = 0
    
    def _build_ui(self):
        """Build the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- Control Panel ---
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)
        
        # Device selector
        device_layout = QVBoxLayout()
        device_layout.addWidget(QLabel("Audio Device:"))
        self.device_combo = QComboBox()
        self._populate_devices()
        device_layout.addWidget(self.device_combo)
        control_layout.addLayout(device_layout)
        
        # Sample rate selector
        rate_layout = QVBoxLayout()
        rate_layout.addWidget(QLabel("Sample Rate:"))
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["44100 Hz", "48000 Hz"])
        rate_layout.addWidget(self.rate_combo)
        control_layout.addLayout(rate_layout)
        
        # Sensitivity slider
        sens_layout = QVBoxLayout()
        sens_layout.addWidget(QLabel("Sensitivity:"))
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(10, 100)  # 1.0 to 10.0
        self.sens_slider.setValue(int(DEFAULT_SENSITIVITY * 10))
        self.sens_slider.valueChanged.connect(self._on_sensitivity_changed)
        sens_layout.addWidget(self.sens_slider)
        self.sens_label = QLabel(f"{DEFAULT_SENSITIVITY:.1f}")
        sens_layout.addWidget(self.sens_label)
        control_layout.addLayout(sens_layout)
        
        # Refractory slider (min interval)
        refr_layout = QVBoxLayout()
        refr_layout.addWidget(QLabel("Min Interval (ms):"))
        self.refr_slider = QSlider(Qt.Horizontal)
        self.refr_slider.setRange(50, 300)
        self.refr_slider.setValue(DEFAULT_REFRACTORY_MS)
        self.refr_slider.valueChanged.connect(self._on_refractory_changed)
        refr_layout.addWidget(self.refr_slider)
        self.refr_label = QLabel(f"{DEFAULT_REFRACTORY_MS} ms")
        refr_layout.addWidget(self.refr_label)
        control_layout.addLayout(refr_layout)
        
        # High-pass toggle
        hp_layout = QVBoxLayout()
        hp_layout.addWidget(QLabel("High-Pass Filter:"))
        self.hp_checkbox = QCheckBox("Enable (50 Hz)")
        self.hp_checkbox.setChecked(True)
        self.hp_checkbox.stateChanged.connect(self._on_highpass_changed)
        hp_layout.addWidget(self.hp_checkbox)
        control_layout.addLayout(hp_layout)
        
        # Start/Stop button
        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.clicked.connect(self._toggle_capture)
        control_layout.addWidget(self.start_btn)
        
        # Export button
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._export_csv)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)
        
        main_layout.addWidget(control_group)
        
        # --- BPM Display ---
        bpm_frame = QFrame()
        bpm_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        bpm_layout = QHBoxLayout(bpm_frame)
        
        self.bpm_label = QLabel("---")
        self.bpm_label.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 72, QFont.Bold)
        self.bpm_label.setFont(font)
        self.bpm_label.setStyleSheet("color: #2196F3;")
        bpm_layout.addWidget(self.bpm_label)
        
        self.bpm_unit_label = QLabel("BPM")
        self.bpm_unit_label.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        unit_font = QFont("Arial", 24)
        self.bpm_unit_label.setFont(unit_font)
        bpm_layout.addWidget(self.bpm_unit_label)
        
        # Onset indicator
        self.onset_indicator = QLabel("●")
        self.onset_indicator.setFont(QFont("Arial", 48))
        self.onset_indicator.setStyleSheet("color: #666666;")
        bpm_layout.addWidget(self.onset_indicator)
        
        main_layout.addWidget(bpm_frame)
        
        # --- Waveform Plot ---
        waveform_group = QGroupBox("Waveform")
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.setLabel('bottom', 'Time', 's')
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_plot.setBackground('#1e1e1e')
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen('#4CAF50', width=1))
        waveform_layout.addWidget(self.waveform_plot)
        
        main_layout.addWidget(waveform_group)
        
        # --- Spectrum Plot ---
        spectrum_group = QGroupBox("Frequency Spectrum")
        spectrum_layout = QVBoxLayout(spectrum_group)
        
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Magnitude', 'dB')
        self.spectrum_plot.setLabel('bottom', 'Frequency', 'Hz')
        self.spectrum_plot.setXRange(0, 8000)
        self.spectrum_plot.setYRange(-60, 0)
        self.spectrum_plot.setBackground('#1e1e1e')
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen('#FF9800', width=1))
        spectrum_layout.addWidget(self.spectrum_plot)
        
        main_layout.addWidget(spectrum_group)
        
        # --- Status Bar ---
        self.statusBar().showMessage("Ready - Select device and click Start")
    
    def _populate_devices(self):
        """Populate device dropdown with available input devices."""
        self.device_combo.clear()
        devices = sd.query_devices()
        
        default_input = sd.default.device[0]
        default_index = 0
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                name = f"{device['name']} (#{i})"
                self.device_combo.addItem(name, i)
                if i == default_input:
                    default_index = self.device_combo.count() - 1
        
        self.device_combo.setCurrentIndex(default_index)
    
    def _on_sensitivity_changed(self, value):
        """Handle sensitivity slider change."""
        sensitivity = value / 10.0
        self.sens_label.setText(f"{sensitivity:.1f}")
        if self.processor:
            self.processor.set_sensitivity(sensitivity)
    
    def _on_refractory_changed(self, value):
        """Handle refractory slider change."""
        self.refr_label.setText(f"{value} ms")
        if self.processor:
            self.processor.set_refractory_ms(value)
    
    def _on_highpass_changed(self, state):
        """Handle high-pass filter toggle."""
        if self.processor:
            self.processor.set_highpass_enabled(state == Qt.Checked)
    
    def _toggle_capture(self):
        """Start or stop audio capture."""
        if self.processor and self.processor.running:
            self._stop_capture()
        else:
            self._start_capture()
    
    def _start_capture(self):
        """Start audio capture."""
        try:
            device_index = self.device_combo.currentData()
            sample_rate = 44100 if self.rate_combo.currentIndex() == 0 else 48000
            
            self.processor = AudioProcessor(sample_rate=sample_rate, device=device_index)
            self.processor.set_sensitivity(self.sens_slider.value() / 10.0)
            self.processor.set_refractory_ms(self.refr_slider.value())
            self.processor.set_highpass_enabled(self.hp_checkbox.isChecked())
            
            self.processor.start()
            self.current_sample_rate = sample_rate
            
            self.start_btn.setText("Stop")
            self.start_btn.setStyleSheet("background-color: #f44336;")
            self.export_btn.setEnabled(True)
            self.device_combo.setEnabled(False)
            self.rate_combo.setEnabled(False)
            
            self.update_timer.start(GUI_UPDATE_INTERVAL_MS)
            self.statusBar().showMessage("Listening... Tap drums to detect BPM")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start: {e}")
    
    def _stop_capture(self):
        """Stop audio capture."""
        self.update_timer.stop()
        
        if self.processor:
            self.processor.stop()
        
        self.start_btn.setText("Start")
        self.start_btn.setStyleSheet("")
        self.device_combo.setEnabled(True)
        self.rate_combo.setEnabled(True)
        
        self.statusBar().showMessage("Stopped - Click Start to begin")
    
    def _update_display(self):
        """Update GUI with current data (called by QTimer)."""
        if not self.processor or not self.processor.running:
            return
        
        # Update BPM display
        bpm = self.processor.get_current_bpm()
        if bpm > 0:
            self.bpm_label.setText(f"{bpm:.1f}")
        else:
            self.bpm_label.setText("---")
        
        # Update onset indicator (flash green on hit)
        if self.processor.last_onset_detected:
            self.onset_indicator.setStyleSheet("color: #4CAF50;")  # Green
            self.onset_flash_counter = 3  # Flash for 3 frames
        elif self.onset_flash_counter > 0:
            self.onset_flash_counter -= 1
            if self.onset_flash_counter == 0:
                self.onset_indicator.setStyleSheet("color: #666666;")  # Gray
        
        # Update waveform
        waveform_data = self.processor.get_waveform_data()
        if len(waveform_data) > 0:
            time_axis = np.linspace(
                -len(waveform_data) / self.current_sample_rate, 0,
                len(waveform_data)
            )
            self.waveform_curve.setData(time_axis, waveform_data)
        
        # Update spectrum
        freqs, magnitudes = self.processor.get_spectrum_data()
        if len(freqs) > 0:
            self.spectrum_curve.setData(freqs, magnitudes)
    
    def _export_csv(self):
        """Export onset data to CSV."""
        if not self.processor or len(self.processor.onset_timestamps) == 0:
            QMessageBox.warning(self, "No Data", "No onset data to export.")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv)"
        )
        if filepath:
            try:
                self.processor.export_to_csv(filepath)
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {len(self.processor.onset_timestamps)} onsets to {filepath}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    def closeEvent(self, event):
        """Clean up on window close."""
        self._stop_capture()
        event.accept()


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Apply dark theme
    palette = app.palette()
    from PySide6.QtGui import QPalette, QColor
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, QColor(208, 208, 208))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, QColor(208, 208, 208))
    palette.setColor(QPalette.ToolTipText, QColor(208, 208, 208))
    palette.setColor(QPalette.Text, QColor(208, 208, 208))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(208, 208, 208))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = DrumBPMWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
