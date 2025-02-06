import sys
import os
from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QLabel, QVBoxLayout, QWidget, QTextEdit, QProgressBar,
                             QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import openai
import time
import math
# import librosa # Remove librosa import
import soundfile as sf  # Import soundfile
import audioread  # Import audioread


load_dotenv()


class TranscriptionWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, api_key, audio_path):
        super().__init__()
        self.api_key = api_key
        self.audio_path = audio_path
        openai.api_key = self.api_key

    def run(self):
        try:
            with open(self.audio_path, 'rb') as audio_file:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file
                )
            self.finished.emit(response['text'])
        except Exception as e:
            self.error.emit(str(e))


class AudioTranscriber(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_path = None
        self.estimated_cost = None
        self.audio_length_seconds = None
        self.initUI()

        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            QMessageBox.critical(
                self, "Error", "API key not found in .env file")
            sys.exit(1)

    def initUI(self):
        self.setWindowTitle('Audio to Text Converter')
        self.setGeometry(100, 100, 600, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create console display
        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #f0f0f0;")

        # Create indeterminate progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Makes it indeterminate

        self.status_label = QLabel('Select an audio file to convert', self)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton('Select Audio File', self)
        self.select_button.clicked.connect(self.select_file)

        self.estimate_label = QLabel('Estimated Cost: N/A', self)
        self.estimate_label.setAlignment(Qt.AlignCenter)

        self.length_label = QLabel('Audio Length: N/A', self)
        self.length_label.setAlignment(Qt.AlignCenter)

        self.convert_button = QPushButton('Convert', self)
        self.convert_button.clicked.connect(self.convert_audio)
        self.convert_button.setEnabled(False)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.convert_button)

        # Add widgets to layout
        layout.addWidget(self.console)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.estimate_label)
        layout.addWidget(self.length_label)
        layout.addLayout(button_layout)

    def log_to_console(self, message):
        self.console.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def select_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.ogg)"
        )

        if file_path:
            self.audio_path = file_path
            self.log_to_console(f"Selected file: {file_path}")
            self.status_label.setText(
                f"File selected: {os.path.basename(file_path)}")
            self.estimate_cost()
            self.convert_button.setEnabled(True)

    def estimate_cost(self):
        if self.audio_path:
            try:
                # Use audioread to get audio duration
                with audioread.audio_open(self.audio_path) as f:
                    self.audio_length_seconds = f.duration
                duration_minutes = self.audio_length_seconds / 60
                self.estimated_cost = 0.006 * math.ceil(duration_minutes)
                self.estimate_label.setText(
                    f"Estimated Cost: ${self.estimated_cost:.4f}")
                self.length_label.setText(
                    f"Audio Length: {self.audio_length_seconds:.2f} seconds")

            except Exception as e:
                QMessageBox.warning(
                    self, "Warning", f"Could not load audio file with audioread.  Using file size estimate. Error: {e}")
                file_size_bytes = os.path.getsize(self.audio_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
                duration_minutes = file_size_mb * 8
                self.estimated_cost = 0.006 * math.ceil(duration_minutes)
                self.estimate_label.setText(
                    f"Estimated Cost: ${self.estimated_cost:.4f} (File size estimate)")
                self.length_label.setText(
                    "Audio Length: Unknown (File size estimate)")
                self.audio_length_seconds = None
        else:
            self.estimate_label.setText("Estimated Cost: N/A")
            self.length_label.setText("Audio Length: N/A")
            self.estimated_cost = None
            self.audio_length_seconds = None

    def convert_audio(self):
        if self.audio_path:
            self.log_to_console("Starting conversion...")
            self.status_label.setText('Converting audio to text...')
            self.progress_bar.setVisible(True)
            self.convert_audio_to_text(self.audio_path)
            self.convert_button.setEnabled(False)
        else:
            QMessageBox.warning(self, "Warning", "No audio file selected.")

    def convert_audio_to_text(self, audio_path):
        # Create and start worker thread
        self.worker = TranscriptionWorker(self.api_key, audio_path)
        self.worker.finished.connect(
            lambda text: self.handle_transcription_complete(text, audio_path))
        self.worker.error.connect(self.handle_error)
        self.worker.start()

    def handle_transcription_complete(self, transcribed_text, audio_path):
        # Create output file path
        file_path_without_ext = os.path.splitext(audio_path)[0]
        output_path = f"{file_path_without_ext}_transcription.txt"

        # Save the transcribed text
        with open(output_path, 'w', encoding='utf-8') as text_file:
            text_file.write(transcribed_text)

        self.log_to_console(f"Transcription saved to: {output_path}")
        self.status_label.setText('Transcription completed successfully!')
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        self.estimate_label.setText(
            f"Final Estimated Cost: ${self.estimated_cost:.4f}")
        if self.audio_length_seconds:
            self.length_label.setText(
                f"Audio Length: {self.audio_length_seconds:.2f} seconds")
        else:
            self.length_label.setText("Audio Length: Unknown")

    def handle_error(self, error_message):
        self.log_to_console(f"Error: {error_message}")
        self.status_label.setText(f'Error: {error_message}')
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)

    def closeEvent(self, event):
        event.accept()
        QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = AudioTranscriber()
    w.show()
    app.exec_()
