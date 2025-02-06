import sys
import os
import time
import math
import audioread
import tiktoken

from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QLabel, QVBoxLayout, QWidget, QTextEdit, QProgressBar,
                             QHBoxLayout, QMessageBox, QLineEdit, QRadioButton, QButtonGroup,
                             QGridLayout, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import openai

load_dotenv()


# --- Worker Threads ---
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


class AudioLoadingWorker(QThread):
    finished = pyqtSignal(float)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            with audioread.audio_open(self.audio_path) as f:
                total_frames = int(f.channels * f.samplerate * f.duration)
                block_size = f.channels * f.samplerate  # One second chunks
                num_blocks = int(f.duration)  # Number of seconds

                for i in range(num_blocks):
                    try:
                        f.read_data(block_size)  # Read a block of audio data
                    except audioread.exceptions.NoBackendError as e:
                        self.error.emit(f"Audio backend error: {e}")
                        return  # Exit if backend error

                    progress_percentage = int((i / num_blocks) * 100)
                    self.progress.emit(progress_percentage)

                self.finished.emit(f.duration)

        except audioread.exceptions.UnsupportedAudioFile as e:
            self.error.emit(f"Unsupported audio file: {e}")
        except Exception as e:
            self.error.emit(str(e))


class SummaryWorker(QThread):
    finished = pyqtSignal(str, int, int)
    error = pyqtSignal(str)

    def __init__(self, api_key, transcription_text, prompt, model):
        super().__init__()
        self.api_key = api_key
        self.transcription_text = transcription_text
        self.prompt = prompt
        self.model = model
        openai.api_key = self.api_key

    def run(self):
        try:
            messages = [
                {"role": "system",
                 "content":
                 "You are a helpful assistant. Only output the user request."},
                {"role": "user",
                 "content": f"{self.prompt}\n\n{self.transcription_text}"}
            ]

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,  # Adjust as needed
            )
            summary_text = response['choices'][0]['message']['content']

            # Token Usage Info
            input_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            total_tokens = response['usage']['total_tokens']

            self.finished.emit(summary_text, input_tokens,
                               completion_tokens)

        except Exception as e:
            self.error.emit(str(e))


# --- Main Application ---
class AudioTranscriber(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize data members
        self.audio_path = None
        self.transcription_path = None
        self.estimated_cost = None
        self.summary_cost = None
        self.audio_length_seconds = None
        self.transcribed_text = None
        self.summary_input_tokens = 0
        self.summary_output_tokens = 0
        self.selected_model = "gpt-4o-mini"  # Default model

        # API Key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            QMessageBox.critical(
                self, "Error", "API key not found in .env file")
            sys.exit(1)

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Audio to Text Converter')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        # Create Tabs
        self.tabs = QTabWidget()
        self.transcription_tab = QWidget()
        self.summary_tab = QWidget()
        self.tabs.addTab(self.transcription_tab, "Transcription")
        self.tabs.addTab(self.summary_tab, "Summary")

        self.main_layout.addWidget(self.tabs)

        self.init_transcription_tab()
        self.init_summary_tab()

        self.init_common_elements()  # Progress bar, status label, console
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.console)

    def init_transcription_tab(self):
        self.transcription_layout = QGridLayout(self.transcription_tab)

        # File name display labels
        self.audio_file_label = QLabel("Audio File: None", self)
        self.transcription_file_label = QLabel(
            "Transcription File: None", self)

        self.select_audio_button = QPushButton('Select Audio File', self)
        self.select_audio_button.clicked.connect(self.select_audio_file)

        self.convert_button = QPushButton('Convert', self)
        self.convert_button.clicked.connect(self.convert_audio)
        self.convert_button.setEnabled(False)

        # Layout
        self.transcription_layout.addWidget(self.audio_file_label, 0, 0, 1, 2)
        self.transcription_layout.addWidget(
            self.transcription_file_label, 1, 0, 1, 2)
        self.transcription_layout.addWidget(self.select_audio_button, 2, 0)
        self.transcription_layout.addWidget(self.convert_button, 3, 0, 1, 2)

        # Stretch last row
        self.transcription_layout.setRowStretch(6, 1)

    def init_summary_tab(self):
        self.summary_layout = QVBoxLayout(self.summary_tab)

        # Added Transcription File elements
        self.transcription_file_label_summary = QLabel(
            "Transcription File: None", self)

        # New button for loading transcription
        self.load_transcription_button = QPushButton(
            'Load Transcription File', self)
        self.load_transcription_button.clicked.connect(self.load_transcription)

        # Added Token Estimate Label
        self.token_estimate_label = QLabel('Estimated Summary Cost: N/A', self)
        self.token_estimate_label.setAlignment(Qt.AlignCenter)

        self.estimate_label = QLabel('Estimated Cost: N/A', self)
        self.estimate_label.setAlignment(Qt.AlignCenter)

        self.length_label = QLabel('Audio Length: N/A', self)
        self.length_label.setAlignment(Qt.AlignCenter)

        # Model Selection Radio Buttons
        self.model_label = QLabel("Select Summarization Model:", self)
        self.model_group = QButtonGroup()

        self.model_mini_button = QRadioButton(
            "gpt-4o-mini ($0.15/1M in, $0.30/1M out)", self)
        self.model_4o_button = QRadioButton(
            "gpt-4o ($2.5/1M in, $1.25/1M out)", self)

        self.model_mini_button.setChecked(True)  # Set default selection

        self.model_group.addButton(self.model_mini_button)
        self.model_group.addButton(self.model_4o_button)

        self.model_mini_button.toggled.connect(self.update_selected_model)
        self.model_4o_button.toggled.connect(self.update_selected_model)

        # Added "Summarize" button
        self.summarize_button = QPushButton('Summarize', self)
        self.summarize_button.clicked.connect(self.summarize_text)
        self.summarize_button.setEnabled(False)  # Disabled initially

        # Added Token Estimate Label
        self.token_usage_label = QLabel('Token Usage', self)
        self.token_usage_label.setAlignment(Qt.AlignCenter)

        # Added Total cost Estimate Label
        self.total_cost_label = QLabel('Total Cost: N/A', self)
        self.total_cost_label.setAlignment(Qt.AlignCenter)

        # Added Prompt Input Field
        self.prompt_label = QLabel('Summary Prompt:', self)
        self.prompt_input = QLineEdit(self)
        self.prompt_input.setText("create a summary of my meeting")

        # Add widgets to layout
        self.summary_layout.addWidget(self.transcription_file_label_summary)
        self.summary_layout.addWidget(self.load_transcription_button)
        self.summary_layout.addWidget(self.estimate_label)
        self.summary_layout.addWidget(self.length_label)
        self.summary_layout.addWidget(self.model_label)
        self.summary_layout.addWidget(self.model_mini_button)
        self.summary_layout.addWidget(self.model_4o_button)
        self.summary_layout.addWidget(self.token_estimate_label)
        self.summary_layout.addWidget(self.token_usage_label)
        self.summary_layout.addWidget(self.total_cost_label)
        self.summary_layout.addWidget(self.prompt_label)
        self.summary_layout.addWidget(self.prompt_input)
        self.summary_layout.addWidget(self.summarize_button)
        self.summary_layout.addStretch(1)  # Add stretch at the end

    def init_common_elements(self):
        # Console display
        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #f0f0f0;")

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        # Set range for determinate progress
        self.progress_bar.setRange(0, 100)

        # Status label
        self.status_label = QLabel(
            'Select an audio or transcription file', self)
        self.status_label.setAlignment(Qt.AlignCenter)

    def log_to_console(self, message):
        self.console.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def select_audio_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.ogg)"
        )

        if file_path:
            self.audio_path = file_path
            self.transcription_path = None  # Clear transcription path
            self.transcribed_text = None  # Clear transcribed text
            self.log_to_console(f"Selected audio file: {file_path}")
            self.status_label.setText(
                f"File selected: {os.path.basename(file_path)}")
            self.audio_file_label.setText(
                f"Audio File: {os.path.basename(file_path)}")  # Update the audio file label
            self.transcription_file_label.setText(
                "Transcription File: None")  # Reset the transcription file label
            self.estimate_cost()
            self.convert_button.setEnabled(True)
            self.summarize_button.setEnabled(False)

    def load_transcription(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Transcription File",
            "",
            "Text Files (*.txt)"
        )

        if file_path:
            self.audio_path = None  # Clear audio path
            self.audio_file_label.setText(
                "Audio File: None")  # Reset audio label

            self.transcription_path = file_path
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.transcribed_text = file.read()
                self.log_to_console(f"Transcription loaded from: {file_path}")
                self.status_label.setText(
                    f"Transcription loaded from: {os.path.basename(file_path)}")
                self.transcription_file_label.setText(
                    f"Transcription File: {os.path.basename(file_path)}")  # Update the transcription file label
                self.transcription_file_label_summary.setText(
                    f"Transcription File: {os.path.basename(file_path)}")  # Update the transcription file label
                self.convert_button.setEnabled(False)
                self.summarize_button.setEnabled(True)
                self.estimate_summary_cost()  # Estimate cost based on loaded text
                self.estimate_label.setText("Estimated Cost: N/A")
                self.length_label.setText("Audio Length: N/A")

            except Exception as e:
                self.handle_error(f"Error loading transcription: {e}")

    def estimate_cost(self):
        if self.audio_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.log_to_console("Loading audio file...")
            self.status_label.setText('Loading audio file...')

            self.audio_loading_worker = AudioLoadingWorker(self.audio_path)
            self.audio_loading_worker.finished.connect(
                self.handle_audio_loaded)
            self.audio_loading_worker.error.connect(self.handle_error)
            self.audio_loading_worker.progress.connect(
                self.update_loading_progress)
            self.audio_loading_worker.start()
        else:
            self.estimate_label.setText("Estimated Cost: N/A")
            self.length_label.setText("Audio Length: N/A")
            self.estimated_cost = None
            self.audio_length_seconds = None

    def update_loading_progress(self, progress):
        self.progress_bar.setValue(progress)

    def handle_audio_loaded(self, duration):
        self.audio_length_seconds = duration
        duration_minutes = self.audio_length_seconds / 60
        self.estimated_cost = 0.006 * math.ceil(duration_minutes)
        self.estimate_label.setText(
            f"Estimated Cost: ${self.estimated_cost:.4f}")
        self.length_label.setText(
            f"Audio Length: {self.audio_length_seconds:.2f} seconds")
        self.progress_bar.setVisible(False)
        self.status_label.setText(
            f"File selected: {os.path.basename(self.audio_path)}")
        self.log_to_console("Audio file loaded.")

    def convert_audio(self):
        if self.audio_path:
            self.log_to_console("Starting conversion...")
            self.status_label.setText('Converting audio to text...')
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Makes it indeterminate
            self.convert_audio_to_text(self.audio_path)
            self.convert_button.setEnabled(False)
            self.summarize_button.setEnabled(
                False)  # Disable until transcription is done

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
        self.transcription_path = f"{file_path_without_ext}_transcription.txt"

        # Save the transcribed text
        with open(self.transcription_path, 'w', encoding='utf-8') as text_file:
            text_file.write(transcribed_text)

        self.log_to_console(
            f"Transcription saved to: {self.transcription_path}")
        self.status_label.setText('Transcription completed successfully!')

        self.transcription_file_label.setText(
            f"Transcription File: {os.path.basename(self.transcription_path)}")
        self.transcription_file_label_summary.setText(
            f"Transcription File: {os.path.basename(self.transcription_path)}")

        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        self.summarize_button.setEnabled(True)  # Enable summarize button

        self.transcribed_text = transcribed_text
        self.estimate_summary_cost()  # Estimate after transcription

        self.estimate_label.setText(
            f"Final Estimated Cost: ${self.estimated_cost:.4f}")
        if self.audio_length_seconds:
            self.length_label.setText(
                f"Audio Length: {self.audio_length_seconds:.2f} seconds")
        else:
            self.length_label.setText("Audio Length: Unknown")

    def estimate_summary_cost(self):
        if not self.transcribed_text:
            self.token_estimate_label.setText("Estimated Summary Cost: N/A")
            return

        model_name = self.selected_model
        encoding = tiktoken.encoding_for_model(model_name)

        num_tokens = len(encoding.encode(self.transcribed_text))

        if self.selected_model == "gpt-4o-mini":
            cost_per_token = 0.15 / 1_000_000  # $0.15 per 1 million tokens
        elif self.selected_model == "gpt-4o":
            cost_per_token = 2.5 / 1_000_000
        else:
            cost_per_token = 0  # should not happen

        estimated_cost = num_tokens * cost_per_token

        self.token_estimate_label.setText(
            f"Estimated Summary Cost: ${estimated_cost:.4f} (Using {self.selected_model})")

    def update_selected_model(self):
        if self.model_mini_button.isChecked():
            self.selected_model = "gpt-4o-mini"
        elif self.model_4o_button.isChecked():
            self.selected_model = "gpt-4o"
        self.log_to_console(
            f"Summarization model selected: {self.selected_model}")
        if self.transcribed_text:
            self.estimate_summary_cost()  # Recalculate cost when model changes

    def summarize_text(self):
        if not self.transcribed_text:
            QMessageBox.warning(
                self, "Warning", "No transcription available to summarize.")
            return

        prompt = self.prompt_input.text()

        self.log_to_console("Starting summarization...")
        self.status_label.setText('Summarizing the transcription...')
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        self.summary_worker = SummaryWorker(
            self.api_key, self.transcribed_text, prompt, self.selected_model)
        self.summary_worker.finished.connect(self.handle_summary_complete)
        self.summary_worker.error.connect(self.handle_error)
        self.summary_worker.start()

    def handle_summary_complete(self, summary_text, input_tokens, output_tokens):
        # Create output file path
        file_path_without_ext = os.path.splitext(self.audio_path)[0]
        summary_path = f"{file_path_without_ext}_AI_summary.txt"

        # Save the summary text
        with open(summary_path, 'w', encoding='utf-8') as text_file:
            text_file.write(summary_text)

        self.log_to_console(f"Summary saved to: {summary_path}")
        self.status_label.setText('Summary completed successfully!')
        self.progress_bar.setVisible(False)

        self.summary_input_tokens = input_tokens
        self.summary_output_tokens = output_tokens
        self.update_total_cost()
        self.update_token_usage_label()

    def update_total_cost(self):
        if self.selected_model == "gpt-4o-mini":
            input_cost = (self.summary_input_tokens / 1_000_000) * 0.15
            output_cost = (self.summary_output_tokens / 1_000_000) * 0.30
        elif self.selected_model == "gpt-4o":
            input_cost = (self.summary_input_tokens / 1_000_000) * 2.5
            output_cost = (self.summary_output_tokens / 1_000_000) * 1.25
        else:
            input_cost = 0
            output_cost = 0

        total_cost = input_cost + output_cost
        self.total_cost = total_cost
        self.total_cost_label.setText(
            f"Total Cost: ${total_cost:.4f} (Using {self.selected_model})")

    def update_token_usage_label(self):
        self.token_usage_label.setText(
            f"Input Tokens: {self.summary_input_tokens}, Output Tokens: {self.summary_output_tokens}")

    def handle_error(self, error_message):
        self.log_to_console(f"Error: {error_message}")
        self.status_label.setText(f'Error: {error_message}')
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        self.summarize_button.setEnabled(True)

    def closeEvent(self, event):
        event.accept()
        QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = AudioTranscriber()
    w.show()
    app.exec_()
