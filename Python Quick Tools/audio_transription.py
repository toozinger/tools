import sys
import os
import time
import math
import audioread
import tiktoken
import io
import subprocess
import concurrent.futures

from dotenv import load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QLabel, QVBoxLayout, QWidget, QTextEdit, QProgressBar,
                             QHBoxLayout, QMessageBox, QLineEdit, QRadioButton, QButtonGroup,
                             QGridLayout, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QCoreApplication

import openai

load_dotenv()


# --- Functions (No More QThreads) ---
def transcribe_audio(api_key, audio_path, audio_chunk):
    """Transcribes a single audio chunk."""
    openai.api_key = api_key
    try:
        audio_buffer = io.BytesIO(audio_chunk)
        audio_buffer.name = "chunk.mp3"  # Required by OpenAI

        files = {"file": audio_buffer}
        data = {"model": "whisper-1"}
        response = openai.Audio.transcribe(**files, **data)
        return response['text']
    except Exception as e:
        raise e  # Re-raise exception to be handled by caller


def load_audio_duration(audio_path, progress_callback, error_callback):
    """Loads audio file and returns duration in seconds."""
    try:
        with audioread.audio_open(audio_path) as f:
            total_frames = int(f.channels * f.samplerate * f.duration)
            block_size = f.channels * f.samplerate  # One second chunks
            num_blocks = int(f.duration)  # Number of seconds
            update_interval = max(1, num_blocks // 100)

            for i in range(num_blocks):
                try:
                    f.read_data(block_size)
                except audioread.exceptions.NoBackendError as e:
                    error_callback(f"Audio backend error: {e}")
                    return None  # Signal error
                if (i + 1) % update_interval == 0 or i == num_blocks - 1:
                    progress_percentage = int(((i + 1) / num_blocks) * 100)
                    progress_callback(progress_percentage)

            return f.duration

    except audioread.exceptions.UnsupportedAudioFile as e:
        error_callback(f"Unsupported audio file: {e}")
        return None
    except Exception as e:
        error_callback(str(e))
        return None


def summarize_text(api_key, transcription_text, prompt, model):
    """Summarizes the given text using OpenAI ChatCompletion."""
    openai.api_key = api_key
    try:
        messages = [
            {"role": "system",
             "content":
             "You are a helpful assistant. Only output the user request."},
            {"role": "user",
             "content": f"{prompt}\n\n{transcription_text}"}
        ]

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=2048,  # Adjust as needed
        )
        summary_text = response['choices'][0]['message']['content']

        # Token Usage Info
        input_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        total_tokens = response['usage']['total_tokens']

        return summary_text, input_tokens, completion_tokens

    except Exception as e:
        raise e


# --- Main Application ---
class AudioTranscriber(QMainWindow):
    # Signals for updating the UI from long operations
    loading_progress = pyqtSignal(int)
    transcription_complete = pyqtSignal(str)  # Pass transcribed text
    summary_complete = pyqtSignal(str, int, int)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Initialize data members
        self.audio_path = None
        self.transcription_path = None
        self.estimated_cost = None
        self.summary_cost = None
        self.audio_length_seconds = None
        self.transcribed_text = ""
        self.summary_input_tokens = 0
        self.summary_output_tokens = 0
        self.selected_model = "gpt-4o-mini"  # Default model
        self.chunk_size = 20 * 1024 * 1024  # 20MB
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            QMessageBox.critical(
                self, "Error", "API key not found in .env file")
            sys.exit(1)

        self.initUI()

        # Thread Pool for background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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

        self.init_common_elements()

        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.console)

        # Connect Signals to Slots
        self.loading_progress.connect(self.update_loading_progress)
        self.transcription_complete.connect(self.handle_transcription_complete)
        self.summary_complete.connect(self.handle_summary_complete)
        self.error_signal.connect(self.handle_error)

    def init_transcription_tab(self):
        self.transcription_layout = QGridLayout(self.transcription_tab)

        # File name display labels
        self.audio_file_label = QLabel("Audio File: None", self)
        self.transcription_file_label = QLabel(
            "Transcription File: None", self)

        # Labels for displaying audio length and cost
        self.length_label = QLabel('Audio Length: N/A', self)
        self.estimate_label = QLabel('Estimated Cost: N/A', self)

        self.select_audio_button = QPushButton('Select Audio File', self)
        self.select_audio_button.clicked.connect(self.select_audio_file)

        self.convert_button = QPushButton('Convert', self)
        self.convert_button.clicked.connect(self.start_transcription)
        self.convert_button.setEnabled(False)

        # Layout
        self.transcription_layout.addWidget(self.audio_file_label, 0, 0, 1, 2)
        self.transcription_layout.addWidget(
            self.transcription_file_label, 1, 0, 1, 2)
        self.transcription_layout.addWidget(self.select_audio_button, 2, 0)
        self.transcription_layout.addWidget(self.convert_button, 3, 0, 1, 2)

        self.transcription_layout.addWidget(self.length_label, 4, 0, 1, 2)
        self.transcription_layout.addWidget(self.estimate_label, 5, 0, 1, 2)

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

        # Model Selection Radio Buttons
        self.model_label = QLabel("Select Summarization Model:", self)
        self.model_group = QButtonGroup()

        self.model_mini_button = QRadioButton(
            "gpt-4o-mini ($0.15/1M in, $0.30/1M out)", self)
        self.model_4o_button = QRadioButton(
            "gpt-4o ($2.5/1M in, $1.25/1M out)", self)

        self.model_mini_button.setChecked(True)  # Default selection

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
        self.summary_layout.addWidget(self.model_label)
        self.summary_layout.addWidget(self.model_mini_button)
        self.summary_layout.addWidget(self.model_4o_button)
        self.summary_layout.addWidget(self.token_estimate_label)
        self.summary_layout.addWidget(self.token_usage_label)
        self.summary_layout.addWidget(self.total_cost_label)
        self.summary_layout.addWidget(self.prompt_label)
        self.summary_layout.addWidget(self.prompt_input)
        self.summary_layout.addWidget(self.summarize_button)
        self.summary_layout.addStretch(1)

    def init_common_elements(self):
        # Console display
        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #f0f0f0;")

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

        # Status label
        self.status_label = QLabel(
            'Select an audio or transcription file', self)
        self.status_label.setAlignment(Qt.AlignCenter)

    def log_to_console(self, message):
        self.console.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        QCoreApplication.processEvents()

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
            self.transcription_path = None
            self.transcribed_text = ""
            self.log_to_console(f"Selected audio file: {file_path}")
            self.status_label.setText(
                f"File selected: {os.path.basename(file_path)}")
            self.audio_file_label.setText(
                f"Audio File: {os.path.basename(file_path)}")
            self.transcription_file_label.setText("Transcription File: None")
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
            self.audio_path = None
            self.audio_file_label.setText("Audio File: None")

            self.transcription_path = file_path
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.transcribed_text = file.read()
                self.log_to_console(f"Transcription loaded from: {file_path}")
                self.status_label.setText(
                    f"Transcription loaded from: {os.path.basename(file_path)}")
                self.transcription_file_label.setText(
                    f"Transcription File: {os.path.basename(file_path)}")
                self.transcription_file_label_summary.setText(
                    f"Transcription File: {os.path.basename(file_path)}")
                self.convert_button.setEnabled(False)
                self.summarize_button.setEnabled(True)
                self.estimate_summary_cost()
            except Exception as e:
                self.handle_error(f"Error loading transcription: {e}")

    def estimate_cost(self):
        if self.audio_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setValue(0)
            self.log_to_console("Loading audio file...")
            self.status_label.setText('Loading audio file...')
            QCoreApplication.processEvents()

            # Submit to the thread pool
            future = self.executor.submit(
                load_audio_duration,
                self.audio_path,
                self.loading_progress.emit,
                self.error_signal.emit
            )

            # Handle the result when it's available
            future.add_done_callback(self.handle_audio_loaded)

        else:
            self.estimated_cost = None
            self.audio_length_seconds = None

    def update_loading_progress(self, progress):
        self.progress_bar.setValue(progress)
        QCoreApplication.processEvents()

    def handle_audio_loaded(self, future):
        """Called when load_audio_duration is complete."""
        try:
            duration = future.result()  # Get result or exception
            if duration is None:
                return  # Error already handled

            self.audio_length_seconds = duration
            duration_minutes = self.audio_length_seconds / 60
            self.estimated_cost = 0.006 * math.ceil(duration_minutes)

            hours = int(self.audio_length_seconds // 3600)
            minutes = int((self.audio_length_seconds % 3600) // 60)
            seconds = int(self.audio_length_seconds % 60)
            formatted_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            self.length_label.setText(f"Audio Length: {formatted_duration}")
            self.estimate_label.setText(
                f"Estimated Cost: ${self.estimated_cost:.4f}")
            self.progress_bar.setVisible(False)
            self.status_label.setText(
                f"File selected: {os.path.basename(self.audio_path)}")
            self.log_to_console("Audio file loaded.")
            QCoreApplication.processEvents()

        except Exception as e:
            self.error_signal.emit(str(e))

    def start_transcription(self):
        if not self.audio_path:
            return

        self.log_to_console("Starting conversion...")
        self.status_label.setText('Converting audio to text...')
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.convert_button.setEnabled(False)
        self.summarize_button.setEnabled(False)
        QCoreApplication.processEvents()

        # Convert audio to optimal format
        converted_audio_path = self.convert_audio_format(self.audio_path)
        if converted_audio_path:
            self.log_to_console(
                f"Using converted audio file: {converted_audio_path}")
            self.audio_path = converted_audio_path
        else:
            self.handle_error("Audio conversion failed.")
            return

        # Transcribe in the background
        self.executor.submit(self.transcribe_in_chunks, self.audio_path)

    def convert_audio_format(self, input_path):
        """Converts audio to MP3 using FFmpeg, optimized for transcription."""
        output_path = os.path.splitext(input_path)[
            0] + "_converted.mp3"  # Create a new name
        try:
            # Example FFmpeg command: Adjust parameters as needed.
            command = [
                'ffmpeg',
                '-y',  # Add -y to automatically overwrite existing files
                '-i', input_path,
                '-vn',  # Disable video
                '-acodec', 'libmp3lame',  # Use MP3 encoder
                '-ac', '1',  # Mono audio
                '-ar', '16000',  # 16kHz sample rate
                '-ab', '128k',  # Audio bitrate
                output_path
            ]

            subprocess.run(command, check=True,
                           capture_output=True)  # Raise exception on error
            return output_path
        except subprocess.CalledProcessError as e:
            self.log_to_console(f"FFmpeg error: {e.stderr.decode()}")
            return None
        except FileNotFoundError:
            self.error_signal.emit(
                "FFmpeg not found. Please ensure it is installed and in your system's PATH.")
            return None

    def transcribe_in_chunks(self, audio_path):
        """Transcribes the audio file in chunks, sequentially."""
        chunk_size = self.chunk_size
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
        except Exception as e:
            self.error_signal.emit(f"Error reading audio file: {e}")
            return

        total_size = len(audio_data)
        num_chunks = math.ceil(total_size / chunk_size)
        transcribed_text = ""

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            audio_chunk = audio_data[start:end]

            try:
                text = transcribe_audio(self.api_key, audio_path, audio_chunk)
                transcribed_text += text
                progress_percentage = int(((i + 1) / num_chunks) * 100)
                QCoreApplication.processEvents()  # Update GUI
                self.loading_progress.emit(progress_percentage)
                self.log_to_console(f"Completed chunk {i+1}/{num_chunks}")

                # Estimate summary cost after each chunk
                self.transcribed_text = transcribed_text
                self.estimate_summary_cost()

            except Exception as e:
                self.error_signal.emit(f"Transcription error: {e}")
                return

        # All chunks processed
        self.transcription_complete.emit(transcribed_text)  # Pass text

    def handle_transcription_complete(self, transcribed_text):
        """Handles transcription completion."""
        file_path_without_ext = os.path.splitext(self.audio_path)[0]
        self.transcription_path = f"{file_path_without_ext}_transcription.txt"

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
        self.summarize_button.setEnabled(True)
        QCoreApplication.processEvents()

        self.transcribed_text = transcribed_text  # Store the transcribed text
        self.estimate_summary_cost()  # Estimate cost after transcription

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

        # Start summarization in background
        future = self.executor.submit(
            summarize_text,
            self.api_key,
            self.transcribed_text,
            prompt,
            self.selected_model
        )
        future.add_done_callback(self.handle_summary_task_complete)

    def handle_summary_task_complete(self, future):
        try:
            summary_text, input_tokens, output_tokens = future.result()
            self.summary_complete.emit(summary_text, input_tokens,
                                       output_tokens)
        except Exception as e:
            self.error_signal.emit(f"Summary error: {e}")

    def handle_summary_complete(self, summary_text, input_tokens, output_tokens):
        """Handles summary completion."""
        file_path_without_ext = os.path.splitext(self.audio_path)[0]
        summary_path = f"{file_path_without_ext}_AI_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as text_file:
            text_file.write(summary_text)

        self.log_to_console(f"Summary saved to: {summary_path}")
        self.status_label.setText('Summary completed successfully!')
        self.progress_bar.setVisible(False)
        QCoreApplication.processEvents()

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
        QCoreApplication.processEvents()

    def update_token_usage_label(self):
        self.token_usage_label.setText(
            f"Input Tokens: {self.summary_input_tokens}, Output Tokens: {self.summary_output_tokens}")
        QCoreApplication.processEvents()

    def handle_error(self, error_message):
        self.log_to_console(f"Error: {error_message}")
        self.status_label.setText(f'Error: {error_message}')
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        self.summarize_button.setEnabled(True)
        QCoreApplication.processEvents()

    def closeEvent(self, event):
        # Shutdown the thread pool
        self.executor.shutdown()
        event.accept()
        QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = AudioTranscriber()
    w.show()
    app.exec_()
