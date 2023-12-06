import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox
)
from pdf2image import convert_from_path


class PDFConverterApp(QWidget):
    """
    A simple PDF to image converter application with a PyQt5 GUI.

    Attributes
    ----------
    file_label : QLabel
        Label for the PDF file path input.
    file_line_edit : QLineEdit
        Text input for the PDF file path.
    browse_button : QPushButton
        Button to browse and select a PDF file.
    dpi_label : QLabel
        Label for the DPI input.
    dpi_line_edit : QLineEdit
        Text input for specifying the DPI value.
    convert_button : QPushButton
        Button to start the conversion process of a single file.
    folder_label : QLabel
        Label for the PDF folder path input.
    folder_line_edit : QLineEdit
        Text input for the PDF folder path.
    folder_browse_button : QPushButton
        Button to browse and select a folder containing PDF files.
    folder_convert_button : QPushButton
        Button to start the conversion process for all PDFs in a folder.
    format_label : QLabel
        Label for the image format selection.
    format_dropdown : QComboBox
        Dropdown menu to select the image format for conversion.
    """

    def __init__(self):
        """Initialize the application and set up the UI."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """Set up the user interface of the application."""
        # Main layout
        layout = QVBoxLayout()

        # File selection layout
        file_selection_layout = QHBoxLayout()
        self.file_label = QLabel('PDF File:')
        self.file_line_edit = QLineEdit()
        self.browse_button = QPushButton('Browse File')
        file_selection_layout.addWidget(self.file_label)
        file_selection_layout.addWidget(self.file_line_edit)
        file_selection_layout.addWidget(self.browse_button)
        self.browse_button.clicked.connect(self.get_pdf_file)

        # Convert button for single file
        self.convert_button = QPushButton('Convert File')
        self.convert_button.clicked.connect(self.convert_file)
        file_selection_layout.addWidget(self.convert_button)

        # Folder selection layout
        folder_selection_layout = QHBoxLayout()
        self.folder_label = QLabel('PDF Folder:')
        self.folder_line_edit = QLineEdit()
        self.folder_browse_button = QPushButton('Browse Folder')
        self.folder_convert_button = QPushButton('Convert Folder')
        folder_selection_layout.addWidget(self.folder_label)
        folder_selection_layout.addWidget(self.folder_line_edit)
        folder_selection_layout.addWidget(self.folder_browse_button)
        folder_selection_layout.addWidget(self.folder_convert_button)
        self.folder_browse_button.clicked.connect(self.get_pdf_folder)
        self.folder_convert_button.clicked.connect(self.convert_folder)

        # DPI and Format layout on the same line
        settings_layout = QHBoxLayout()
        # DPI layout
        dpi_layout = QHBoxLayout()
        self.dpi_label = QLabel('DPI:')
        self.dpi_line_edit = QLineEdit('500')  # Default to 500 DPI
        dpi_layout.addWidget(self.dpi_label)
        dpi_layout.addWidget(self.dpi_line_edit)
        # Add DPI layout with stretch factor 1
        settings_layout.addLayout(dpi_layout, 1)

        # Format layout
        format_layout = QHBoxLayout()
        self.format_label = QLabel('Format:')
        self.format_dropdown = QComboBox()
        self.format_dropdown.addItems(
            ['png', 'jpeg', 'jpg', 'bmp', 'tiff', 'gif'])
        self.format_dropdown.setCurrentText('png')  # Default to PNG format
        format_layout.addWidget(self.format_label)
        format_layout.addWidget(self.format_dropdown)
        # Add format layout with stretch factor 1
        settings_layout.addLayout(format_layout, 1)

        # Add layouts to the main layout
        layout.addLayout(file_selection_layout)
        layout.addLayout(folder_selection_layout)
        layout.addLayout(settings_layout)

        # Set the window layout and properties
        self.setLayout(layout)
        self.setWindowTitle('PDF to Image Converter')
        self.setGeometry(300, 300, 800, 150)
        self.show()

    def get_pdf_file(self):
        """Open a file dialog to select a PDF file and set the file path in the text input."""
        pdf_file, _ = QFileDialog.getOpenFileName(
            self, 'Select PDF file', '', 'PDF files (*.pdf)')
        if pdf_file:
            self.file_line_edit.setText(pdf_file)

    def get_pdf_folder(self):
        """Open a file dialog to select a folder and set the folder path in the text input."""
        pdf_folder = QFileDialog.getExistingDirectory(
            self, 'Select PDF Folder')
        if pdf_folder:
            self.folder_line_edit.setText(pdf_folder)

    def convert_file(self):
        """Convert the specified PDF file to selected image format with the provided DPI."""
        pdf_file_path = self.file_line_edit.text()
        self.convert_pdf(pdf_file_path)

    def convert_folder(self):
        """Convert all PDF files in the specified folder to selected image format with the provided DPI."""
        pdf_folder_path = self.folder_line_edit.text()
        if not os.path.isdir(pdf_folder_path):
            print("The specified folder does not exist or is not a directory.")
            return
        for pdf_file in os.listdir(pdf_folder_path):
            if pdf_file.lower().endswith('.pdf'):
                pdf_file_path = os.path.join(pdf_folder_path, pdf_file)
                self.convert_pdf(pdf_file_path)

    def convert_pdf(self, pdf_file_path):
        """
        Convert a PDF file to selected image format with the provided DPI.

        Parameters
        ----------
        pdf_file_path : str
            The file path to the PDF to be converted.
        """
        # Check if the file exists and is a file
        if not os.path.isfile(pdf_file_path):
            print(
                f"The specified file does not exist or is not a file: {pdf_file_path}")
            return

        # Get the DPI value
        try:
            dpi = int(self.dpi_line_edit.text())
        except ValueError:
            print("Invalid DPI Input. Using the default value 500 DPI.")
            dpi = 500

        # Get the selected image format from the dropdown
        image_format = self.format_dropdown.currentText()

        # Get the directory and base name of the PDF file
        pdf_directory = os.path.dirname(pdf_file_path)
        pdf_basename = os.path.splitext(os.path.basename(pdf_file_path))[0]

        # Convert the PDF to the selected image format
        images = convert_from_path(pdf_file_path, dpi=dpi)
        for i, image in enumerate(images):
            image_filename = os.path.join(
                pdf_directory, f'{pdf_basename}_pg{i+1}.{image_format}')
            image.save(image_filename, image_format.upper())
            print(f'Saved: {image_filename}')


# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = PDFConverterApp()
    ex.show()
    # sys.exit(app.exec_())
