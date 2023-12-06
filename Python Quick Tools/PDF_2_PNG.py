import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from pdf2image import convert_from_path


class PDFConverterApp(QWidget):
    """
    A simple PDF to PNG converter application with a PyQt5 GUI.

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
        Button to start the conversion process.
    """

    def __init__(self):
        """
        Initialize the application and set up the UI.
        """
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Set up the user interface of the application.
        """
        # Create layout
        layout = QVBoxLayout()

        # File selection layout
        file_selection_layout = QHBoxLayout()
        self.file_label = QLabel('PDF File:')
        self.file_line_edit = QLineEdit()
        self.browse_button = QPushButton('Browse')
        file_selection_layout.addWidget(self.file_label)
        file_selection_layout.addWidget(self.file_line_edit)
        file_selection_layout.addWidget(self.browse_button)
        self.browse_button.clicked.connect(self.get_pdf_file)

        # DPI selection layout
        dpi_layout = QHBoxLayout()
        self.dpi_label = QLabel('DPI:')
        self.dpi_line_edit = QLineEdit('500')  # Set default DPI
        dpi_layout.addWidget(self.dpi_label)
        dpi_layout.addWidget(self.dpi_line_edit)

        # Convert button
        self.convert_button = QPushButton('Convert')
        self.convert_button.clicked.connect(self.convert_pdf_to_png)

        # Add widgets to the main layout
        layout.addLayout(file_selection_layout)
        layout.addLayout(dpi_layout)
        layout.addWidget(self.convert_button)

        # Set the window layout and properties
        self.setLayout(layout)
        self.setWindowTitle('PDF to PNG Converter')
        self.setGeometry(300, 300, 800, 150)
        self.show()

    def get_pdf_file(self):
        """
        Open a file dialog to select a PDF file and set the file path
        in the text input.
        """
        pdf_file, _ = QFileDialog.getOpenFileName(
            self, 'Select PDF file', '', 'PDF files (*.pdf)')
        if pdf_file:
            self.file_line_edit.setText(pdf_file)

    def convert_pdf_to_png(self):
        """
        Convert the specified PDF file to PNG images with the provided DPI.
        """
        # Get the PDF file path and DPI from the input fields
        pdf_file_path = self.file_line_edit.text()
        if not os.path.isfile(pdf_file_path):
            print("The specified PDF file does not exist or is not a file.")
            return

        try:
            dpi = int(self.dpi_line_edit.text())
        except ValueError:
            print("Invalid DPI Input. Using the default value 500 DPI.")
            dpi = 500

        # Get the directory and base name of the PDF file
        pdf_directory = os.path.dirname(pdf_file_path)
        pdf_basename = os.path.splitext(os.path.basename(pdf_file_path))[0]

        # Convert the PDF to PNG
        images = convert_from_path(pdf_file_path, dpi=dpi)
        for i, image in enumerate(images):
            image_filename = os.path.join(
                pdf_directory, f'{pdf_basename}_pg{i+1}.png')
            image.save(image_filename, 'PNG')
            print(f'Saved: {image_filename}')


# Run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = PDFConverterApp()
    sys.exit(app.exec_())
