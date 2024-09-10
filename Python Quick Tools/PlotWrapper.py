import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QLineEdit, QHBoxLayout, QFileDialog, QLabel
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotWrapper(QMainWindow):
    def __init__(self, ax, fig, settings, parent=None):
        super(PlotWrapper, self).__init__(parent)

        self.ax = ax
        self.fig = fig
        self.settings = settings

        # Set up the QWidget as the central widget of the window
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Set up overall window layout
        layout = QVBoxLayout(self.central_widget)

        # Add the Matplotlib navigation toolbar at the top
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Add the "Tight Layout" button to the right of the built-in toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.toolbar)
        self.tight_layout_button = QPushButton("Tight Layout")
        toolbar_layout.addWidget(self.tight_layout_button)

        # Add the toolbar layout (toolbar + tight layout button) to the main layout
        layout.addLayout(toolbar_layout)

        # Top layout for the inputs and buttons
        form_layout = QHBoxLayout()

        # Label and text input for save path
        self.save_path_input = QLineEdit(self.settings.get("save_path", ""))
        self.browse_button = QPushButton("Browse")
        form_layout.addWidget(QLabel("Save Location:"))
        form_layout.addWidget(self.save_path_input)
        # Add "Browse" button after file path input
        form_layout.addWidget(self.browse_button)

        # Label and text input for save file name
        self.save_file_name_input = QLineEdit("plot.png")
        form_layout.addWidget(QLabel("File Name:"))
        form_layout.addWidget(self.save_file_name_input)

        # Save button to the right of the file name input field
        self.save_button = QPushButton("Save Plot")
        form_layout.addWidget(self.save_button)

        # Add the input form layout to the main layout
        layout.addLayout(form_layout)

        # Canvas for the current Matplotlib plot below the input form (already added)
        layout.addWidget(self.canvas)

        # Connect signals to methods
        self.save_button.clicked.connect(self.save_plot)
        self.browse_button.clicked.connect(self.browse_folder)

        # Connect the tight layout button to the function
        self.tight_layout_button.clicked.connect(self.apply_tight_layout)

        # Dynamically resize the window based on the Matplotlib figure size
        self.adjust_window_size()

        self.setWindowTitle('Matplotlib Plot in PyQt Window')
        self.show()
        plt.tight_layout()

    def adjust_window_size(self):
        """
        Adjust the size of the PyQt window based on the Matplotlib figure size.
        """
        # Get figure size in inches
        fig_width, fig_height = self.fig.get_size_inches()

        # Convert figure size to window size in pixels
        dpi = self.fig.dpi
        window_width = int(fig_width * dpi)
        window_height = int(fig_height * dpi)

        # Set window size with a little extra room for input form and toolbar
        input_space = 50  # Additional space for inputs and the toolbar
        self.setGeometry(50, 50, window_width, window_height + input_space)

    def apply_tight_layout(self):
        """
        Apply tight layout to the current Matplotlib figure, re-adjusting it.
        """
        self.fig.tight_layout()
        self.canvas.draw_idle()  # Redraw the canvas to apply changes

    def browse_folder(self):
        """
        Open a file dialog to let the user select a folder for saving the plot.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.save_path_input.setText(folder)

    def save_plot(self):
        """
        Save the current plot to the specified location with the specified filename.
        """
        save_path = self.save_path_input.text()
        file_name = self.save_file_name_input.text()

        # Make sure the directory exists or is valid
        if not os.path.exists(save_path):
            print(f"Error: Save path '{save_path}' does not exist.")
            return

        # Ensure the file name ends with a common image format extension
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
            print(f"Error: Invalid file type for '{file_name}'.")
            return

        if not file_name.strip():
            print("Error: Please provide a valid file name.")
            return

        full_path = os.path.join(save_path, file_name)

        # Save the figure to the specified file path
        self.fig.savefig(full_path, dpi=self.settings.get("dpi", 500))
        print(f"Saved plot as '{full_path}'")


# Start the PyQt Application and show plot in a separate window with the PlotWrapper class
if __name__ == "__main__":
    # app = QApplication(sys.argv)
    plt.ioff()

    # Sample data for testing
    xs = [0, 1, 2, 3, 4, 5]
    ys = [0, 1, 4, 9, 16, 25]

    # Create a plot with Matplotlib
    fig, ax = plt.subplots(figsize=(12, 4))  # Figure size in inches
    ax.set_xlabel('Shear Rate [1/s]', fontsize=12)
    ax.set_ylabel('Viscosity [Pa*s]', fontsize=12)
    ax.plot(xs, ys)  # Plot the sample data

    settings = {
        "dpi": 500,
        "save_path": r"C:\Users\dowdt\Downloads",
    }

    # Show the PyQt window with the embedded Matplotlib plot
    plot_window = PlotWrapper(ax, fig, settings=settings)
    # sys.exit(app.exec_())
