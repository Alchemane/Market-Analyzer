from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit
from command_handler import CommandHandler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from data_processor import DataProcessor
from alpha_vantage import AlphaVantage

class Terminal(QPlainTextEdit):
    def __init__(self, parent=None):
        super(Terminal, self).__init__(parent)
        self.setReadOnly(True)

class Prompt(QLineEdit):
    def __init__(self, parent=None):
        super(Prompt, self).__init__(parent)
        self.setPlaceholderText("get list cmd")

class HistoricalDataPlot(QWidget):
    def __init__(self, dates, prices, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.plot(dates, prices)

    def plot(self, dates, prices):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(dates, prices, '-o', markersize=4)
        ax.set_title('Historical Price Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self, api_key):
        super().__init__()
        self.setWindowTitle("Market Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.alpha_vantage = AlphaVantage(api_key=api_key)
        self.data_processor = DataProcessor()
        self.command_handler = CommandHandler(main_window=self, api_key=api_key)
        self.initUI()
        self.applyStyles()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Initialize the graph (hidden by default)
        self.graph = HistoricalDataPlot([], [], self)  # Empty data initially
        self.graph.hide()  # Hide the graph initially
        layout.addWidget(self.graph)

        # Initialize the terminal
        self.terminal = Terminal()
        layout.addWidget(self.terminal)
        self.terminal.appendPlainText(f"> Welcome to the Market Analyzer tool. Plug in 'get list cmd' for a list of all commands.")

        # Initialize the prompt
        self.prompt = Prompt()
        self.prompt.returnPressed.connect(self.process_command)
        layout.addWidget(self.prompt)

    def applyStyles(self):
        # Apply the stylesheet to the application
        self.setStyleSheet("""
            QMainWindow {
                background-color: #262626;
            }
            QPlainTextEdit, QLineEdit {
                background-color: #131418;
                color: #dcdcdc;
                border: 2px solid #6A6A6A;
                font-family: 'Consolas', 'Courier New', monospace;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #292829;
                width: 14px;
                margin: 14px 0 14px 0;
            }

            QScrollBar::handle:vertical {
                background-color: #3a393a;
                min-height: 30px;
            }

            QScrollBar::add-line:vertical {
                border: none;
                background: none;
            }
            QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QScrollBar::add-page:vertical {
                border: none;
                background: none;
            }
            QScrollBar::sub-page:vertical {
                border: none;
                background: none;
            }
            QLineEdit {
                background-color: #333333;
                color: #ffffff;
                border: 2px solid #6A6A6A;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            }
        """)

    def process_command(self):
        command = self.prompt.text()
        if command:
            result = self.command_handler.handler(command)
            self.terminal.appendPlainText(f"> {command}\n{result}")
            self.prompt.clear()

    def show_historical_data(self, dates, prices):
        self.graph.plot(dates, prices)
        self.graph.show()