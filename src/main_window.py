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
        self.setPlaceholderText("Interact with the Analyzer...")

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
        self.terminal.appendPlainText(f"> Welcome to the Market Analyzer tool. Input 'get list cmd' for all possible commands.")

        # Initialize the prompt
        self.prompt = Prompt()
        self.prompt.returnPressed.connect(self.process_command)
        layout.addWidget(self.prompt)

    def applyStyles(self):
        # Apply the stylesheet to the application
        self.setStyleSheet("""
            QMainWindow {
                background-color: #262626; /* Dark background for the main window */
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
            QLineEdit {
                background-color: #333333;
                color: #ffffff;
                border: 2px solid #6A6A6A;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            }
            /* Add more styles as needed */
        """)

    def process_command(self):
        command = self.prompt.text()
        if command:
            result = self.command_handler.handler(command)
            self.terminal.appendPlainText(f"> {command}\n{result}")
            self.prompt.clear()

    def show_historical_data(self, symbol):
        historical_data = self.alpha_vantage.fetch_historical_data(symbol)
        dates, prices = self.data_processor.process_historical_data(historical_data)
        self.graph.plot(dates, prices)
        self.graph.show()