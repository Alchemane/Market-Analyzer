from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit
import sys
from app_logic import CommandHandler

class Terminal(QPlainTextEdit):
    def __init__(self, parent=None):
        super(Terminal, self).__init__(parent)
        self.setReadOnly(True)

class Prompt(QLineEdit):
    def __init__(self, parent=None):
        super(Prompt, self).__init__(parent)
        self.setPlaceholderText("Interact with the Analyzer...")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Market Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.command_handler = CommandHandler()
        self.initUI()
        self.applyStyles()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Initialize the terminal
        self.terminal = Terminal()
        layout.addWidget(self.terminal)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())