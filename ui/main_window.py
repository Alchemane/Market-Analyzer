from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit, QLineEdit
from ui.styles import AppStyles
import sys

class Terminal(QPlainTextEdit):
    def __init__(self, parent=None):
        super(Terminal, self).__init__(parent)
        self.setReadOnly(True)

class Prompt(QLineEdit):
    def __init__(self, parent=None):
        super(Prompt, self).__init__(parent)
        self.setPlaceholderText("Interact with the Analyzer...")

class CommandHandler:
    def __init__(self):
        pass

    def handler(self, command):
        # Dispatch the command to the appropriate method
        if command.startswith("get price"):
            return self.get_price(command.split()[2])

    def get_price(self, item):
        return item

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.apply_styles()
        self.setWindowTitle("Market Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.command_handler = CommandHandler()
        self.initUI()

    def apply_styles(self):
        self.terminal.setStyleSheet(AppStyles.get_terminal_style())
        self.prompt.setStyleSheet(AppStyles.get_prompt_style())
        self.setStyleSheet(AppStyles.get_main_window_style())

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