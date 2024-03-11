from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QPlainTextEdit, QLineEdit, QMenuBar, QDialog, QPushButton, 
                               QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox)
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Signal
from command_handler import CommandHandler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from settings import Settings
settings = Settings()
import os

class Terminal(QPlainTextEdit):
    def __init__(self, parent=None):
        super(Terminal, self).__init__(parent)
        self.setReadOnly(True)

class Prompt(QLineEdit):
    def __init__(self, parent=None):
        super(Prompt, self).__init__(parent)
        self.setPlaceholderText("lst cmd")

class HamburgerMenu(QMenuBar):
    openSettingsDialog = Signal()

    def __init__(self, parent=None):
        super(HamburgerMenu, self).__init__(parent)
        self.createMenuItems()

    def createMenuItems(self):
        menu = self.addMenu("☰")
        settingsAction = QAction('Settings', self)
        settingsAction.triggered.connect(self.onSettingsTriggered)
        menu.addAction(settingsAction)

    def onSettingsTriggered(self):
        self.openSettingsDialog.emit()

class SettingsDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.main_window = main_window
        self.setWindowTitle('Settings')
        self.layout = QVBoxLayout(self)
        self.formLayout = QFormLayout()
        self.init_widgets()
        self.saveButton = QPushButton('Save')
        self.saveButton.clicked.connect(self.save_settings)
        self.layout.addLayout(self.formLayout)
        self.layout.addWidget(self.saveButton)
        self.settings = Settings()

    def init_widgets(self):
        # Default Days
        self.default_days_widget = QSpinBox()
        self.default_days_widget.setValue(settings.default_days)
        self.formLayout.addRow("Default Days:", self.default_days_widget)

        # Watching Ticker
        self.watching_ticker_widget = QLineEdit()
        self.watching_ticker_widget.setText(settings.watching_ticker)
        self.formLayout.addRow("Watching Ticker:", self.watching_ticker_widget)

        # Test Size
        self.test_size_widget = QDoubleSpinBox()
        self.test_size_widget.setSingleStep(0.01)
        self.test_size_widget.setValue(settings.test_size)
        self.formLayout.addRow("Test Size:", self.test_size_widget)

        # Random State
        self.random_state_widget = QSpinBox()
        self.random_state_widget.setValue(settings.random_state)
        self.formLayout.addRow("Random State:", self.random_state_widget)

        # Poly Degree
        self.poly_degree_widget = QSpinBox()
        self.poly_degree_widget.setValue(settings.poly_degree)
        self.formLayout.addRow("Poly Degree:", self.poly_degree_widget)

        # SVR Kernel
        self.svr_kernel_widget = QComboBox()
        self.svr_kernel_widget.addItems(['linear', 'poly', 'rbf', 'sigmoid'])
        self.svr_kernel_widget.setCurrentText(settings.svr_kernel)
        self.formLayout.addRow("SVR Kernel:", self.svr_kernel_widget)

        # SVR C
        self.svr_c_widget = QDoubleSpinBox()
        self.svr_c_widget.setMaximum(1000)
        self.svr_c_widget.setValue(settings.svr_c)
        self.formLayout.addRow("SVR C:", self.svr_c_widget)

        # SVR Gamma
        self.svr_gamma_widget = QLineEdit()
        self.svr_gamma_widget.setText(settings.svr_gamma)
        self.formLayout.addRow("SVR Gamma:", self.svr_gamma_widget)

        # RFR N Estimators
        self.rfr_n_estimators_widget = QSpinBox()
        self.rfr_n_estimators_widget.setMaximum(1000)
        self.rfr_n_estimators_widget.setValue(int(settings.rfr_n_estimators))
        self.formLayout.addRow("RFR N Estimators:", self.rfr_n_estimators_widget)

        # ARIMA Steps
        self.arima_steps_widget = QSpinBox()
        self.arima_steps_widget.setValue(settings.arima_steps)
        self.formLayout.addRow("ARIMA Steps:", self.arima_steps_widget)

        # ARIMA Order
        self.arima_order_widget = QLineEdit()
        self.arima_order_widget.setText(str(settings.arima_order))
        self.formLayout.addRow("ARIMA Order:", self.arima_order_widget)

        # LSTM Units
        self.lstm_units_widget = QSpinBox()
        self.lstm_units_widget.setMaximum(1000)
        self.lstm_units_widget.setValue(settings.lstm_units)
        self.formLayout.addRow("LSTM Units:", self.lstm_units_widget)

        # LSTM Activation
        self.lstm_activation_widget = QComboBox()
        self.lstm_activation_widget.addItems(['relu', 'sigmoid', 'tanh', 'linear'])
        self.lstm_activation_widget.setCurrentText(settings.lstm_activation)
        self.formLayout.addRow("LSTM Activation:", self.lstm_activation_widget)

        # LSTM Dense
        self.lstm_dense = QSpinBox()
        self.lstm_dense.setValue(settings.lstm_dense)
        self.formLayout.addRow("LSTM Dense:", self.lstm_dense)

        # LSTM Optimizer
        self.lstm_optimizer = QComboBox()
        self.lstm_optimizer.addItems(['adam', 'sgd', 'rmsprop', 'adamax'])
        self.lstm_optimizer.setCurrentText(settings.lstm_optimizer)
        self.formLayout.addRow("LSTM Optimizer:", self.lstm_optimizer)

        # LSTM Loss
        self.lstm_loss = QComboBox()
        self.lstm_loss.addItems(['mse', 'mae', 'logcosh', 'cosine_similarity'])
        self.lstm_loss.setCurrentText(settings.lstm_loss)
        self.formLayout.addRow("LSTM Loss:", self.lstm_loss)

        # LSTM Epochs
        self.lstm_epochs = QSpinBox()
        self.lstm_epochs.setMaximum(1000)  # Adjust maximum as needed
        self.lstm_epochs.setValue(settings.lstm_epochs)
        self.formLayout.addRow("LSTM Epochs:", self.lstm_epochs)

        # LSTM Batch Size
        self.lstm_batch_size = QSpinBox()
        self.lstm_batch_size.setMaximum(1000)  # Adjust maximum as needed
        self.lstm_batch_size.setValue(settings.lstm_batch_size)
        self.formLayout.addRow("LSTM Batch Size:", self.lstm_batch_size)

        # LSTM Input Shape
        self.lstm_input_shape = QLineEdit()
        self.lstm_input_shape.setText(str(settings.lstm_input_shape))
        self.formLayout.addRow("LSTM Input Shape:", self.lstm_input_shape)
        
    def save_settings(self):
        settings.default_days = self.default_days_widget.value()
        settings.watching_ticker = self.watching_ticker_widget.text()
        settings.test_size = self.test_size_widget.value()
        settings.random_state = self.random_state_widget.value()
        settings.poly_degree = self.poly_degree_widget.value()
        settings.svr_kernel = self.svr_kernel_widget.currentText()
        settings.svr_c = self.svr_c_widget.value()
        settings.svr_gamma = self.svr_gamma_widget.text()
        settings.rfr_n_estimators = self.rfr_n_estimators_widget.value()
        settings.arima_steps = self.arima_steps_widget.value()
        settings.arima_order = eval(self.arima_order_widget.text())
        settings.lstm_units = self.lstm_units_widget.value()
        settings.lstm_activation = self.lstm_activation_widget.currentText()
        settings.lstm_dense = self.lstm_dense.value()
        settings.lstm_optimizer = self.lstm_optimizer.currentText()
        settings.lstm_loss = self.lstm_loss.currentText()
        settings.lstm_epochs = self.lstm_epochs.value()
        settings.lstm_batch_size = self.lstm_batch_size.value()
        settings.lstm_input_shape = eval(self.lstm_input_shape.text())

        settings.save_settings()
        self.main_window.notify_terminal("Settings Saved!")
        self.accept()

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
        ax.plot(dates, prices, '-o', markersize=4, label='Historical Prices', color='blue')
        ax.set_title('Historical Price Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(loc='best')
        self.figure.autofmt_xdate()
        self.canvas.draw()

class MainWindow(QMainWindow):
    update_signal = Signal(str)

    def __init__(self, api_key):
        super().__init__()        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, '..', 'resources', 'window-icon.png')
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("Market Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.command_handler = CommandHandler(main_window=self, api_key=api_key)
        self.initUI()
        self.applyStyles()
        self.update_signal.connect(self.update_terminal)
        self.hamburgerMenu.openSettingsDialog.connect(self.showSettingsDialog)
        

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
        self.terminal.appendPlainText(f"> Welcome to the Market Analyzer tool. Plug in 'lst cmd' for a list of all commands. Exercise caution when overriding default settings for hyperparameter tuning, this could have adverse effects on the predictions and model capabilities. Keep in mind the models predict based on patterns and do not consider the Random walk hypothesis.")

        # Initialize the prompt
        self.prompt = Prompt()
        self.prompt.returnPressed.connect(self.process_command)
        layout.addWidget(self.prompt)

        # Initialize the hamburger menu
        self.hamburgerMenu = HamburgerMenu()
        self.setMenuBar(self.hamburgerMenu)

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
            QMenuBar {
                background-color: #1a1a1a;
                color: #fff;
            }
            QMenuBar::item {
                background-color: #333;
                color: #fff;
            }
            QMenuBar::item:selected {
                background-color: #555;
            }
            QMenu {
                background-color: #333;
                color: #fff;
                border: 1px solid #666;
            }
            QMenu::item:selected {
                background-color: #555;
            }         
        """)

    # Thread signal related stuff?
    def process_command(self):
        command = self.prompt.text()
        if command:
            self.terminal.appendPlainText(f"> {command}\nProcessing...")
            self.prompt.clear()
            # Trigger asynchronous handling in the CommandHandler.
            self.command_handler.handle_command(command)

    def update_terminal(self, result):
        self.terminal.appendPlainText(f"> {result}")
        self.prompt.clear()

    def show_historical_data(self, dates, prices):
        self.graph.plot(dates, prices)
        self.graph.show()

    def showSettingsDialog(self):
        dialog = SettingsDialog(self)
        dialog.exec_()

    def notify_terminal(self, message):
        self.update_signal.emit(message)