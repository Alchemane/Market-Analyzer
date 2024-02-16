import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from src.app_logic import AppLogic

def main():
    app = QApplication(sys.argv)
    # Initialize the application logic
    app_logic = AppLogic()
    
    # Create the main window and pass the app logic to it
    window = MainWindow(app_logic)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()