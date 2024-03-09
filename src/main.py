import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
from settings import Settings
settings = Settings()
settings.load_settings()

def main():
    app = QApplication(sys.argv)
    api_key = "ZXU8ZDWQW76T5XUA"
    window = MainWindow(api_key=api_key)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()