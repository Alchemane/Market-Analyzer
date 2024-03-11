import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
from settings import Settings

def main():
    settings = Settings()
    app = QApplication(sys.argv)
    api_key = "ZXU8ZDWQW76T5XUA"
    window = MainWindow(api_key=api_key)
    window.show()
    settings.load_settings()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()