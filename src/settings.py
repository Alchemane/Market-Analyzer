import json

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Settings(metaclass=SingletonMeta):
    def __init__(self):
        # Default settings
        self.default_days = 30
        self.watching_ticker = 'nvda' # Default watching ticker
        # Fitting ml models
        self.test_size = 0.2
        self.random_state = 42
        # Model specific settings
        self.poly_degree = 2
        self.svr_kernel = 'rbf'
        self.svr_c = 100
        self.svr_gamma = 'auto'
        self.rfr_n_estimators = '100'
        self.arima_steps = 1
        self.arima_order = (1, 1, 1)
        self.lstm_units = 50
        self.lstm_activation = 'sigmoid'
        self.lstm_dense = 1
        self.lstm_optimizer = 'adam'
        self.lstm_loss = 'mse'
        self.lstm_epochs = 50
        self.lstm_batch_size = 32
        self.lstm_input_shape = (10, 1)

        self.settings_dict = None

    def to_dict(self):
        return {
            "default_days": self.default_days,
            "watching_ticker": self.watching_ticker,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "poly_degree": self.poly_degree,
            "svr_kernel": self.svr_kernel,
            "svr_c": self.svr_c,
            "svr_gamma": self.svr_gamma,
            "rfr_n_estimators": self.rfr_n_estimators,
            "arima_steps": self.arima_steps,
            "arima_order": self.arima_order,
            "lstm_units": self.lstm_units,
            "lstm_activation": self.lstm_activation,
            "lstm_dense": self.lstm_dense,
            "lstm_optimizer": self.lstm_optimizer,
            "lstm_loss": self.lstm_loss,
            "lstm_epochs": self.lstm_epochs,
            "lstm_batch_size": self.lstm_batch_size,
            "lstm_input_shape": self.lstm_input_shape,
        }

    def update_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a recognized setting")

    def load_settings(self):
        try:
            with open('settings.json', 'r') as file:
                self.settings_dict = json.load(file)
                for key, value in self.settings_dict.items():
                    setattr(self, key, value)
                    print("json values: ", key, value)
                print(self.settings_dict)
        except FileNotFoundError:
            print("Settings file not found. Using default settings.")

    def get_settings(self):
        return self.settings_dict

    def save_settings(self):
        self.settings_dict = self.to_dict()
        with open('settings.json', 'w') as file:
            json.dump(self.settings_dict, file, indent=4)