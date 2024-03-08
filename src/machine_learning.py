from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from data_processor import DataProcessor
from datetime import timedelta
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from keras import Sequential
from keras.layers import LSTM, Dense
import numpy as np, pandas as pd 

class DataPreprocessor:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.scaler = StandardScaler()
        self.start_date = None
        self.last_date = None

    def prepare_data(self, historical_data, days=None):
        dates, prices = self.data_processor.process_historical_data(historical_data)
        self.start_date = datetime.strptime(dates[0], '%Y-%m-%d')
        self.last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
        if days is not None:
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_data = {date: value for date, value in historical_data.items() if datetime.strptime(date, '%Y-%m-%d') >= cutoff_date}
            dates, prices = zip(*[(date, value["4. close"]) for date, value in filtered_data.items()])

        X, y = self.convert_variables(dates, prices)
        return X, y
    
    def convert_variables(self, dates, prices):
        X = [(datetime.strptime(date, '%Y-%m-%d') - self.start_date).days for date in dates]
        # Convert X and y into the desired shape for sklearn models
        X = np.array(X).reshape(-1, 1)
        y = np.array(prices, dtype=float)
        return X, y
    
    def prepare_X_new(self):
        if self.start_date is None:
            raise ValueError("Start date not set. Please process historical data first.")
        
        next_day = self.last_date + timedelta(days=1)
        day_index_next_day = (next_day - self.start_date).days

        X_new = np.array([day_index_next_day]).reshape(-1, 1)
        return X_new
    
    def prepare_X_new_for_lstm(self, historical_data, n_steps=10):
        prices_list = [float(value['4. close']) for value in list(historical_data.values())][-n_steps:]
        X_new = np.array(prices_list).reshape(1, n_steps, 1)
        return X_new
    
    # ARIMA and LSTM special cases in data preprocessing
    def prepare_data_for_arima(self, historical_data, days=None):
        dates, prices = self.data_processor.process_historical_data(historical_data)
        if days is not None:
            pass
        prices_series = pd.Series(prices, index=pd.to_datetime(dates))
        return prices_series
    
    def prepare_data_for_lstm(self, historical_data, days=None, n_steps=10):
        _, prices = self.data_processor.process_historical_data(historical_data)
        prices = self.normalize(prices)
        # Reshape data for LSTM
        X, y = [], []
        for i in range(len(prices) - n_steps):
            X.append(prices[i:i+n_steps])
            y.append(prices[i + n_steps])
        X, y = np.array(X), np.array(y)
        # Reshape X to [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y
    
    def normalize(self, data):
        print("Original data:", data)  # Debug print to inspect the original data
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Check if data is empty after reshaping
        if data.size == 0:
            print("Error: Data array is empty after reshaping.")
            return data  # Or handle the empty case as appropriate
        
        normalized_data = self.scaler.fit_transform(data)
        print("Normalized data shape:", normalized_data.shape)  # Debug print to inspect the shape after normalization
        
        if normalized_data.shape[1] == 1:
            normalized_data = normalized_data.ravel()
            print(normalized_data)
        return normalized_data
    
class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train_and_split(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        return X_test, y_test

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

class LinearRegressor:
    def __init__(self, **kwargs):
        self.regressor = LinearRegression(**kwargs)
        self.is_fitted = False
        self.last_day_index = None

    def fit(self, X, y):
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        self.regressor.fit(X, y)
        self.is_fitted = True
        self.last_day_index = X.shape[0] - 1

    def predict(self, X):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        X = np.array(X).reshape(-1, 1)
        return self.regressor.predict(X)
    
class PolynomialRegressor:
    def __init__(self, degree=2, **kwargs):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.regressor = LinearRegression(**kwargs)
        self.is_fitted = False
        self.last_day_index = None

    def fit(self, X, y):
        X_poly = self.poly_features.fit_transform(np.array(X).reshape(-1, 1))
        y = np.array(y).reshape(-1, 1)
        self.regressor.fit(X_poly, y)
        self.is_fitted = True
        self.last_day_index = X.shape[0] - 1

    def predict(self, X):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        X_poly = self.poly_features.transform(np.array(X).reshape(-1, 1))
        return self.regressor.predict(X_poly)

class SVRRegressor:
    def __init__(self, kernel='rbf', C=100, gamma='auto'):
        self.regressor = SVR(kernel=kernel, C=C, gamma=gamma)
        self.is_fitted = False
        self.last_day_index = None

    def fit(self, X, y):
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        self.regressor.fit(X, y)
        self.is_fitted = True
        self.last_day_index = X.shape[0] - 1

    def predict(self, X):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        X = np.array(X).reshape(-1, 1)
        return self.regressor.predict(X)
    
class DecisionTreeRegression:
    def __init__(self, random_state=42):
        self.regressor = DecisionTreeRegressor(random_state=random_state)
        self.is_fitted = False
        self.last_day_index = None

    def fit(self, X, y):
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        self.regressor.fit(X, y)
        self.is_fitted = True
        self.last_day_index = X.shape[0] - 1

    def predict(self, X):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        X = np.array(X).reshape(-1, 1)
        return self.regressor.predict(X)

class RandomForestRegression:
    def __init__(self, n_estimators=100, random_state=42):
        self.regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.is_fitted = False
        self.last_day_index = None

    def fit(self, X, y):
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        self.regressor.fit(X, y)
        self.is_fitted = True
        self.last_day_index = X.shape[0] - 1

    def predict(self, X):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        X = np.array(X).reshape(-1, 1)
        return self.regressor.predict(X)

class MultipleLinearRegressor:
    def __init__(self):
        # Placeholder class for MLR which is not integrated yet
        self.regressor = LinearRegression()
        self.is_fitted = False
        self.last_day_index = None

    def fit(self, X, y):
        # Assuming X is already a 2D array for Multiple Linear Regression
        X = np.array(X)
        y = np.array(y)
        self.regressor.fit(X, y)
        self.is_fitted = True
        self.last_day_index = X.shape[0] - 1

    def predict(self, X):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        X = np.array(X)
        return self.regressor.predict(X)
    
class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.is_fitted = False

    def fit(self, y):
        self.model = ARIMA(y, order=self.order).fit()
        self.is_fitted = True

    def predict(self, steps=1):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        forecast = self.model.forecast(steps=steps)
        return forecast
    
class LSTMModel:
    def __init__(self):
        self.model = None
        self.is_fitted = False

    def initialize_model(self, input_shape):
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs=20, batch_size=32):
        if not self.model:  # Model not yet initialized
            input_shape = (X.shape[1], X.shape[2])
            self.initialize_model(input_shape)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        return self.model.predict(X)
    
