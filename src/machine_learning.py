from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np

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
        self.last_day_index = len(X)

    def predict(self):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        # Predict the next day's price based on the last day index used during fitting
        next_day_index = np.array([[self.last_day_index]]).reshape(-1, 1)
        predicted_price = self.regressor.predict(next_day_index)
        return predicted_price
    
    def predict_future_prices(self, n_days=30):
        if not self.is_fitted:
            raise Exception("Model not fitted. Please fit the model with historical data first.")
        
        future_indices = np.array([self.last_day_index + i for i in range(1, n_days + 1)]).reshape(-1, 1)
        predicted_prices = self.regressor.predict(future_indices)
        return predicted_prices

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
        self.last_day_index = len(X)

    def predict(self):
        if not self.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        # Predict the next day's price based on the last day index used during fitting
        next_day_index = np.array([[self.last_day_index]]).reshape(-1, 1)
        predicted_price = self.regressor.predict(next_day_index)
        return predicted_price
    
    def predict_future_prices(self, n_days=30):
        if not self.is_fitted:
            raise Exception("Model not fitted. Please fit the model with historical data first.")
        
        future_indices = np.array([self.last_day_index + i for i in range(1, n_days + 1)]).reshape(-1, 1)
        predicted_prices = self.regressor.predict(future_indices)
        return predicted_prices