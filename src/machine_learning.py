from sklearn.linear_model import LinearRegression
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