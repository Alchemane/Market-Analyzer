from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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