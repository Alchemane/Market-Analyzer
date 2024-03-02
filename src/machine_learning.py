from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_processor import DataProcessor
from datetime import timedelta
import numpy as np, datetime

class DataPreprocessor:
    def __init__(self):
        self.data_processor = DataProcessor()

    def prepare_data(self, historical_data, days=None):
        dates, prices = self.data_processor.process_historical_data(historical_data)

        if days is not None:
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_data = {date: value for date, value in historical_data.items() if datetime.strptime(date, '%Y-%m-%d') >= cutoff_date}
            dates, prices = zip(*[(date, value["4. close"]) for date, value in filtered_data.items()])

        # Convert dates and prices to the format required by your models
        X, y = self.convert_variables(dates, prices)
        return X, y
    
    def convert_variables(self, dates, prices):
        start_date = datetime.strptime(dates[0], '%Y-%m-%d')
        X = [(datetime.strptime(date, '%Y-%m-%d') - start_date).days for date in dates]
        # Convert X and y into the desired shape for sklearn models
        X = np.array(X).reshape(-1, 1)
        y = np.array(prices)
        return X, y

class ModelEvaluator:
    def __init__(self, model, X, y, test_size=0.2, random_state=42):
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def fit_and_evaluate(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        print(f"Mean Squared Error: {mse}")

    def predict_new(self, X_new):
        return self.model.predict(X_new)

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