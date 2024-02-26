import requests
from alpha_vantage import AlphaVantage
from machine_learning import (LinearRegressor, SVRRegressor, PolynomialRegressor, 
                              DecisionTreeRegressor, RandomForestRegressor)
from data_processor import DataProcessor
from datetime import datetime, timedelta

class CommandHandler:
    def __init__(self, main_window=None, api_key=""):
        self.main_window = main_window
        self.av = AlphaVantage(api_key=api_key)
        self.data_processor = DataProcessor()
        self.command_map = {
            "get price": self.get_price,
            "lst cmd": self.get_list_cmd,
            "show hist": self.show_historical_data,
            "fit lr": self.fit_linear_regression,
            "pred lr": self.predict_linear_regression,
            "fit svr": self.fit_svr,
            "pred svr": self.predict_svr,
            "fit poly": self.fit_polynomial_regression,
            "pred poly": self.predict_polynomial_regression,
            "fit dt": self.fit_decision_tree_regression,
            "pred dt": self.predict_decision_tree_regression,
            "fit rf": self.fit_random_forest_regression,
            "pred rf": self.predict_random_forest_regression,
        }
        self.models = {
            "lr": LinearRegressor(),
            "svr": SVRRegressor(),
            "poly": PolynomialRegressor(),
            "dt": DecisionTreeRegressor(),
            "rf": RandomForestRegressor(),
            #: MultipleLinearRegressor(),
        }
        for model_key in self.models.keys():
            self.command_map[f"fit {model_key}"] = self.fit_model
            self.command_map[f"pred {model_key}"] = self.predict_model
        
    def handle_command(self, command):
        command = command.strip()
        parts = command.split()
        cmd_key = parts[0]
        if cmd_key not in self.command_map:
            return "Unknown command"

        args = parts[1:]
        try:
            if cmd_key in ["fit lr", "fit svr", "pred lr", "pred svr"] and len(args) < 1:
                raise ValueError("Symbol is required.")
            symbol = args[0] if len(args) > 0 else None
            days = int(args[1]) if len(args) > 1 else None
            return self.command_map[cmd_key](symbol, days)
        except ValueError as e:
            return f"Error: {str(e)}"

    def get_list_cmd(self):
        commands = [
            "get price {symbol} - Returns the price of the specified symbol.",
            "lst cmd - Lists all available commands.",
            "show hist {symbol} - Displays historical price data for the specified symbol.",
            "fit lr {symbol} {days} - Fits the linear regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "pred lr {symbol} - Predicts the next price for the specified symbol using the fitted linear regression model. Requires the model to be fitted first.",
            "fit svr {symbol} {days} - Fits the SVR model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "pred svr {symbol} - Predicts the next price for the specified symbol using the fitted SVR model. Requires the model to be fitted first.",
            "fit poly {symbol} {days} - Fits the polynomial regression model.",
            "pred poly {symbol} - Predicts the next price using the polynomial regression model.",
            "fit dt {symbol} {days} - Fits the decision tree regression model.",
        ]
        return "\n".join(commands)
    
    def show_historical_data(self, symbol):
        historical_data = self.av.fetch_historical_data(symbol)
        dates, prices = self.data_processor.process_historical_data(historical_data)

        dates.reverse()
        prices.reverse()
        self.main_window.show_historical_data(dates, prices)
        return "Displaying historical data..."
    
    def get_price(self, symbol):
        try:
            price = self.av.fetch_real_time_price(symbol=symbol)
            if price is not None:
                return f"Price of {symbol} is {price}"
            else:
                return f"Failed to fetch data: '{symbol}' not recognized, data unavailable, or API limit of 25 calls has been reached."
        except requests.exceptions.HTTPError as e:
            return f"Failed to fetch data: Could not connect to the API. Error: {e}"
        
    def fit_model(self, model_key, symbol, days=None):
        if model_key not in self.models:
            return "Unknown model"
        historical_data = self.av.fetch_historical_data(symbol=symbol)
        if days is not None:
            days = int(days)
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_historical_data = {date: value for date, value in historical_data.items() if datetime.strptime(date, '%Y-%m-%d') >= cutoff_date}
        else:
            filtered_historical_data = historical_data
        model = self.models[model_key]
        dates = list(filtered_historical_data.keys())
        prices = [value["4. close"] for value in filtered_historical_data.values()]
        dates, prices = zip(*sorted(zip(dates, prices), key=lambda x: x[0]))
        day_indices = list(range(len(dates)))
        self.model_key.fit(day_indices, list(prices))
        return f"Fitted {symbol} to the {model_key.upper()} model using the last {(days + 'days') if days else 'entire'} period's data..."
        

    def predict_linear_regression(self, symbol):
        if not self.linear_regressor.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        predicted_price = self.linear_regressor.predict()
        return f"Predicted price for {symbol} is {predicted_price[0]}"


    
    def predict_model(self, model_key, symbol):
        if model_key not in self.models:
            return "Unknown model"
        model = self.models[model_key]



        return f"Predicted price for {symbol} using {model_key.upper()} model..."