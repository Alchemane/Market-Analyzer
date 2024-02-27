import requests
from alpha_vantage import AlphaVantage
from machine_learning import (LinearRegressor, SVRRegressor, PolynomialRegressor, 
                              DecisionTreeRegression, RandomForestRegression)
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
        }
        self.models = {
            "lr": LinearRegressor(),
            "svr": SVRRegressor(),
            "polyr": PolynomialRegressor(),
            "dtr": DecisionTreeRegression(),
            "rfr": RandomForestRegression(),
            #: MultipleLinearRegressor(),
        }
        for model_key in self.models.keys():
            self.command_map[f"fit {model_key}"] = lambda symbol, days=None, mk=model_key: self.fit_model(mk, symbol, days)
            self.command_map[f"pred {model_key}"] = lambda symbol, mk=model_key: self.predict_model(mk, symbol)
        
    def handle_command(self, command):
        command = command.strip()
        parts = command.split()
        cmd_key = parts[0]
        if cmd_key not in self.command_map:
            return "Unknown command"
        args = parts[1:]
        if cmd_key.startswith("fit") or cmd_key.startswith("pred"):
            if len(args) < 1: # Symbol provided?
                return "Error: Symbol is required."

            symbol = args[0]
            days = int(args[1]) if len(args) > 1 and cmd_key.startswith("fit") else None
            model_key = cmd_key.split()[1]
            if model_key not in self.models:
                return f"Error: Unknown model '{model_key}'."

            if cmd_key.startswith("fit"):
                return self.fit_model(model_key, symbol, days)
            elif cmd_key.startswith("pred"):
                return self.predict_model(model_key, symbol)
        else:
            try:
                return self.command_map[cmd_key](*args)
            except ValueError as e:
                return f"Error: {str(e)}"

    def get_list_cmd(self):
        commands = [
            "lst cmd - Lists all available commands.",
            "get price {symbol} - Returns the price of the specified symbol.",
            "show hist {symbol} - Displays historical price data for the specified symbol.",
            "fit lr {symbol} {days} - Fits the linear regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "pred lr {symbol} - Predicts the next price for the specified symbol using the fitted linear regression model. Requires the model to be fitted first.",
            "fit svr {symbol} {days} - Fits the support vector regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "pred svr {symbol} - Predicts the next price for the specified symbol using the fitted support vector regression model. Requires the model to be fitted first.",
            "fit polyr {symbol} {days} - Fits the polynomial regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "pred polyr {symbol} - Predicts the next price for the specified symbol using the fitted polynomial regression model. Requires the model to be fitted first.",
            "fit dtr {symbol} {days} - Fits the decision tree regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "pred dtr {symbol} - Predicts the next price for the specified symbol using the fitted decision tree regression model. Requires the model to be fitted first.",
            "fit rfr {symbol} {days} - Fits the random forest regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "pred rfr {symbol} - Predicts the next price for the specified symbol using the fitted random forest regression model. Requires the model to be fitted first.",
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
            return "Error: Unknown model"
        historical_data = self.av.fetch_historical_data(symbol=symbol)
        if days is not None:
            days = int(days)
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_historical_data = {date: value for date, value in historical_data.items() if datetime.strptime(date, '%Y-%m-%d') >= cutoff_date}
        else:
            filtered_historical_data = historical_data
        dates = list(filtered_historical_data.keys())
        prices = [value["4. close"] for value in filtered_historical_data.values()]
        dates, prices = zip(*sorted(zip(dates, prices), key=lambda x: x[0]))
        day_indices = list(range(len(dates)))
        model = self.models[model_key]
        model.fit(day_indices, list(prices))
        return f"Fitted {symbol} to the {model_key.upper()} model using the last {(days + 'days') if days else 'entire'} period's data..."
        
    def predict_model(self, model_key, symbol):
        if model_key not in self.models:
            return "Error: Unknown model"
        model = self.models[model_key]
        next_day_index = [[model.last_day_index + 1]]
        predicted_price = model.predict([[next_day_index]])
        return f"Predicted price for {symbol} tomorrow using {model_key.upper()} model is {predicted_price[0]}"