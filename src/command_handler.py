import requests
from alpha_vantage import AlphaVantage
from machine_learning import LinearRegressor, SVRRegressor
from data_processor import DataProcessor
from datetime import datetime, timedelta

class CommandHandler:
    def __init__(self, main_window=None, api_key=""):
        self.main_window = main_window
        self.av = AlphaVantage(api_key=api_key)
        self.data_processor = DataProcessor()
        self.linear_regressor = LinearRegressor()
        self.svr_regressor = SVRRegressor()
        self.command_map = {
            "get price": self.get_price,
            "get list cmd": self.get_list_cmd,
            "show history": self.show_historical_data,
            "fit linear_regression": self.fit_linear_regression,
            "predict linear_regression": self.predict_linear_regression,
            "fit svr": self.fit_svr,
            "predict svr": self.predict_svr,
            "show predictions": self.show_future_predictions
        }
        
    def handle_command(self, command):
        command = command.strip()
        parts = command.split()
        if command == "get list cmd":
            return self.get_list_cmd()
        if len(parts) > 1 and parts[0] == "show" and parts[1] == "predictions":
            if len(parts) >= 4:
                symbol = parts[2]
                model_type = parts[3]
                n_days = int(parts[4]) if len(parts) > 4 else 30
                return self.show_future_predictions(symbol, model_type, n_days)
            else:
                return "Error: Insufficient arguments for 'show predictions'. Expected format: 'show predictions {symbol} {model_type} {n_days}'."
        if parts[0] == "fit" and len(parts) >= 3:
            model_type = parts[1]
            symbol = parts[2]
            days = int(parts[3]) if len(parts) > 3 else None
            cmd_key = f"fit {model_type}"
            if cmd_key in self.command_map:
                try:
                    return self.command_map[cmd_key](symbol, days)
                except TypeError as e:
                    return f"Error in command usage: {e}"
            else:
                return "Unknown command"
        else:
            cmd_key = parts[0]
            args = parts[1:]
            if cmd_key in self.command_map:
                try:
                    return self.command_map[cmd_key](*args)
                except TypeError as e:
                    return f"Error in command usage: {e}"
            else:
                return "Unknown command"

    def get_list_cmd(self):
        commands = [
            "get price {symbol} - Returns the price of the specified symbol.",
            "get list cmd - Lists all available commands.",
            "show history {symbol} - Displays historical price data for the specified symbol.",
            "fit linear_regression {symbol} {days} - Fits the linear regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "predict linear_regression {symbol} - Predicts the next price for the specified symbol using the fitted linear regression model. Requires the model to be fitted first.",
            "fit svr {symbol} {days} - Fits the SVR model to the historical data of the specified symbol. Optionally, specify the number of recent days to use.",
            "predict svr {symbol} - Predicts the next price for the specified symbol using the fitted SVR model. Requires the model to be fitted first.",
            "show predictions {symbol} {model_type} {n_days} - Displays future price predictions for the specified symbol using the specified model ('linear' or 'svr'). Optionally, specify the number of days to predict into the future."
        ]
        return "\n".join(commands)
    
    def show_historical_data(self, symbol):
        historical_data = self.av.fetch_historical_data(symbol)
        dates, prices = self.data_processor.process_historical_data(historical_data)

        dates.reverse()
        prices.reverse()
        self.main_window.show_historical_data(dates, prices)
        return "Displaying historical data..."
    
    def show_future_predictions(self, symbol, model_type='linear', n_days=30):
        model = self.linear_regressor if model_type == 'linear' else self.svr_regressor
        if not model.is_fitted:
            return f"{model_type.upper()} model not fitted. Please fit the model with historical data first."
        
        historical_data = self.av.fetch_historical_data(symbol)
        dates, prices = self.data_processor.process_historical_data(historical_data)
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        prices = [float(price) for price in prices]

        future_dates = [dates[-1] + timedelta(days=i) for i in range(1, n_days + 1)]
        future_predictions = model.predict_future_prices(n_days)

        if len(future_predictions) != len(future_dates):
            return "Error: Mismatch in lengths of future dates and predictions."

        future_predictions = list(future_predictions)
        self.main_window.plot_future_predictions(dates, prices, future_dates, future_predictions)
        return f"Displaying future predictions for {symbol} using {model_type} model..."
    
    def get_price(self, symbol):
        try:
            price = self.av.fetch_real_time_price(symbol=symbol)
            if price is not None:
                return f"Price of {symbol} is {price}"
            else:
                return f"Failed to fetch data: '{symbol}' not recognized, data unavailable, or API limit of 25 calls has been reached."
        except requests.exceptions.HTTPError as e:
            return f"Failed to fetch data: Could not connect to the API. Error: {e}"
        
    # Simple Linear Regression
    def fit_linear_regression(self, symbol, days=None):
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
        self.linear_regressor.fit(day_indices, list(prices))
        return f"Fitted {symbol} to the Linear Regression model using the last {days if days else 'entire'} period's data..."

    def predict_linear_regression(self, symbol):
        if not self.linear_regressor.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        predicted_price = self.linear_regressor.predict()
        return f"Predicted price for {symbol} is {predicted_price[0]}"
    
    # Support Vector Regression
    def fit_svr(self, symbol, days=None):
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
        self.svr_regressor.fit(day_indices, list(prices))
        return f"Fitted {symbol} to the SVR model using the last {days if days else 'entire'} period's data..."

    def predict_svr(self, symbol):
        if not self.svr_regressor.is_fitted:
            return "SVR model not fitted. Please fit the model with historical data first."
        predicted_price = self.svr_regressor.predict()
        return f"Predicted price for {symbol} with SVR is {predicted_price[0]}"