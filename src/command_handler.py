import requests
from alpha_vantage import AlphaVantage
from machine_learning import LinearRegressor

class CommandHandler:
    def __init__(self, main_window=None, api_key=""):
        self.main_window = main_window
        self.av = AlphaVantage(api_key=api_key)
        self.linear_regressor = LinearRegressor()
        self.command_map = {
        "get price": self.get_price,
        "get list cmd": self.get_list_cmd,
        "show history": self.show_historical_data,
        "fit linear_regression": self.set_fit_linear_regression,
        "predict linear_regression": self.get_linear_regression
    }

    def handle_command(self, command):
        parts = command.split()
        cmd, *args = parts
        if cmd in self.command_map:
            return self.command_map[cmd](*args)
        else:
            return "Unknown command"

    def get_list_cmd(self):
        commands = [
            "get price {symbol} - Returns the price of the specified symbol.",
            "get list cmd - Lists all available commands.",
            "show history {symbol} - Displays historical price data for the specified symbol.",
            "fit linear_regression {symbol} - Fits the linear regression model to the historical data of the specified symbol.",
            "predict linear_regression {symbol} - Predicts the next price for the specified symbol using the fitted linear regression model. Requires the model to be fitted first."
            ]
        return "\n".join(commands)
    
    def show_historical_data(self, symbol):
        historical_data = self.av.fetch_historical_data(symbol)
        dates, prices = self.data_processor.process_historical_data(historical_data)
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
        
    def set_fit_linear_regression(self, symbol):
        historical_data = self.av.fetch_historical_data(symbol=symbol)
        prices = [float(data["4. close"]) for data in historical_data.values()]
        days = list(range(len(prices)))
        self.linear_regressor.fit(days, prices)

    def get_linear_regression(self, symbol):
        if not self.linear_regressor.is_fitted:
            return "Model not fitted. Please fit the model with historical data first."
        predicted_price = self.linear_regressor.predict()
        return f"Predicted price for {symbol} is {predicted_price[0]}"