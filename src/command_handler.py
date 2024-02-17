import requests
from alpha_vantage import AlphaVantage

class CommandHandler:
    def __init__(self, main_window=None, api_key=""):
        self.main_window = main_window
        self.av = AlphaVantage(api_key=api_key)

    def handler(self, command):
        if command == "get list cmd":
            return self.get_list_cmd()
        elif command.startswith("get price"):
            symbol = command.split()[2]
            return self.get_price(symbol)
        elif command.startswith("show history"):
            symbol = command.split()[2]
            self.main_window.show_historical_data(symbol)
            return "Displaying historical data..."

    def get_list_cmd(self):
        commands = [
            "get price {symbol} - Returns the price of the specified symbol.",
            "get list cmd - Lists all available commands.",
            "show history {symbol} - Displays historical price data for the specified symbol."
        ]
        return "\n".join(commands)
    
    def get_price(self, symbol):
        try:
            price = self.av.fetch_real_time_price(symbol=symbol)
            if price is not None:
                return f"Price of {symbol} is {price}"
            else:
                return f"Failed to fetch data: '{symbol}' not recognized, data unavailable, or API limit of 25 calls has been reached."
        except requests.exceptions.HTTPError as e:
            return f"Failed to fetch data: Could not connect to the API. Error: {e}"