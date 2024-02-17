import requests

class AlphaVantage:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_real_time_price(self, symbol):
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "Global Quote" in data and "05. price" in data["Global Quote"]:
                price = data["Global Quote"]["05. price"]
                return price
            else:
                return None
        else:
            response.raise_for_status()
    
    def fetch_historical_data(self, symbol):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize=compact"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            time_series = data.get("Time Series (Daily)", {})
            return time_series
        else:
            response.raise_for_status()
