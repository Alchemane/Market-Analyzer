class DataProcessor:
    def __init__(self):
        pass

    def process_historical_data(self, time_series):
        dates = []
        closing_prices = []
        for date, daily_data in time_series.items():
            dates.append(date)
            closing_prices.append(float(daily_data["4. close"]))
        return dates, closing_prices