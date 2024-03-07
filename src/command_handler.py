import requests
from alpha_vantage import AlphaVantage
from machine_learning import (LinearRegressor, SVRRegressor, PolynomialRegressor, 
                              DecisionTreeRegression, RandomForestRegression, ARIMAModel, LSTMModel, 
                              ModelEvaluator, ModelTrainer, DataPreprocessor)
from data_processor import DataProcessor
from concurrent.futures import ThreadPoolExecutor
import textwrap, functools

class CommandHandler:
    def __init__(self, main_window=None, api_key=""):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.main_window = main_window
        self.av = AlphaVantage(api_key=api_key)
        self.data_processor = DataProcessor()
        self.data_preprocessor = DataPreprocessor()
        self.command_map = {
            "get price": self.get_price,
            "lst cmd": self.get_list_cmd,
            "show hist": self.show_historical_data,
            "get %price": self.get_price_change_percentage,
            "get mktcap": self.get_market_cap,
            "get vol": self.get_volume,
        }
        self.models = {
            "lr": LinearRegressor(),
            "svr": SVRRegressor(),
            "polyr": PolynomialRegressor(),
            "dtr": DecisionTreeRegression(),
            "rfr": RandomForestRegression(),
            #: MultipleLinearRegressor(),
            "arima": ARIMAModel(),
            "lstm": LSTMModel(),
        }
        self.trained_models = {}
        for model_key in self.models.keys():
            self.command_map[f"fit {model_key}"] = lambda symbol, days=None, model_key=model_key: self.fit_model(model_key=model_key, symbol=symbol, days=days)
            self.command_map[f"pred {model_key}"] = lambda symbol, model_key=model_key: self.predict_model(model_key=model_key, symbol=symbol)

    def run_async(self, func, callback=None, *args, **kwargs):
        # Submit a function to be executed asynchronously by the ThreadPoolExecutor.
        if callback:
            def callback_on_complete(future):
                self.main_window.queue_function(functools.partial(callback, future.result()))
            future = self.executor.submit(func, *args, **kwargs)
            future.add_done_callback(lambda future: callback_on_complete(future))
        else:
            return self.executor.submit(func, *args, **kwargs)
        
    def handle_command(self, command):
        print(f"Received command: {command}")
        command = command.strip()
        parts = command.split()
        cmd_key = ' '.join(parts[:2]) if len(parts) > 1 else parts[0]
        print(f"cmd_key: {cmd_key}")
        if cmd_key == "lst cmd":
            return self.get_list_cmd()
        print(f"Command map keys: {list(self.command_map.keys())}")
        if cmd_key in self.command_map:
            args = parts[2:] if len(parts) > 2 else []
        else:
            cmd_key = parts[0]
            args = parts[1:] if len(parts) > 1 else []

        if cmd_key not in self.command_map:
            return "Unknown command"

        try:
            # Handle commands that require symbol and possibly days
            if cmd_key in ["fit lr", "fit svr", "fit polyr", "fit dtr", "fit rfr", "fit arima", "fit lstm",
                            "pred lr", "pred svr", "pred polyr", "pred dtr", "pred rfr", "pred arima", "pred lstm", "get %price"]:
                if not args:  # Symbol provided?
                    return "Error: Symbol is required."
                symbol = args[0]
                days = int(args[1]) if len(args) > 1 and cmd_key not in ["fit lstm", "pred lstm"] else None
                if "fit" in cmd_key or cmd_key == "get %price":
                    return self.command_map[cmd_key](symbol, days) if days is not None else self.command_map[cmd_key](symbol)
                else:
                    return self.command_map[cmd_key](symbol)
            else:
                return self.command_map[cmd_key](*args)
        except ValueError as e:
            return f"Error: {str(e)}"

    def get_list_cmd(self):
        commands = [
            ("lst cmd", "Lists all available commands."),
            ("get price {symbol}", "Returns the current price of the specified symbol."),
            ("show hist {symbol}", "Displays historical price data for the specified symbol."),
            ("get %price {symbol} {days}", "Returns the percentage change of the price of the specified symbol since the beginning. Optionally, specify the number of recent days to use."),
            ("get mktcap {symbol}", "Returns the current market capitalization of the specified symbol."),
            ("get volume {symbol}", "Returns the trading current volume of the specified symbol."),
            ("fit lr {symbol} {days}", "Fits the linear regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use."),
            ("pred lr {symbol}", "Predicts the next price for the specified symbol using the fitted linear regression model. Requires the model to be fitted first."),
            ("fit svr {symbol} {days}", "Fits the support vector regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use."),
            ("pred svr {symbol}", "Predicts the next price for the specified symbol using the fitted support vector regression model. Requires the model to be fitted first."),
            ("fit polyr {symbol} {days}", "Fits the polynomial regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use."),
            ("pred polyr {symbol}", "Predicts the next price for the specified symbol using the fitted polynomial regression model. Requires the model to be fitted first."),
            ("fit dtr {symbol} {days}", "Fits the decision tree regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use."),
            ("pred dtr {symbol}", "Predicts the next price for the specified symbol using the fitted decision tree regression model. Requires the model to be fitted first."),
            ("fit rfr {symbol} {days}", "Fits the random forest regression model to the historical data of the specified symbol. Optionally, specify the number of recent days to use."),
            ("pred rfr {symbol}", "Predicts the next price for the specified symbol using the fitted random forest regression model. Requires the model to be fitted first."),
            ("fit arima {symbol} {days}", "Fits the Autoregressive integrated moving average model to the historical data of the specified symbol. Optionally, specify the number of recent days to use."),
            ("pred arima {symbol}", "Predicts the next price for the specified symbol using the fitted Autoregressive integrated moving average model. Requires the model to be fitted first."),
            ("fit lstm {symbol} {days}", "Fits the Long short-term memory model to the historical data of the specified symbol. Optionally, specify the number of recent days to use."),
            ("pred lstm {symbol}", "Predicts the next price for the specified symbol using the fitted Long short-term memory model. Requires the model to be fitted first."),
        ]
        first_column_width = 30
        max_description_width = 70
        formatted_commands = []
        for command, description in commands:
            wrapped_description_lines = textwrap.wrap(description, width=max_description_width)
            
            first_line = f"{command.ljust(first_column_width)}{wrapped_description_lines[0]}"
            formatted_commands.append(first_line)
            
            for additional_line in wrapped_description_lines[1:]:
                formatted_commands.append(' ' * first_column_width + additional_line)

        return "\n".join(formatted_commands)
    
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
        
    def get_price_change_percentage(self, symbol, days=None):
        try:
            historical_data = self.av.fetch_historical_data(symbol)
            sorted_dates = sorted(historical_data.keys())
            if days is None or days >= len(sorted_dates):
                target_date = sorted_dates[0]
            else:
                target_date = sorted_dates[-days - 1]
            
            latest_date = sorted_dates[-1]
            latest_price = float(historical_data[latest_date]["4. close"])
            target_price = float(historical_data[target_date]["4. close"])
            percentage_change = ((latest_price - target_price) / target_price) * 100
            return f"The price of {symbol} has changed {percentage_change:.2f}% since {'the beginning' if days is None else f'the last {days} days'}."

        except Exception as e:
            return f"Failed to calculate price change: {str(e)}"
        
    def get_market_cap(self, symbol):
        market_cap = self.av.fetch_market_capitalization(symbol)
        return f"Market Capitalization for {symbol}: {int(market_cap):,}"
    
    def get_volume(self, symbol):
        volume = self.av.fetch_real_time_volume(symbol)
        return f"Today's trading volume for {symbol}: {int(volume):,}"
    
    def fit_model(self, model_key, symbol, days=None):
        if model_key not in self.models:
            return "Error: Unknown model"
        historical_data = self.av.fetch_historical_data(symbol=symbol)
        model = self.models[model_key]  # Initialize model from the models dictionary
        metrics = "N/A"
        if model_key == 'arima':
            data = self.data_preprocessor.prepare_data_for_arima(historical_data, days)
            model.fit(data)
        elif model_key == 'lstm':
            X, y = self.data_preprocessor.prepare_data_for_lstm(historical_data, days)
            input_shape = (10, 1)
            print("Training input_shape:", input_shape)
            model.initialize_model(input_shape=input_shape)
            model.fit(X, y, epochs=50, batch_size=32)
        else:
            X, y = self.data_preprocessor.prepare_data(historical_data, days)
            trainer = ModelTrainer(model)
            X_test, y_test = trainer.train_and_split(X, y)
            evaluator = ModelEvaluator(model)
            metrics = evaluator.evaluate(X_test, y_test)

        self.trained_models[model_key] = (model, metrics)
        return f"Fitted {symbol} to the {model_key.upper()} model. Metrics: {metrics}"
        
    def predict_model(self, model_key, symbol):
        if model_key not in self.models or model_key not in self.trained_models:
            return "Error: Model not trained or unknown model"
        historical_data = self.av.fetch_historical_data(symbol)
        model, _ = self.trained_models[model_key]
        if model_key == 'lstm':
            X_new = self.data_preprocessor.prepare_X_new_for_lstm(historical_data)
            print(type(X_new))
            print(X_new)
            print(X_new.shape)
            predicted_price = model.predict(X_new)
        else:
            # For ARIMA and other models
            X_new = self.data_preprocessor.prepare_X_new()
            predicted_price = model.predict(X_new)
        
        return f"Predicted price for {symbol} using {model_key.upper()} model is {predicted_price[0]}"