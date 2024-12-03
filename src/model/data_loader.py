import pandas as pd
import numpy as np
from model.stock import Stock
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.daily_returns = None
        self.expected_returns = None
        self.std_devs = None
        self.cov_matrix = None
        self.latest_prices = None

    def load_data(self):
        """
        Load the Excel file into a DataFrame.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")

        try:
            self.raw_data = pd.read_excel(self.file_path)
            print("Data loaded successfully.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the data: {e}")

    def preprocess_data(self):
        """
        Preprocess the raw data by setting the date column as the index and sorting the data.
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Please run 'load_data()' first.")

        if 'Dates' not in self.raw_data.columns:
            raise KeyError("The data must contain a 'Dates' column.")

        # Set 'Dates' as index
        self.raw_data.set_index('Dates', inplace=True)
        # Sort the DataFrame by date
        self.raw_data.sort_index(inplace=True)
        # print("Data preprocessing completed.")

    def calculate_daily_returns(self):
        """
        Calculate daily returns for each stock.
        """
        if self.raw_data is None:
            raise ValueError("Data not preprocessed. Please run 'preprocess_data()' first.")

        # Calculate daily percentage change
        self.daily_returns = self.raw_data.pct_change().dropna()
        # print("Daily returns calculated.")

    def calculate_statistics_with_geometric_average_returns(self):
        """
        Calculate expected returns, standard deviations, and the covariance matrix using geometric returns.
        """
        if self.daily_returns is None:
            raise ValueError("Daily returns not calculated. Please run 'calculate_daily_returns()' first.")

        # Annualization factor
        trading_days = 252

        # Number of periods (days)
        num_periods = len(self.daily_returns)

        # Calculate growth factors
        growth_factors = self.daily_returns + 1

        # Calculate cumulative growth
        cumulative_growth = growth_factors.prod()

        # Compute geometric mean daily return
        geometric_mean_daily_returns = cumulative_growth ** (1 / num_periods) - 1

        # Annualize the geometric mean daily return
        self.expected_returns = (1 + geometric_mean_daily_returns) ** trading_days - 1

        # Calculate log returns for standard deviation
        log_returns = np.log(growth_factors)

        # Calculate annualized standard deviations
        self.std_devs = log_returns.std() * np.sqrt(trading_days)

        # Calculate the covariance matrix
        self.cov_matrix = log_returns.cov() * trading_days

        print("Statistical calculations completed using geometric mean for expected returns.")

    def calculate_statistics_with_arithmetic_average_returns(self):
        """
        Calculate expected returns, standard deviations, and the covariance matrix.
        """
        if self.daily_returns is None:
            raise ValueError("Daily returns not calculated. Please run 'calculate_daily_returns()' first.")

        # Annualization factor
        trading_days = 252

        # Calculate expected returns and standard deviations
        self.expected_returns = self.daily_returns.mean() * trading_days
        self.std_devs = self.daily_returns.std() * np.sqrt(trading_days)

        # Calculate the covariance matrix
        self.cov_matrix = self.daily_returns.cov() * trading_days
        # print("Statistical calculations completed.")

    def get_latest_prices(self):
        """
        Retrieve the latest closing prices for each stock.
        """
        if self.raw_data is None:
            raise ValueError("Data not preprocessed. Please run 'preprocess_data()' first.")

        self.latest_prices = self.raw_data.iloc[-1]
        # print("Latest stock prices retrieved.")

    def create_stock_objects(self):
        """
        Create a list of Stock objects from the calculated statistics.
        """
        if any(v is None for v in [self.expected_returns, self.std_devs, self.latest_prices]):
            raise ValueError("Statistics not fully calculated. Please run all calculation methods first.")

        stocks = []
        for stock_name in self.raw_data.columns:
            stock = Stock(
                name=stock_name,
                expected_return=self.expected_returns[stock_name],
                std_dev=self.std_devs[stock_name],
                price=self.latest_prices[stock_name]
            )
            stocks.append(stock)
        # print("Stock objects created.")
        return stocks

    def get_data(self):
        """
        Execute all steps and return the list of Stock objects and the covariance matrix.
        """
        self.load_data()
        self.preprocess_data()
        self.calculate_daily_returns()
        self.calculate_statistics_with_arithmetic_average_returns()
        self.get_latest_prices()
        stocks = self.create_stock_objects()
        return stocks, self.cov_matrix
