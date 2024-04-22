from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

end = datetime.now().date

class StockPrice:
    
    
    def __init__(self, window_size, start_date, end_date) -> None:
        self.window_size = window_size
        self.start_date = start_date
        self.end_date = end_date
        
    @classmethod
    def ofWeek(start_date = datetime(end.year, end.month, end.day -7), end_date = end):
        return StockPrice(7, start_date= start_date, end_date= end_date)
    
    @classmethod
    def ofMonth(start_date = datetime(end.year, end.month -1, end.day), end_date = end):
        return StockPrice(30, start_date= start_date, end_date= end_date)
    
    @classmethod
    def ofYear(start_date = datetime(end.year -1, end.month, end.day), end_date = end):
        return StockPrice(365, start_date= start_date, end_date= end_date)

    def predict(self):
        pass
    
    def calculate(self):
        pass
    