import pandas as pd
import yfinance as yf
from model import StockPricePredictor

df = pd.read_csv(f"exercise.csv")
predictor = StockPricePredictor(df["Adj Close"])
predictor.prepare_data()
predictor.train_model()
future_prices = predictor.predict_price(3)
print("Predicted stock prices for the next 3 days:\n", future_prices)
predictor.plot_prices(3, title= key)
