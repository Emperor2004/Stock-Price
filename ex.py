import yfinance as yf

ticker_symbol = "AAPL"

data = yf.download(ticker_symbol, start="2023-01-01", end="2023-04-01")

with open("exercise.csv", "w") as file:
    file.write(str(data))