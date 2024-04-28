import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

class StockPricePredictor:
    def __init__(self, historical_prices):
        self.historical_prices = historical_prices
        self.X = None
        self.y = None
        self.model = None

    def prepare_data(self):
        # Splitting data into features (X) and target variable (y)
        self.X = np.array([i for i in range(len(self.historical_prices))]).reshape(-1, 1)
        self.y = np.array(self.historical_prices).reshape(-1, 1)

    def train_model(self):
        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Creating and training the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Evaluating the model
        y_pred = self.model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_pred, y_test)
        print("Root Mean Squared Error:", rmse)
        print("R2 Score:", r2)

    def predict_price(self, days_ahead):
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None
        else:
            # Predicting future prices
            future_X = np.array([i for i in range(len(self.historical_prices), len(self.historical_prices) + days_ahead)]).reshape(-1, 1)
            future_prices = self.model.predict(future_X)
            return future_prices
    
    def plot_prices(self, days_ahead, title = 'Stock Price Prediction'):
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
        else:
            # Plotting historical prices
            plt.plot(self.X, self.y, label='Historical Prices')

            # Plotting predicted prices
            future_X = np.array([i for i in range(len(self.historical_prices), len(self.historical_prices) + days_ahead)]).reshape(-1, 1)
            future_prices = self.model.predict(future_X)
            plt.plot(future_X, future_prices, label='Predicted Prices', linestyle='--', color='red')

            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.title(title)
            plt.legend()
            plt.show()

if __name__ == "__main__":
    # Example
    historical_prices = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    predictor = StockPricePredictor(historical_prices)
    predictor.prepare_data()
    predictor.train_model()

    future_prices = predictor.predict_price(3)
    print("Predicted stock prices for the next 3 days:\n", future_prices)
    predictor.plot_prices(3)

    