# Problem 2 (Multiple Linear Regression)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Experiment-4/house_data.csv")

print(data.head())
X = data[["total_sqft", "bhk"]]
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nActual Prices:", y_test.values)
print("Predicted Prices:", y_pred)

print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_house = [[1500, 3]]

predicted_price = model.predict(new_house)

print("\nPredicted Price:")
print(predicted_price[0])