# Problem 1 (House Price Prediction)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Experiment-4/house_data.csv")

print(data.head())
X = data[["total_sqft"]]
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

new_area = [[1500]]

predicted_price = model.predict(new_area)

print("\nPredicted Price for 1500 sq.ft:", predicted_price[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.xlabel("Area (sq.ft)")
plt.ylabel("House Price")
plt.title("House Price Prediction")

plt.show()