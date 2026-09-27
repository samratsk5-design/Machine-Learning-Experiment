# Problem 3 (Polynomial Regression)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

data = pd.read_csv("Experiment-4/house_data.csv")

data["total_sqft"] = pd.to_numeric(data["total_sqft"], errors="coerce")

data = data.dropna(subset=["total_sqft", "bhk", "price"])

X = data[["total_sqft", "bhk"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_pred)
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)
poly_r2 = r2_score(y_test, poly_pred)

print("Linear Regression R2 Score:", round(linear_r2, 4))
print("Polynomial Regression R2 Score:", round(poly_r2, 4))