#Problem 3 (StandardScaler vs MinMaxScaler)
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    "Age": [21, 25, 28, 37, 34, 33],
    "Years_Experience": [1, 2, 4, 9, 8, 6]
}
df = pd.DataFrame(data)
print("Original Data")
print(df)

standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(df)

print("\n--- StandardScaler Data ---")
print(standard_data)
print("\nStandardScaler Range:")
print("Minimum:", standard_data.min())
print("Maximum:", standard_data.max())

minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(df)

print("\nMinMaxScaler Data")
print(minmax_data)
print("\nMinMaxScaler Range:")
print("Minimum:", minmax_data.min())
print("Maximum:", minmax_data.max())