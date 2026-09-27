# Problem 3 (Correlation Heatmap)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
corr_matrix = df.corr()

print("--- Correlation Matrix ---")
print(corr_matrix)
corr = corr_matrix.copy()

for i in range(len(corr)):
    corr.iloc[i, i] = -1
pair = corr.stack().idxmax()
value = corr.stack().max()

print("\n--- Strongest Positive Correlation ---")
print("Feature 1:", pair[0])
print("Feature 2:", pair[1])
print("Correlation:", value)

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap - Wine Dataset")
plt.tight_layout()
plt.show()