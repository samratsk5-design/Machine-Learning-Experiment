#Problem 1 (Complete EDA on Wine Dataset)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

print("First Five Rows")
print(df.head())

print("\nLast Five Rows ")
print(df.tail())
print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

print("\n--- Target Class Distribution ---")
print(df["target"].value_counts())

print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

df.hist(figsize=(15, 12), bins=15)
plt.suptitle("Wine Dataset - Feature Distributions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="alcohol",
    y="color_intensity",
    hue="target",
    palette="viridis"
)
plt.title("Alcohol vs Color Intensity")
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.show()
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")

plt.title("Wine Class Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(15, 7))
sns.boxplot(data=df.drop("target", axis=1))

plt.title("Boxplot of Wine Features")
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()