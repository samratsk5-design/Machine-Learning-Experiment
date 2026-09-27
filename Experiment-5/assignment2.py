# Problem 2 (Decision Threshold Comparison)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

study_hours = np.array([
    1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
    7, 8, 8, 9
])
attendance = np.array([
    50, 55, 60, 62, 68, 70, 75, 78,
    82, 85, 88, 90, 92, 94, 96, 98
])

X = np.column_stack((study_hours, attendance))
y = np.array([
    0, 0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]

print("Actual Values:", y_test)
print("Probabilities:", np.round(probability, 2))

for threshold in [0.3, 0.5, 0.7]:
    y_pred = (probability >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    print("\nThreshold:", threshold)
    print("Predicted Values:", y_pred)
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))