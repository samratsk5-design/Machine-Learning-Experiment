#Problem 3 (ROC Curve and AUC)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

hours = np.array([
    1, 2, 2, 3, 3, 4, 4, 5,
    5, 6, 6, 7, 7, 8, 8, 9
])
attendance_rate = np.array([
    48, 54, 59, 64, 69, 71, 76, 79,
    83, 86, 89, 91, 93, 95, 97, 99
])
features = np.column_stack((hours, attendance_rate))

result = np.array([
    0, 0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1
])

X_train, X_test, y_train, y_test = train_test_split(
    features, result, test_size=0.25, random_state=42, stratify=result
)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

probability = classifier.predict_proba(X_test)[:, 1]
false_positive, true_positive, threshold = roc_curve(
    y_test, probability
)

auc_value = roc_auc_score(y_test, probability)

print("AUC Score:", round(auc_value, 2))

plt.plot(false_positive, true_positive, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()