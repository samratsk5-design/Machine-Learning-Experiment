# Problem 1 (Student Pass/Fail Prediction)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

study_hours = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7])
attendance = np.array([50, 55, 60, 62, 68, 70, 75, 78, 82, 85, 88, 92])

X = np.column_stack((study_hours, attendance))
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Actual Values    :", y_test)
print("Predicted Values :", y_pred)
print("Accuracy         :", round(accuracy, 2))

new_students = np.array([
    [2, 60],
    [4, 75],
    [6, 85],
    [7, 90]
])

new_prediction = model.predict(new_students)
print("\nNew Student Predictions:")
for i in range(len(new_students)):
    if new_prediction[i] == 1:
        result = "Pass"
    else:
        result = "Fail"
    print("Study Hours:", new_students[i][0],
          "Attendance:", new_students[i][1],
          "Result:", result)