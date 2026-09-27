#Problem 2 (Filter Students Scoring Above 80)
import pandas as pd

data = {
    "Student Name": ["Ankit", "Mohak", "Souvik", "Sahida", "Roma"],
    "Roll Number": [10, 25, 13, 34, 45],
    "Marks": [79, 93, 69, 81, 97],
    "Attendance": [68, 76, 79, 80, 91]
}

df = pd.DataFrame(data)
print("Complete DataFrame.")
print(df)

print("\nStudents Scoring Above 80")
print(df[df["Marks"] > 80])