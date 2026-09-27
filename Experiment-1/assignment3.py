#Problem 3  (Dynamically Calculate Student Grades)
import pandas as pd

data = {
    "Student Name": ["Ankit", "Mohak", "Souvik", "Sahida", "Roma"],
    "Roll Number": [10, 25, 13, 34, 45],
    "Marks": [79, 93, 69, 81, 97],
}
df=pd.DataFrame(data)
def cal_grade(Marks):
    if Marks>= 90:
        return "O"
    elif Marks>=80:
        return "E"
    elif Marks>=70:
        return "B"
    elif Marks>=60:
        return "C"
    else:
        return "F"
df["Grade"]=df["Marks"].apply(cal_grade)
print(df)