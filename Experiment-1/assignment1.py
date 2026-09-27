#Problem 1 (Student Marks Statistics)
import numpy as np
marks=np.array([67,79,58,78,90,80,83,65,77,93])
mean=np.mean(marks)
median=np.median(marks)
st_dev=np.std(marks)
maximum=  np.max(marks)
minimum=  np.min(marks)

print("Mean:",mean)
print("Median:",median)
print("Standard deviation:",st_dev)
print("Maximum:",maximum)
print("Minimm:",minimum)