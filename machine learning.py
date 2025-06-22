from sklearn.linear_model import LinearRegression
import pandas as pd
df = pd.read_csv("/home/danktr/kod/machine learning/Student_Marks.csv")
y = df[["Marks"]]
x = df[["number_courses", "time_study"]]
l=LinearRegression()
model=l.fit(x,y)
model.predict([[4,4]])
df[['Marks']].max()
model.score(x,y)