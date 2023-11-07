# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Libraries.
2. Import the csv file.
3. Declare X and Y values with respect to the dataset.
4. Plot the graph using the matplotlib library.
5. Print the plot.
6. End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SHANMUGAVEL.R.M
RegisterNumber: 212222230142
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
dataset.head()
X=dataset.iloc[:, :-1].values
X
y=dataset.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred
y_test
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
```

## Output:

![Screenshot 2023-08-24 090723](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/57f2db55-8f27-466c-9df8-39aa70f806d8)

![Screenshot 2023-08-24 090808](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/77492632-c012-4107-83c8-10ffcd2e228a)

![Screenshot 2023-08-24 090845](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/1135cd01-a2fe-4e51-afb2-774f084312a4)

![Screenshot 2023-08-24 090917](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/1fd74109-307e-4cbf-9e02-22898e5aa096)

![Screenshot 2023-08-24 090950](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/12a42d1d-9558-4524-be57-1159e167369b)

![Screenshot 2023-08-24 091230](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/42880876-83b2-446f-8089-f8c10baf1e02)

![Screenshot 2023-08-24 091301](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/9a6c5889-f156-4210-a2f6-49e3d6abfe5e)

![Screenshot 2023-08-24 091337](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/ec28aec1-76b7-42f4-a733-11799cb3e9cd)

![Screenshot 2023-08-24 093517](https://github.com/praveenmax55/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497509/2b5381a7-cdf4-48e8-b21a-0c4f3f42615c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
