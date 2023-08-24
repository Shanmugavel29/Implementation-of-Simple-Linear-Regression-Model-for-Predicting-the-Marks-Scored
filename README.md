# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries
2.Set variables for assigning dataset values
3.Import linear regression from sklearn.
4.Compare the graphs and hence we obtained the linear regression for the given datas
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Shanmugavel.R.M
RegisterNumber:212222230142
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

#segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,-1].values
Y

#spitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#display predicted values
Y_pred

#display actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="Red")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
###df.head()
"C:\Users\SEC\Pictures\Screenshots\Screenshot 2023-08-24 112038.png"
###df.tail()
"C:\Users\SEC\Pictures\Screenshots\Screenshot 2023-08-24 112047.png"
###Array value of X
"C:\Users\SEC\Pictures\Screenshots\Screenshot 2023-08-24 110410.png"
###Array value of Y
"C:\Users\SEC\Pictures\Screenshots\Screenshot 2023-08-24 110418.png"
###values of Y prediction
"C:\Users\SEC\Pictures\Screenshots\Screenshot 2023-08-24 110426.png"
###Array value of Y test
"C:\Users\SEC\Pictures\Screenshots\Screenshot 2023-08-24 110433.png"
###Training set Graph
"C:\Users\SEC\Pictures\Screenshots\Screenshot 2023-08-24 110441.png"
###Test set Graph
"C:\Users\SEC\Pictures\Screenshots\Screenshot 2023-08-24 110452.png"
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
