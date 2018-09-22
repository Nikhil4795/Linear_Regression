"""
Created on Sat Sep 1 00:00:00 2018

@author: Nikhil
"""
#### Simple Linear regression 

### y = mx+c
# y = dependent variable
# m = slope
# c = coeffiecents
# x = independent variable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

salary_dataset = pd.read_csv('Salary_Data.csv')
#salary_dataset = pd.read_csv('input_data.csv')

x = salary_dataset.iloc[:, :-1].values
y = salary_dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, lin_reg.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Train Data set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, lin_reg.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test Data set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import mean_squared_error, r2_score

print('r^2 statistic: %.2f' % r2_score(y_test, y_pred))

slop=lin_reg.coef_[0]
intercpt=lin_reg.intercept_
print("slope=",slop, "intercept=",intercpt)

""" 
    If you have any questions or suggestions regarding this script,
    feel free to contact me via nikhil.ss4795@gmail.com
"""
