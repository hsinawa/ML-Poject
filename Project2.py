import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


print('Awanish Msihra Project part 2')
data = pd.read_csv('RegressionDataSet.csv')

print( 'The dataset is : ', data)

X = data.iloc[:,:-1].values
Y = data.iloc[:,:1].values

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=1/3)

regressor = LinearRegression()

regressor.fit(X_train , Y_train)

Y_prediction = regressor.predict(X_test)

print('The Predicted Values are ' , Y_prediction)

print('The Orginial Values are ' , Y_test)


plt.scatter(X_train,Y_train,color='#A569BD') 

#plt.show()

plt.plot(X_train,regressor.predict(X_train),color='blue')


plt.title('Salary vs Experience')
plt.xlabel('Experince')
plt.ylabel('Salary')

plt.show()