import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix




print('Awanish Msihra Project part 1')

dataset = pd.read_csv('DataSet.csv')
print( 'The Data set is : \n', dataset)

X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.25)



sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)





print('The Training Data is : \n ' , X_train )
print('The Testing Data is : \n ',X_test)




classifier = KNeighborsClassifier( n_neighbors = 5 , metric='minkowski' )

classifier.fit(X_train,Y_train)

Y_prediction = classifier.predict(X_test)

print( 'The Predicted Data Set',  Y_prediction)

print('\n\n\nThe Testing Data Set was ' , Y_test )


cm = confusion_matrix(Y_test,Y_prediction)

print('\n\n\nThe Comparision  ' , cm)

accuracy = (cm[0][0] + cm[1][1] )/(cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1] ) 

print( 'The Accuracy of the model is = ' ,(cm[0][0] + cm[1][1] )/(cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1] ) )

if(accuracy>0.68):
	print('This Model is Acceptable')


