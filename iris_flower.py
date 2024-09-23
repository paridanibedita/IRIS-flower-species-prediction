
##BUSINESS PROBLEM :
''' Iris flower has three species; setosa, versicolor, and virginica, which differs according to their
measurements. Now assume that you have the measurements of the iris flowers according to
their species, and here your task is to train a machine learning model that can learn from the
measurements of the iris species and classify them.'''


##BUSINESS OBJECTIVE :
''' Determine species by considering measurement of flowers.'''
##BUSINESS CONSTRAINTS:
'''Minimize Cost of Detection'''
    
##SUCCESS CRITERIA: 
'''Business success criteria - Increase effectiveness of species detection by at least 50%
ML success criteria - Achieve an accuracy of more than 80%
Economic success criteria - Increasing revenue by 20%'''

##Data Understanding:
'''
1.Id : serial number
2.SepalLengthCm : sepal length in cm
3.SepalWidthCm: sepal width in cm
4.PetalLengthCm: petal length in cm
5.PetalWidthCm: petal width in cm
6.Species: species name (setosa,versicolor,virginica) '''


##Load all the necessary libraries:
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import pickle    
import joblib
    
##load the data
data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Documents\Data science\project2\TASK1\Iris.csv")    
data    
    
##describe the data
data.describe()

data.info()    

##check how many unique values there in the data set
data["Species"].value_counts()    

##check null values
data.isnull().sum()

##drop id column from the data set
data = data.drop("Id",axis = 1)    
data.columns
    
##Encode the target variable (Species) into numerical format
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

##Split the data into features (X) and target (y)
X = data.drop(columns=['Species'])
y = data['Species']

##check outliers with boxplot
X.plot(kind= "box", subplots = True,sharey = False,figsize = (15,13))

#applying winsorization techniques using iqr method 
winsor= Winsorizer(capping_method = 'iqr',tail = "both",fold = 1.5,variables =["SepalWidthCm"])

# Fit the winsorizer to the data
outlier = winsor.fit(X[["SepalWidthCm"]])

# Save the winsorizer model
joblib.dump(outlier, 'winsor.pkl')

# Apply the transformation
X["SepalWidthCm"] = outlier.transform(X[["SepalWidthCm"]]) 

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    

# Define base learners (individual models)
base_learners = [
    ('decision_tree', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True)),
    ('random_forest', RandomForestClassifier(n_estimators = 10, random_state = 42))
]    
    
# Define meta-model (the model used to combine base learners' predictions)
meta_model = LogisticRegression()    
    
# Create the stacking classifier
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

# Train the stacking classifier
stacking_clf.fit(X_train, y_train)


# Evaluate the model
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Stacking Classifier Accuracy: {accuracy:.2f}')    
    
# Save the Stacking model 
pickle.dump(stacking_clf, open('stacking_iris.pkl', 'wb'))  
import os     
os.getcwd()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
