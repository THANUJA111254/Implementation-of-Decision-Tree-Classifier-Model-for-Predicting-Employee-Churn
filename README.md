# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.


2.Calculate the null values present in the dataset and apply label encoder.


3.Determine test and training data set and apply decison tree regression in dataset.


4.Calculate Mean square error,data prediction and r2.



## Program:
```
import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PANGA THANUJA
RegisterNumber:  212224040231
*/
```

## Output:

Data Head:
![exp 8 ml 1](https://github.com/user-attachments/assets/6dca7114-ff17-41b0-9dff-e33d184a0cbd)




Data Head:
![exp8 ml 2](https://github.com/user-attachments/assets/5a8862aa-3796-4009-b836-52a7e430d942)
![exp8ml 3](https://github.com/user-attachments/assets/3dd51dd0-eb2a-4ff0-94a4-2374dbb04c48)


isnull() sum():


![exp 8 ml 4](https://github.com/user-attachments/assets/0357b833-e18c-4757-8ac0-697ca00e6fd9)


Data Head for salary:


![exp 8 ml 5](https://github.com/user-attachments/assets/b556ee78-5a29-440e-9166-9fa1286dcbc4)



MSE:


![ML 9](https://github.com/user-attachments/assets/ee498c62-465a-4c4e-a009-151a77da8a59)



R2:


![10](https://github.com/user-attachments/assets/8f03e6f5-e956-4b53-a063-52a0da3dbde0)




DATA PREDICTION:

![ML11](https://github.com/user-attachments/assets/65b8e4b8-a49d-4eb3-b60d-409bd27ce817)







## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
