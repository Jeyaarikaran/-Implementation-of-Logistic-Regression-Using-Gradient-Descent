# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages
2.Read the dataset. 
3.Define X and Y array. 
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.

## Program:
```.py
  import pandas  as pd
  import numpy as np
  import matplotlib.pyplot as plt
  dataset=pd.read_csv('placement_Data.csv')
  dataset
  dataset=dataset.drop('sl_no',axis=1)
  dataset=dataset.drop('salary',axis=1)
  dataset["gender"]=dataset["gender"].astype('category')
  dataset["ssc_b"]=dataset["ssc_b"].astype('category')
  dataset["hsc_b"]=dataset["hsc_b"].astype('category')
  dataset["degree_t"]=dataset["degree_t"].astype('category')
  dataset["workex"]=dataset["workex"].astype('category')
  dataset["specialisation"]=dataset["specialisation"].astype('category')
  dataset["status"]=dataset["status"].astype('category')
  dataset["hsc_s"]=dataset["hsc_s"].astype('category')
  dataset.dtypes
  dataset["gender"]=dataset["gender"].cat.codes
  dataset["ssc_b"]=dataset["ssc_b"].cat.codes
  dataset["hsc_b"]=dataset["hsc_b"].cat.codes
  dataset["degree_t"]=dataset["degree_t"].cat.codes
  dataset["workex"]=dataset["workex"].cat.codes
  dataset["specialisation"]=dataset["specialisation"].cat.codes
  dataset["status"]=dataset["status"].cat.codes
  dataset["hsc_s"]=dataset["hsc_s"].cat.codes
  dataset
  X=dataset.iloc[:, :-1].values
  Y=dataset.iloc[:, -1].values
  y
  theta=np.random.randn(X.shape[1])
  Y=y
  def sigmoid(z):
      return 1/(1+np.exp(-z))
  def loss(theta,x,Y):
      h=sigmoid(X.dot(theta))
      return -np.sum(y* np.log(h)+(1-y)*np.log(1-h))
  
  def gradient_descent(theta,x,Y,alpha,num_iterations):
      m=len(y)
      for i in range(num_iterations):
          h=sigmoid(x.dot(theta))
          gradient=x.T.dot(h-Y)/m
          theta=alpha*gradient
      return theta
  theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
  def predict(theta,X):
      h=sigmoid(x.dot(theta))
      y_pred=np.where(h>=0.5,1,0)
      return y_pred
  y_pred=predict(theta,x)
  accuracy=np.mean(y_pred.flatten()==Y)
  print("Accuracy:=",accuracy)
  print(y_pred)
  print (y)
  xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
  y_prednew=predict(theta,xnew)
  print(y_prednew)
  xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
  y_prednew=predict(theta,xnew)
  print(y_prednew)
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JEYAARIKARAN P
RegisterNumber: 212224240064
*/
```

## Output:
 ## Placement Dataset
 ![Screenshot 2025-04-15 113649](https://github.com/user-attachments/assets/dd61f5d5-224e-4120-88c7-9957c314e2cd)

 
 ## Dataset After Feature Engineering

 
 ![Screenshot 2025-04-15 113956](https://github.com/user-attachments/assets/5a880959-eef4-43ff-9272-e2a49122ea32)
## Datatypes Of Feature Column
![Screenshot 2025-04-15 113920](https://github.com/user-attachments/assets/f2e38110-2447-4799-92b8-24905563c7e8)
 ## Dataset After Encoding
![Screenshot 2025-04-15 114055](https://github.com/user-attachments/assets/9dd5f737-1667-433c-a13c-6385a1680392)

 ## Y Values

![Screenshot 2025-04-16 101107](https://github.com/user-attachments/assets/e64696a8-69e2-489a-8983-17c0b1074d4e)
## Accuracy

![screenshot 2025-04-16 101108](https://github.com/user-attachments/assets/e42501f2-dbe8-45ba-9f9b-e3ba04716666)
## Y Predicted

![image](https://github.com/user-attachments/assets/5dd89d21-d3ed-4c2a-a6d6-0af4bb37d45a)

## Y Values

![image](https://github.com/user-attachments/assets/acdc7a04-0fd1-47b7-9249-2b17d2f960fc)
## Y Predicted With Different X Values

![image](https://github.com/user-attachments/assets/128e3b9c-d780-493d-bc1d-b93bd08bc748)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

