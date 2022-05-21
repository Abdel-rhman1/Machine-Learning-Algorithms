import numpy as np

import pandas as pd
import math as m
import matplotlib.pyplot as plt
from regex import P

# step_One 
print("*" * 100)
print("Step#1 Importing Data")

data = pd.read_csv('data/cars.csv')

print( "*"*20 ,"Data Sample" , "*"*20 , end = "\n")
print(data.head())


X = data[{"enginesize" , "carlength" , "carwidth" , "carheight"}].values


Y = data['price'].values

print( "*"*20 ,"Data Info" , "*"*20 , end = "\n")
print(data.info())

print("*" * 100)


print("*" * 100)

print("Step # 2 Visulizing Data")

plt.figure(figsize=(10,10))
plt.title("Data Visualization")
plt.xlabel("enginsize")
plt.ylabel("price")
plt.scatter(data['enginesize'] , Y)
plt.show()
plt.figure(figsize=(10,10))
plt.title("Data Visualization")
plt.xlabel("enginsize")
plt.ylabel("price")
plt.pie(Y)
plt.show()
print("*" * 100)


class preprocessing:

    def __init__(self , X , Y):
        self.X = X
        self.Y = Y
        self.n , self.m = X.shape

    def splitData(self):
        #Split Data Into Traing Data and Testign Data
        pound =  m.ceil(0.8* self.n)
        x_training = self.X[0:pound]
        y_training = self.Y[0:pound]
        x_testing = self.X[pound+1:self.n]
        y_testing = self.Y[pound+1:self.n]
        return x_training , y_training , x_testing , y_testing

    def scale(self):
        x_scaled = self.X - np.mean(self.X, axis=0)
        x_scaled = x_scaled / np.std(x_scaled, axis=0)
        self.X = x_scaled
        return x_scaled





class linearRession:
    def __init__(self, X , Y , lRate , theta , iterations):
        self.X = X
        self.Y = Y
        self.theta = theta
        self.lRate = lRate
        self.iterations = iterations
        self.costs = []
        self.n , self.m = X.shape

    def fit(self):
        for i in range(self.iterations):
            y_prec = self.theta[0] + np.dot(self.X , self.theta[1:])
            mse = (1/self.n) * np.sum((y_prec - self.Y)**2)
            

            d_theta0 = (1/self.n) * np.sum(y_prec - self.Y)
            d_theta = (1/self.n) * np.dot(self.X.T , (y_prec - self.Y))

            self.theta[0] = self.theta[0] - self.lRate * d_theta0
            self.theta[1:] = self.theta[1:] - self.lRate * d_theta

            self.costs.append(mse)

        return self.costs

    def predict(self , x_testing , y_testing):
        predicted = self.theta[0] + np.dot(x_testing , self.theta[1:])
        return (1/x_testing.shape[0]) * np.sum((predicted - y_testing)**2)

##Model Paraeters
learning_Rate = 0.001
iterations = 1000
theta = np.zeros(1+X.shape[1])

preprocessing = preprocessing(X , Y)

x_scaled = preprocessing.scale()

training_x , training_y , testing_x , testing_y = preprocessing.splitData()


Linear = linearRession(training_x , training_y , 0.01 , theta ,1000)



loses = Linear.fit()
print(loses[0] , " "*20 , loses[-1])



testing_error = Linear.predict(testing_x , testing_y)


print("Testing_Error" , testing_error)
