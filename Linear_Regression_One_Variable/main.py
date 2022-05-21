#step #1 importing using Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#step #2  Load Data
data = pd.read_csv('../data/student_scores.csv')
#step #3 printing and Visualizing Data

x = data['Hours'].values

y = data['Scores'].values

plt.figure(figsize=(8,6))
plt.title('Student Scores Visualization')
plt.scatter(x , y)
plt.xlabel('hours')
plt.ylabel('scores')
plt.show()

# Initial Paramter
learing_rate = 0.001
num_iteration = 10000
theta1 = 0.0
theta2 = 0.0
n = x.shape[0]

losses = []


for i in range(num_iteration):
    h_x = theta1 + theta2 * x
    mse = (1/n*np.sum(h_x - y)**2)
    losses.append(mse)
    d_theta_1 = (1/n) * np.sum(h_x-y)
    d_theta_2 = (1/n) * np.sum( (h_x-y)*x)
    theta1 = theta1 - learing_rate * d_theta_1
    theta2 = theta2 - learing_rate * d_theta_2

# x = int(input("Enter Score That You Want To predict:\n"));
x_pre = 9
predict = theta1 + x_pre * theta2


x_line = np.linspace(0,10,100)

print(x_line)
y_line = theta1 + theta2*x_line
plt.figure(figsize=(8,6))
plt.title('Data distribution')
plt.plot(x_line, y_line, c='r')
plt.scatter(x, y, s=10)
plt.xlabel('hours')
plt.ylabel('score')
plt.show()




plt.title('Loss values')
plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()