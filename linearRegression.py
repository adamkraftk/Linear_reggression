import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

# Load the Data
d = load_diabetes()
d_X = d.data[:, np.newaxis, 2]

# First 402 points of training data
dx_train = d_X[:-402]
dy_train = d.target[:-402]

# Last 20 points of testing data
dx_test = d_X[-20:] 
dy_test = d.target[-20:]

# function that determines the least squares 
def least_squares(x,y):
    
    np.squeeze(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = 0
    den = 0
    
    for i in range(len(x)):
        num += (x[i] - x_mean)*(y[i] - y_mean)
        den += (x[i] - x_mean)**2
    
    m = num/den
    c = y_mean - m * x_mean
    y_pred = m*x + c
    return y_pred

y_int = least_squares(dx_train,dy_train)

# Plotting the data
plt.scatter(dx_test, dy_test, c = 'g')
plt.scatter(dx_train,dy_train, c = 'r')
plt.plot(dx_train, y_int, c='b')
plt.legend(["Testing Data", "Training Data", "Line of best Fit"])
plt.show()