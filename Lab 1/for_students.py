from data import get_data, inspect_data, split_data
import numpy as np
import matplotlib.pyplot as plt

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]
X = np.c_[np.ones(len(x_train)), x_train]  # create a matrix with 1s and x_test
beta = np.linalg.inv(X.T @ X) @ X.T @ y_train  # closed-form solution
theta_best = beta
print('theta:', theta_best)

# TODO: calculate error
y_pred = theta_best[0] + theta_best[1] * x_test
mse = np.mean((y_test - y_pred) ** 2)
print('MSE:', mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()


# TODO: standardization
x_train = (x_train - np.mean(x_test)) / np.std(x_test)
y_train = (y_train - np.mean(y_test)) / np.std(y_test)


# TODO: calculate theta using Batch Gradient Descent
learning_rate = 0.05
n_iterations = 500
N = len(x)
theta = np.random.rand(2, 1)
X = np.c_[np.ones(len(x_train)), x_train]
Y = np.c_[y_train]

for epoch in range(n_iterations):
    # variant 1
    # dldw = 0.0
    # dldb = 0.0
    # for xi, yi in zip(x_train, y_train):
    #     dldw += -2 * xi * (yi - (theta * xi + b))
    #     dldb += -2 * (yi - (theta * xi + b))
    # theta -= dldw * learning_rate
    # b -= dldb * learning_rate

    # variant 2
    # dldw = -2 * np.mean(x_train * (y_train - (theta * x_train + b)))
    # dldb = -2 * np.mean((y_train - (theta * x_train + b)))
    # theta -= dldw * learning_rate
    # b -= dldb * learning_rate
    
    # variant 3
    mse = X.T @ (X @ theta - Y) / N * 2
    theta -= mse * learning_rate


theta = np.array(theta.T)[0] # convert to 1D array
# revert the standardization
theta[1] = theta[1] * np.std(y_test) / np.std(x_test)
theta[0] = theta[0] * np.std(y_test) + np.mean(y_test) - np.mean(x_test) * theta[1]

theta_best = theta
print('theta:', theta_best)

# TODO: calculate error
y_pred = theta_best[0] + theta_best[1] * x_test
mse = np.mean((y_test - y_pred) ** 2)
print('MSE:', mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
