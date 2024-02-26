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
theta = np.random.rand()
b = np.random.rand()

for epoch in range(n_iterations):
    dldw = 0.0
    dldb = 0.0
    for xi, yi in zip(x_train, y_train):
        dldw += -2 * xi * (yi - (theta * xi + b))
        dldb += -2 * (yi - (theta * xi + b))

    theta -= (1/N) * dldw * learning_rate
    b -= (1/N) * dldb * learning_rate


# revert the standardization
theta = theta * np.std(y_test) / np.std(x_test)
b = b * np.std(y_test) + np.mean(y_test) - np.mean(x_test) * theta

theta_best[0] = b
theta_best[1] = theta

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
