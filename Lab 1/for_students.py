from data import get_data, inspect_data, split_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
X = np.c_[np.ones(len(x_test)), x_test]  # create a matrix with 1s and x_test

beta = np.linalg.inv(X.T @ X) @ X.T @ y_test  # closed-form solution
theta_best[0] = beta[0]
theta_best[1] = beta[1]
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
theta_best = [0, 0]
y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()
x_test = x_test/500
y_test = y_test/500


# print('x_test:', x_test)
# print('y_test:', y_test)

learning_rate = 0.01
n_iterations = 1000
theta = np.random.randn(1, 1)
b = np.random.randn(1, 1)

# TODO: calculate theta using Batch Gradient Descent


def descent(x, y, theta, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = len(x)
    for xi, yi in zip(x, y):
        dldw += -2 * xi * (yi - (theta * xi + b))
        dldb += -2 * (yi - (theta * xi + b))

    theta -= (1/N) * dldw * learning_rate
    b -= (1/N) * dldb * learning_rate
    return theta, b


for epoch in range(n_iterations):
    theta, b = descent(x_test, y_test, theta, b, learning_rate)

b = b * 500
x_test = x_test * 500
y_test = y_test * 500

print(f'theta: {theta}, b: {b}')

theta_best[0] = b[0][0]
theta_best[1] = theta[0][0]


# TODO: calculate error

yhat = theta * x_test + b
error = np.mean((y_test - yhat) ** 2)
print('error:', error)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
