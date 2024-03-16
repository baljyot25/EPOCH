import numpy as np
from sklearn.datasets import fetch_california_housing


def compute_cost(x, y, slope, intercept):
    cost = 0
    a = np.array([1, slope])
    b = np.array([0, intercept])
    a = a / np.linalg.norm(a)
    for i in range(len(x)):
        p = [x[i], y[i]]
        dist = (np.eye(2) - np.outer(a, a)) @ (p - b)
        cost += dist.T @ dist
    return cost


california = fetch_california_housing()

x = california.data[:, 6]
y = california.data[:, 7]

x_mean = np.mean(x)
y_mean = np.mean(y)

central_sigma_x = np.sum((x - x_mean) ** 2)
central_sigma_y = np.sum((y - y_mean) ** 2)
central_sigma_xy = np.sum((x - x_mean) * (y - y_mean))

slope1 = (central_sigma_y - central_sigma_x + np.sqrt((central_sigma_y - central_sigma_x) ** 2 + 4 * central_sigma_xy ** 2)) / (2 * central_sigma_xy)
intercept1 = y_mean - slope1 * x_mean

slope2 = (central_sigma_y - central_sigma_x - np.sqrt((central_sigma_y - central_sigma_x) ** 2 + 4 * central_sigma_xy ** 2)) / (2 * central_sigma_xy)
intercept2 = y_mean - slope2 * x_mean

cost1 = compute_cost(x, y, slope1, intercept1)
cost2 = compute_cost(x, y, slope2, intercept2)

# print("Slope 1:", slope1)
# print("Intercept 1:", intercept1)
# print("Cost 1:", cost1)
# print()
# print("Slope 2:", slope2)
# print("Intercept 2:", intercept2)
# print("Cost 2:", cost2)
# print()
if cost1 < cost2:
    print("The best line is y =", slope1, "x + ", intercept1, " with a cost of", cost1)
    a = [1, slope1]
    a /= np.linalg.norm(a)
    b = [0, intercept1]
    print("a:", a)
    print("b:", b)
else:
    print("The best line is y =", slope2, "x + ", intercept2, " with a cost of", cost2)
    a = [1, slope2]
    a /= np.linalg.norm(a)
    b = [0, intercept2]
    print("a:", a)
    print("b:", b)
