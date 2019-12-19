import numpy as np
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, 'b.')
plt.show()

X_b = np.c_[np.ones((100, 1)), X]  # adding x0=1 to each instances
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)


x_new = np.array([[0], [2]])
X_b_new = np.c_[np.ones((2, 1)), x_new]
y_predict = X_b_new.dot(theta_best)
print("Prediction")
print(y_predict)

plt.plot(x_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()
