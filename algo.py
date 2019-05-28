import numpy as np
import unittest
def gradient_descent(x, y, iteration, alpha=0.05, tolerance=1e-4):
    '''Use gradient descent to get parameter vector theta for linear regression
    Args:
        x: feature matrix with shape (m, n)
        y: target vector with shape (m, 1)
        theta: starting theta vector for gradient descent with shape (n+1, 1)
        alpha: learning rate for gradient descent
        iteration: number of max iterations
        tolerance: set threshold to determine when to converge 
                   based on old vs. new paramter changes at end of each iteration
    Returns:
        Parameter vector, array like'''
    N = x.shape[0]
    x = np.hstack((x, np.ones(N).reshape(N, 1)))
    theta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
    x_trans = x.transpose()
    for i in range(0, iteration):      
        hypothesis = np.dot(x, theta)
        error = hypothesis - y
        cost = np.sum(error ** 2) / (2 * len(x))
        gradient = np.dot(x_trans, error) / len(x)
        print("Iteration {} | Cost: {} | Gradient: {}".format(i, cost, gradient))
        # update theta
        theta_new = theta - alpha * gradient
        if sum(abs(theta_new - theta)).all() < tolerance:
            break
        theta = theta_new
    return theta_new


class MyTest(unittest.TestCase):
    def test_gradient_descent(self):
        theta_test = gradient_descent(x=np.array([[1, 1], [1, 2], [2, 2], [2, 3]]),
                                     y=np.array([[6], [8], [9], [10]]),
                                     iteration=10000)
        theta_test = np.round(theta_test, 2)
        self.assertEqual(np.array_equal(theta_test, np.array([[1.00], [1.50], [3.75]])), True)
unittest.main(exit=False)