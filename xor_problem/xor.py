import numpy as np

# Possible Inputs for XOR operation.
layer1 = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
target_output = np.array([[0, 1, 1, 0]]).T

#Random initialisation of weights
np.random.seed(1)
weights_1 = np.random.random((3, 2))
weights_2 = np.random.random((3, 1))
bias = np.ones((1, 4))

def sigmoid(g):
    return 1/(1 + np.exp(-g))

def sigmoid_gradient(g):
    return g*(1 - g)

for iter in range(100000):
    a2 = sigmoid(np.dot(layer1, weights_1))
    a2 = a2.T
    a2 = np.vstack((a2, bias)).T
    a3 = sigmoid(np.dot(a2, weights_2))

    a3_error = target_output - a3
    a2_error = np.dot(a3_error, weights_2[0:2, :].T)*sigmoid(np.dot(layer1, weights_1))

    a3_delta = a3_error*sigmoid_gradient(a3)
    a2_delta = a2_error*sigmoid_gradient(a2[:, 0:2])

    weights_2 += np.dot(a2.T, a3_delta)
    weights_1 += np.dot(layer1.T, a2_delta)


print("After training : ")
print(a3)

