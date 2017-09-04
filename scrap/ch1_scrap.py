
import numpy as np
# import numpy.linalg as npla

inputs = np.identity(10)
modifier = 0.01*np.ones(10)
inputs = np.add(inputs, modifier)
modifier = 0.02*np.identity(10)
inputs = np.subtract(inputs, modifier)
outputs = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1]])
outputs = np.transpose(outputs)

print(outputs.shape)

# WX = B
# W = BX^{-1}

weights = np.matmul(outputs, np.linalg.inv(inputs))

print(weights)