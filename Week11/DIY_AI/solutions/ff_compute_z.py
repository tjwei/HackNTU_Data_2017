def relu(x):
    return np.maximum(x, 0)
def sigmoid(x):
    return 1/(1+np.exp(-x))
z_relu = relu(A@x + b)
z_sigmoid = sigmoid(A@x + b )