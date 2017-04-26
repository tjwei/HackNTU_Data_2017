def softmax(x):
    t = np.exp(x)
    return t/t.sum()
q_relu = softmax(C @ relu(A @ x + b) + d)
q_sigmoid = softmax(C @ sigmoid(A @ x + b) + d)