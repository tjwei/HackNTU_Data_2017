accuracy_history = []
γ = 0.03
A = np.random.normal(size=(10,9))
b = np.random.normal(size=(10,1))
C = np.random.normal(size=(3,10))
d = np.random.normal(size=(3,1))

for epochs in range(500):
    for i in range(512):
        x = np.array([[(i>>j)&1] for j in range(9)])
        y = truth(x)
        U = relu(A@x+b)
        q = softmax(C@U+d)
        L = - np.log(q[y])
        p = np.eye(3)[y][:, None]
        grad_d = q - p
        grad_C = grad_d @ U.T
        grad_b = (C.T @ grad_d ) * Drelu(A@x+b)
        grad_A = grad_b @ x.T
        A -= γ * grad_A
        b -= γ * grad_b
        C -= γ * grad_C
        d -= γ * grad_d
    score = 0
    for i in range(512):
        x = np.array([[(i>>j)&1] for j in range(9)])
        x = np.array([[(i>>j)&1] for j in range(9)])        
        U = relu(A@x+b)
        q = softmax(C@U+d)
        score += q.argmax() == truth(x)
    accuracy_history.append(score/512)
    if epochs%20==0:
        print(epochs, score/512)