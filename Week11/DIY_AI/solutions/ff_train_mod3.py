L_history = []
γ = 0.03
A = np.random.normal(size=(10,4))
b = np.random.normal(size=(10,1))
C = np.random.normal(size=(3,10))
d = np.random.normal(size=(3,1))
for t in range(20000):
    i = np.random.randint(0,16)
    x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
    y = i%3
    U = relu(A@x+b)
    q = softmax(C@U+d)
    L = - np.log(q[y])
    L_history.append(L)
    p = np.eye(3)[y][:, None]
    grad_d = q - p
    grad_C = grad_d @ U.T
    grad_b = (C.T @ grad_d ) * Drelu(A@x+b)
    grad_A = grad_b @ x.T
    A -= γ * grad_A
    b -= γ * grad_b
    C -= γ * grad_C
    d -= γ * grad_d
