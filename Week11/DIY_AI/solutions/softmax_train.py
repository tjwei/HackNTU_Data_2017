W = Matrix(np.random.normal(size=(3,4)))
b = Vector(np.random.normal(size=(3,)))
X = np.array([Vector(*[(i>>j)%2 for j in range(4)]) for i in range(16)])
y = np.array([i%3 for i in range(16)])
one_y = np.eye(3)[y][..., None]
# 紀錄 loss
loss_history = []
for epoch in range(50):
    d = np.exp(W @ X + b)
    q = d/d.sum(axis=(1,2), keepdims=True)
    loss = -np.log(q[range(len(y)), y]).mean()
    loss_history.append(loss)
    accuracy = (q.argmax(axis=1).ravel() == y).mean()
    print(epoch, accuracy)
    grad_b_all = q - one_y
    grad_b = grad_b_all.mean(axis=0)
    grad_W_all = grad_b_all @ X.swapaxes(1,2)
    grad_W = grad_W_all.mean(axis=0)
    W -=  grad_W
    b -= grad_b    
