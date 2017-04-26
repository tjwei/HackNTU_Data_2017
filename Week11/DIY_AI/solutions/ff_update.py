γ = 0.5
A -= γ * grad_A
b -= γ * grad_b
C -= γ * grad_C
d -= γ * grad_d

U = relu(A@x+b)
q = softmax(C@U+d)
L = - np.log(q[y])
print("after update, L=", L)