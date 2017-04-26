grad_W =   q @ x.T
grad_W[y] -= x.ravel()

# or 
grad_W = grad_b @ x.T
