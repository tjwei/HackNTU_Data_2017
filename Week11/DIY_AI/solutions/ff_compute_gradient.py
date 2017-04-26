p = np.eye(3)[y][:, None]
grad_d = q - p
grad_C = grad_d @ U.T
grad_b = (C.T @ grad_d ) * Drelu(A@x+b)
grad_A = grad_b @ x.T