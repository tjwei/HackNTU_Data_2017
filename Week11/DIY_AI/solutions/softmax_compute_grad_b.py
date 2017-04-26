grad_b = q.copy()
grad_b[y] -= 1
# or
grad_b = q - np.eye(3)[y][:, None]
