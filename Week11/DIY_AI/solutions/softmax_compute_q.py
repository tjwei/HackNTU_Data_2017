# Wx+b
c = W @ x + b

# d = exp(Wx+b)
d = np.exp(c)

# q = d/sum(d)
q = d/d.sum()
