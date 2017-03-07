u = train_X[0]
v = train_X[1]
print ( ((u - v)**2).sum() )
print ( np.linalg.norm(u-v)**2 )