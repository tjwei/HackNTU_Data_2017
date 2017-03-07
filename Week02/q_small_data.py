u = np.expand_dims(train_X[:1000], 1)
v = np.expand_dims(test_X[:2000], 0)
prediction = train_Y[((u - v)**2).sum(axis=2).argmin(axis=0)]
print((prediction == test_Y[:2000]).mean())