u = np.expand_dims(train_X[:500], 1)
v = np.expand_dims(test_X[:100], 0)
prediction = train_y[((u - v)**2).sum(axis=2).argmin(axis=0)]
print((prediction == test_y[:100]).mean())
