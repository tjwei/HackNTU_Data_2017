test_X = np.linspace(-10,10, 100)[:, np.newaxis]
plt.plot(X, Y, 'o')
plt.plot(test_X, regr.predict(test_X));