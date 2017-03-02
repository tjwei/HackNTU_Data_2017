x = np.arange(30)
a = np.arange(30)
a[1::2] = -a[1::2]
plt.plot(a)
plt.plot(x[::2], a[::2], 'x')
plt.plot(x[1::2], a[1::2], 'v')