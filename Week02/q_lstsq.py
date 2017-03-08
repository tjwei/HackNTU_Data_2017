# 常數項
# 產生隨機數據
X = np.random.normal(0, 3, size=(50,1))
one = np.ones_like(X)
X = np.concatenate([X, one], axis=1)
Y = X @ [3, 15] + np.random.normal(0, size=50)
# 用 numpy 的 lstsq
a = np.linalg.lstsq(X, Y)[0]
print("a=", a)
# 畫出來
plt.plot(X[:, 0], Y, 'o')
plt.plot(X[:, 0], X @ a, 'o');