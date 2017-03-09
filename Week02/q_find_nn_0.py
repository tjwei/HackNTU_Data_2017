# 顯示 test_X[0]
showX(test_X[0])
# 計算方差
_ = ((train_X - test_X[0])**2).sum(axis=1)
# 找出方差最小的 index
idx = _.argmin()
print("train_X[{}]".format(idx))
showX(train_X[idx])
print("train_X[{}] = {}".format(idx, train_y[idx]))
print("test_y[0] =", test_y[0])
