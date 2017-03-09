N = 10
correct = 0
for i in range(N):
    # 顯示 test_X[i]
    print("test_X[{}]".format(i))
    showX(test_X[i])
    # 計算方差
    _ = ((train_X - test_X[i])**2).sum(axis=1)
    # 找出方差最小的 index
    idx = _.argmin()
    print("train_X[{}]".format(idx))
    showX(train_X[idx])
    print("train_X[{}] = {}".format(idx, train_y[idx]))
    print("train_X[{}] = {}".format(i, test_y[i]))
    if train_y[idx] == test_y[i]:
        correct+=1
print("Accuracy", correct/N)
