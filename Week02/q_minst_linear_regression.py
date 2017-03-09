# 另外一個用 numpy 設定 one-hot encoding 的方式
test_Y = np.eye(10)[test_y]
# 後面一樣
predict_y = np.argmax(regr.predict(test_X), axis=1)
print(np.mean(predict_y == test_y))