with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
    
train_X, train_y = train_set
test_X, test_y = test_set
def half_size(X):
    X = X[:, :, ::2]+X[:, :, 1::2]
    return (X[:, ::2, :]+X[:, 1::2, :])/4
train_X2 = half_size(train_X.reshape(-1,28,28)).reshape(-1, 14*14)
test_X2 = half_size(test_X.reshape(-1,28,28)).reshape(-1, 14*14)

clf = tree.DecisionTreeClassifier()
clf.fit(train_X2, train_y)
print("train:", np.mean(clf.predict(train_X2) == train_y))
print("test:", np.mean(clf.predict(test_X2) == test_y))