# 初始權重
A = np.random.normal(size=(5,4))
b = np.random.normal(size=(5,1))
C = np.random.normal(size=(3,5))
d = np.random.normal(size=(3,1))
# 開始計算
i = 9
x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
y = i%3
U = relu(A@x+b)
q = softmax(C@U+d)
L = - np.log(q[y])
print("before update, L=", L)