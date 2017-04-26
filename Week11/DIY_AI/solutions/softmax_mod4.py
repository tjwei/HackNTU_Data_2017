W = Matrix([-1,-1,0,0], [1,-1,0,0], [-1,1,0,0], [1,1,0,0])
b = Vector(0,0,0,0)
for i in range(16):
    x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
    r = W @ x + b
    print("i=", i, "predict:", r.argmax(), "ground truth:", i%4)
