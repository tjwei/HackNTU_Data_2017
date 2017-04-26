A = Matrix([0,0,0,0], 
           [1,-1,1,-1], 
           [-1,1,-1,1],
           [-10,10,-10,10],
           [10,-10,10,-10],
          )
b = Vector(0.1,0,0,-12,-12)
C = Matrix([1,0,0,0,0], 
           [0,1,0,1,0], 
           [0,0,1,0,1],
          )
d = Vector(0,0,0) 
for i in range(16):
    x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)
    q = softmax(C@relu(A@x+b)+d)
    print("i={}, i%3={}, q={}".format(i, i%3, q.argmax()))