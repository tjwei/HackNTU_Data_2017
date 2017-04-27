import numpy as np

# 先寫一個正常的程式，來生成 Game of Life 的下一個狀態，用來檢查
def game(board):
    board_pad = np.pad(board, 1, 'constant', constant_values = 0)
    # 用比較笨的方式，厲害一點用 http://stackoverflow.com/questions/32660953/numpy-sliding-2d-window-calculations
    rtn = np.zeros_like(board)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            alive = board[i, j]
            neighbord_count = board_pad[i:i+3, j:j+3].sum() - alive
            if (alive and neighbord_count == 2) or neighbord_count==3:
                rtn[i,j] = 1
    return rtn

# 下面來定義 CNN 網路
from keras.models import Model
from keras.layers import Conv2D, Input

# 權重
def W(size):    
    rtn = np.ones(shape=(3,3,1,4))
    rtn[1,1,0,2:] = 10
    return rtn

def b(size):    
    return np.array([-2,-3, -12,-13])

def W2(size):
    return np.array([1,-2,1,-2]).reshape(1,1,4,1)

def b2(size):
    # just to be safe
    return np.full(size, -0.5)

# 網路模型定義
inputs = Input(shape=(None,None,1))
hidden = Conv2D(filters=4, kernel_size=3, padding='same', activation="relu",
             kernel_initializer=W, bias_initializer=b)(inputs)
out = Conv2D(filters=1, kernel_size=1, padding='same', activation="relu",
             kernel_initializer=W2, bias_initializer=b2)(hidden)
model = Model(inputs, out)

# 檢查看看結果是否正確
N = 10
# 隨機 100x100 盤面
boards = np.random.randint(0,2, size=(N,100,100))
# 用 CNN 模型跑下個盤面
rtn = model.predict(boards[..., None])
# >0 的值當成活著， <0 的值當成死的 (應該不會有 0的值)
rtn = (rtn>0).astype('int')
# 一一檢查
for i in range(N):
    b = game(boards[i])
    assert (b == rtn[i, :, :, 0]).all()
    print("OK", i)