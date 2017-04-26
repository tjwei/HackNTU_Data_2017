{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "%run magic.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Supervised learning for classification\n",
    "給一堆 $x$, 和他的分類，我們找出計算 x 的分類的方式\n",
    "\n",
    "### One hot encoding\n",
    "如果我們有三類種類別， 我們可以來編碼這三個類別\n",
    "* $(1,0,0)$\n",
    "* $(0,1,0)$\n",
    "* $(0,0,1)$\n",
    "\n",
    "### 問題\n",
    "* 為什麼不直接用 1,2,3 這樣的編碼呢？\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Softmax Regression 的模型是這樣的\n",
    "我們的輸入 $x=\\begin{pmatrix} x_0 \\\\ x_1 \\\\ x_2 \\\\ x_3 \\end{pmatrix} $ 是一個向量，我們看成 column vector 好了\n",
    "\n",
    "而 Weight: $W = \\begin{pmatrix} W_0 \\\\ W_1 \\\\ W_2 \\end{pmatrix} =  \n",
    "\\begin{pmatrix} W_{0,0} & W_{0,1} &  W_{0,2} & W_{0,3}\\\\ \n",
    " W_{1,0} & W_{1,1} &  W_{1,2} & W_{1,3} \\\\ \n",
    " W_{2,0} & W_{2,1} &  W_{2,2} & W_{2,3} \\end{pmatrix} $\n",
    " \n",
    " Bias: $b=\\begin{pmatrix} b_0 \\\\ b_1 \\\\ b_2 \\end{pmatrix} $ \n",
    "\n",
    "\n",
    "我們先計算\"線性輸出\"  $ c = \\begin{pmatrix} c_0 \\\\ c_1 \\\\ c_2 \\end{pmatrix} =  Wx+b =\n",
    "\\begin{pmatrix} W_0 x + b_0 \\\\ W_1 x + b_1 \\\\ W_2 x + b_2 \\end{pmatrix}   $， 然後再取 $exp$ (逐項取)。 最後得到一個向量。\n",
    " \n",
    " $d = \\begin{pmatrix} d_0 \\\\ d_1 \\\\ d_2 \\end{pmatrix} = e^{W x + b} = \\begin{pmatrix} e^{c_0} \\\\ e^{c_1} \\\\ e^{c_2} \\end{pmatrix}$\n",
    "\n",
    "\n",
    "將這些數值除以他們的總和。\n",
    "給定輸入 x， 我們希望算出來的數字 q_i 會符合 x 的類別是 i 的機率。\n",
    "\n",
    "###  $q_i = Predict_{W,b}(Y=i|x)  = \\frac {e^{W_i x + b_i}} {\\sum_j e^{W_j x + b_j}} = \\frac {d_i} {\\sum_j d_j}$\n",
    "\n",
    "### 合起來看，就是 $q = \\frac {d} {\\sum_j d_j} $\n",
    "\n",
    "### 問題\n",
    "* 為什麼要用 $exp$?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "先隨便算一個 $\\mathbb{R}^2 \\rightarrow \\mathbb{R}^3$ 的網路"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Weight\n",
    "W = Matrix([1,2],[3,4], [5,6])\n",
    "W"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Bias\n",
    "b = Vector(1,0,-1)\n",
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 輸入\n",
    "x = Vector(2,-1)\n",
    "x"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 任務：計算最後的猜測機率 $q$\n",
    "Hint: `np.exp` 可以算 $exp$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 參考答案\n",
    "#%load solutions/softmax_compute_q.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "%run -i solutions/softmax_compute_q.py\n",
    "# 顯示 q\n",
    "q"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 練習\n",
    "設計一個網路:\n",
    "* 輸入是二進位 0 ~ 15\n",
    "* 輸出依照對於 4 的餘數分成四類\n",
    "\n",
    "Hint: 可以參考上面 W, b 的設定方式"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Hint 下面產生數字 i 的 2 進位向量\n",
    "i = 13\n",
    "x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 參考答案\n",
    "#%load solutions/softmax_mod4.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 練習\n",
    "設計一個網路:\n",
    "* 輸入是二進位 0 ~ 15\n",
    "* 輸出依照對於 3 的餘數分成三類\n",
    "\n",
    "Hint: 不用全部正確，用猜的，但正確率要比亂猜高。可以利用統計的結果猜猜看。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 參考答案\n",
    "#%load solutions/softmax_mod3.py\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Gradient descent"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 誤差函數\n",
    "為了要評斷我們的預測的品質，要設計一個評斷誤差的方式\n",
    "\n",
    "假設輸入值 $x$ 對應到的真實類別是 $y$, 那我們定義誤差函數\n",
    "\n",
    "## $ loss = -\\log(q_y)=- \\log(Predict_{W,b}(Y=y|x)) $\n",
    "\n",
    "這個方法叫做 Cross entropy\n",
    "\n",
    "其實比較一般但比較複雜一點的寫法是\n",
    "\n",
    "## $ loss = - \\sum_i p_i\\log(q_i)  = -  p \\cdot \\log q$\n",
    "\n",
    "其中 $i$ 是所有類別， 而 $ p_i = \\Pr(Y=i|x) $ 是真實發生的機率\n",
    "\n",
    "但我們目前 $x$ 對應到的真實類別是 $y$， 所以直接 $p_i = 1$\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 想辦法改進。 \n",
    "我們用一種被稱作是 gradient descent 的方式來改善我們的誤差。\n",
    "\n",
    "因為我們知道 gradient 是讓函數上升最快的方向。所以我們如果朝 gradient 的反方向走一點點（也就是下降最快的方向），那麼得到的函數值應該會小一點。\n",
    "\n",
    "記得我們的變數是 $W$ 和 $b$ (裡面有一堆 W_i,j b_i 這些變數)，所以我們要把 $loss$ 對 $W$ 和 $b$ 裡面的每一個參數來偏微分。\n",
    "\n",
    "還好這個偏微分是可以用手算出他的形式，而最後偏微分的式子也不會很複雜。\n",
    "\n",
    "$loss$ 展開後可以寫成\n",
    "## $loss = -\\log(q_y) = \\log(\\sum_j d_j) - d_i \\\\\n",
    " = \\log(\\sum_j e^{W_j x + b_j}) - W_i x - b_i$\n",
    "\n",
    "注意 $d_j = e^{W_j x + b_j}$ 只有變數 $b_j, W_j$ \n",
    "\n",
    " 對 $k \\neq i$ 時, $loss$ 對 $b_k$ 的偏微分是 \n",
    " $$ \\frac{e^{W_k x + b_k}}{\\sum_j e^{W_j x + b_j}} = q_k$$\n",
    "對 $k = i$ 時, $loss$ 對 $b_k$ 的偏微分是 \n",
    "$$ q_k - 1$$\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "對 $W$ 的偏微分也不難\n",
    "\n",
    " 對 $k \\neq i$ 時, $loss$ 對 $W_{k,t}$ 的偏微分是 \n",
    " $$ \\frac{e^{W_k x + b_k}  x_t}{\\sum_j e^{W_j x + b_j}} = q_k x_t$$\n",
    "對 $k = i$ 時, $loss$ 對 $W_{k,t}$ 的偏微分是 \n",
    "$$ q_k x_t - x_t$$\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 實做部份"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 先產生隨機的 W 和 b\n",
    "W = Matrix(np.random.normal(size=(3,4)))\n",
    "b = Vector(np.random.normal(size=(3,)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "W"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 問題\n",
    "W, b 的 size 為什麼要這樣設定？"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 任務： 隨便設定一組 x, y, 我們來跑跑看 gradient descent"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "i = 14\n",
    "x = Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2)\n",
    "y = i%3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 步驟：計算 q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 參考答案(跟前面一樣)¶ \n",
    "#%load solutions/softmax_compute_q.py\n",
    "%run -i solutions/softmax_compute_q.py\n",
    "#顯示 q\n",
    "q"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 步驟： 計算 loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算 \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 參考答案(跟前面一樣)\n",
    "%run -i solutions/softmax_compute_loss1.py\n",
    "#顯示 loss\n",
    "loss"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 步驟：計算對 b 的 gradient"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算 grad_b\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#參考答案\n",
    "%run -i  solutions/softmax_compute_grad_b.py\n",
    "grad_b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 步驟：計算對 W 的 gradient"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#參考答案\n",
    "%run -i  solutions/softmax_compute_grad_W.py\n",
    "grad_W\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 步驟：更新 W, b  各減掉 0.5 * gradient， 然後看看新的 loss 是否有進步了？"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 參考答案\n",
    "%run -i solutions/softmax_update_Wb.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 原先的 q\n",
    "q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 原先的 loss\n",
    "loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 現在的 loss\n",
    "%run -i solutions/softmax_compute_q.py\n",
    "%run -i solutions/softmax_compute_loss1.py\n",
    "loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "q"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 一次訓練多組資料\n",
    "上面只針對一組 x (i=14) 來訓練，如果一次對所有 x 訓練呢？\n",
    "\n",
    "通常我們會把組別放在 axis-0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "X = np.array([Vector(i%2, (i>>1)%2, (i>>2)%2, (i>>3)%2) for i in range(16)])\n",
    "for i in range(4):\n",
    "    print(\"i=\", i)\n",
    "    display(X[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 對應的組別 \n",
    "y = np.array([i%3 for i in range(16)])\n",
    "y"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 任務： 將訓練向量化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 請在這裡計算\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# 參考解答如後"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "對照\n",
    "```python\n",
    "d = np.exp(W @ x + b)\n",
    "q = d/d.sum()\n",
    "q\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d = np.exp(W @ X + b)\n",
    "q = d/d.sum(axis=(1,2), keepdims=True)\n",
    "q"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "對照\n",
    "```python\n",
    "loss = -np.log(q[y])\n",
    "loss\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "loss = -np.log(q[range(len(y)), y])\n",
    "loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 用平均當成我們真正的 loss\n",
    "loss.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "對照\n",
    "```python\n",
    "grad_b = q - np.eye(3)[y][:, None]\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# fancy indexing :p\n",
    "one_y = np.eye(3)[y][..., None]\n",
    "grad_b_all = q - one_y\n",
    "grad_b = grad_b_all.mean(axis=0)\n",
    "grad_b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "對照\n",
    "```python\n",
    "grad_W = grad_b @ x.T\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "grad_W_all = grad_b_all @ X.swapaxes(1,2)\n",
    "grad_W = grad_W_all.mean(axis=0)\n",
    "grad_W"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "W -=  0.5 * grad_W\n",
    "b -=  0.5 * grad_b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 之前的 loss\n",
    "loss.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "d = np.exp(W @ X + b)\n",
    "q = d/d.sum(axis=(1,2), keepdims=True)\n",
    "loss = -np.log(q[range(len(y)), y])\n",
    "loss.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "### 任務：全部合在一起\n",
    "* 設定 W,b\n",
    "* 設定 X\n",
    "* 訓練三十次\n",
    "    * 計算 q 和 loss    \n",
    "    * 計算 grad_b 和 grad_W\n",
    "    * 更新 W, b\n",
    "* 看看準確度"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 在這裡計算\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 參考答案\n",
    "%run -i solutions/softmax_train.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 畫出 loss 的曲線\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "plt.plot(loss_history);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 對答案\n",
    "display((W @ X + b).argmax(axis=1).ravel())\n",
    "display(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
