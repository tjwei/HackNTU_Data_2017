from urllib.request import urlopen
url = "https://raw.githubusercontent.com/playcanvas/engine/master/examples/images/animation.png"
simg = Image.open(urlopen(url))
simg_array = np.array(simg)
print("原始大小")
show(simg_array)
print("放大兩倍")
simg_array_h2 = simg_array[[i//2 for i in range(2*simg_array.shape[0])]]
simg_array_w2 = simg_array_h2[:, [i//2 for i in range(2*simg_array.shape[1])]]
show(simg_array_w2)