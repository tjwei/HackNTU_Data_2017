img_array2 = img_array.copy()
x, y = np.indices(img_array2.shape[:2])
idx = (x - 300)**2 + (y - 300)**2 < 100**2
img_array2[idx] = [200,100,0, 255]
show(img_array2)