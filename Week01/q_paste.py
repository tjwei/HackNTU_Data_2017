simg_array = np.array(simg)
img_array2 = img_array.copy()
print("簡單的")
img_array2[200:400, 300:500] = simg_array
show(img_array2)
print("這樣呢？")
img_array2 = img_array.copy()
img_array2[200:400, 300:500][simg_array!=3] = simg_array[simg_array!=3]
show(img_array2)