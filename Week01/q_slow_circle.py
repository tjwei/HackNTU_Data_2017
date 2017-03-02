img_array2 = img_array.copy()
for y in range(img_array2.shape[0]):
    for x in range(img_array.shape[1]):
        if (x-300)**2 + (y-300)**2 < 100*100:
            img_array2[y, x] = 0
show(img_array2)