img2 = 0.299*img_array[:,:,0]+0.587*img_array[:,:,1]+0.114*img_array[:,:,2]
img2 = np.array(img2, dtype=np.uint8)
show(img2)