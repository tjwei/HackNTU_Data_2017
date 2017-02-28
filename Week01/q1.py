from PIL import Image
img = Image.open('img/Green-Rolling-Hills-Landscape-800px.png')
img_array = np.array(img)
print("img_array.shape={}".format(img_array.shape))
print("img_array.dtype={}".format(img_array.dtype))
Image.fromarray(img_array)