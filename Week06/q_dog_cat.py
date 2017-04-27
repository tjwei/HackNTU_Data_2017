import keras
from keras.models import Sequential
from PIL import Image
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Reshape
import numpy as np
train_X=[]
for i in range(10000):
    img = Image.open("train/cat.%d.jpg"%i).resize((64,64))
    train_X.append(np.array(img))
    img = Image.open("train/dog.%d.jpg"%i).resize((64,64))
    train_X.append(np.array(img))
train_X=np.float32(train_X)/255

train_y=[0,1]*10000  
train_Y = np.eye(2)[train_y]

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation="relu", input_shape=(64,64,3)))
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation="relu"))
model.add(MaxPool2D())
model.add(Reshape((-1,)))
model.add(Dense(units=1024, activation="relu"))
model.add(Dense(units=2, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(train_X, train_Y, validation_split=0.1, batch_size=64, epochs=20)