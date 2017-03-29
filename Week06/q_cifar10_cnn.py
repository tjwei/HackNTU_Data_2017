import keras
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Reshape
model = Sequential()
model.add(Reshape((3, 32, 32), input_shape=(3*32*32,) ))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation="relu", data_format='channels_first'))
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation="relu", data_format='channels_first'))
model.add(MaxPool2D())
model.add(Reshape((-1,)))
model.add(Dense(units=1024, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(train_X, train_Y, validation_split=0.02, batch_size=128, epochs=30)
rtn = model.evaluate(test_X, test_Y)
print("\ntest accuracy=", rtn[1])