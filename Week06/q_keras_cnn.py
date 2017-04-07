from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Reshape
model = Sequential()
model.add(Reshape((28, 28, 1), input_shape=(784,) ))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation="relu"))
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation="relu"))
model.add(MaxPool2D())
model.add(Reshape((-1,)))
model.add(Dense(units=1024, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
display(SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')))
model.fit(train_X, train_Y, validation_data=(validation_X, validation_Y), batch_size=128, epochs=15)
rtn = model.evaluate(test_X, test_Y)
print("\ntest accuracy=", rtn[1])