import keras
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Reshape
model = Sequential()
model.add(Dense(units=10, input_dim=3072, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.fit(train_X, train_Y, validation_split=0.02, batch_size=128, epochs=50, verbose=2)
rtn = model.evaluate(test_X, test_Y)
print("\ntest accuracy=", rtn[1])