import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

data = np.load('data.npy')
target = np.load('target.npy')

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(data.shape[1:])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dropout(0.33))

model.add(Dense(32, activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size = 0.2)

checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor = 'val_loss', verbose=0, save_best_only=True, mode='auto')
history = model.fit(train_data, train_target, epochs=25, callbacks=[checkpoint], validation_split=0.2)